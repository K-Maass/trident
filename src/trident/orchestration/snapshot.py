"""Daily equity snapshot (doc 06) — the safety job that watches a live book between trades.

WHY THIS EXISTS. `equity_snapshots` and the kill-switch are written/evaluated only inside
`runner.run_agent`, which for `momentum_12_1` runs MONTHLY. Once QUANT holds a live book
(2026-07-15) that leaves two holes between rebalances: the equity curve goes flat for weeks,
and — the dangerous one — a drawdown that happens purely from position marks would NOT trip the
kill-switch until the next monthly run. This job closes both by marking every agent's book to
market once per session and running the exact same finalize + kill-switch path run_agent uses,
with NO agent call and NO orders.

Design (per the approved riders):
- Reuses `runner._finalize` + `runner.evaluate_kill_switch` verbatim — one kill-switch code
  path, never a divergent copy.
- All agents including BENCH, so the comparison curves stay aligned.
- Trading sessions only, and only after the close (`is_open` guard) — never mid-session marks.
- Idempotent: one snapshot per (agent, session). If a snapshot already exists for the session
  (this job ran, or run_agent did on a rebalance/deploy day), the agent is skipped — no
  duplicate row, no second HWM ratchet.
- Never guesses a price (golden rule 8): if ANY held ticker has no live price, that agent's
  snapshot is skipped with a WARN, not written with a stale mark.
- A firing kill-switch and every skip/failure are surfaced; the caller alerts out-of-band.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from trident import persistence
from trident.data import calendar, store
from trident.models import AgentName, AgentStatus, HealthSeverity
from trident.orchestration import runner

ALL_AGENTS = [AgentName.QUANT, AgentName.SENTINEL, AgentName.ORACLE, AgentName.BENCH]


def session_ts(session) -> datetime:
    """Canonical UTC timestamp for a session's snapshot — stable per session date so the
    (agent, ts) primary key dedupes re-runs. 20:00 UTC ≈ the US regular close; the exact minute
    is immaterial, stability is what matters."""
    return datetime(session.year, session.month, session.day, 20, 0, tzinfo=UTC)


def is_snapshot_due(asof: datetime | None = None) -> bool:
    """True only in the post-close window of a trading session — a pure calendar check, NO DB.
    False when the market is open (post-close job) or `asof`'s date is not a trading session
    (weekend/holiday). Lets the wrapper skip the Docker/DB precheck entirely on no-op fires. The
    UTC date is used deliberately: a post-close evening fire in any whole-hour-offset local tz
    still resolves to the correct session date (a 22:20 CEST Friday fire = 20:20 UTC Friday)."""
    asof = asof or datetime.now(UTC)
    return not calendar.is_open(asof) and calendar.is_session(asof.date())


@dataclass
class SnapshotReport:
    session: object = None
    ran: bool = True  # False when skipped wholesale (market open / not a session)
    reason: str = ""
    snapshotted: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    skipped_missing_price: list[tuple[str, list[str]]] = field(default_factory=list)
    halted: list[str] = field(default_factory=list)
    alerts: list[tuple[str, str]] = field(default_factory=list)  # (subject, body) for the caller

    def alert(self, subject: str, body: str) -> None:
        self.alerts.append((subject, body))


async def snapshot_all(broker, asof: datetime | None = None) -> SnapshotReport:
    """Mark every agent's book to market and record one equity snapshot for the latest completed
    session, running the kill-switch on each. Returns a report; the CALLER fires `report.alerts`
    out-of-band (this module cannot depend on the alert channel, and health_events can't report
    a DB outage anyway)."""
    asof = asof or datetime.now(UTC)

    # Calendar-only gate (no DB): a snapshot is due ONLY in the post-close window of a trading
    # day. This short-circuits BEFORE any DB access so weekend/holiday and intraday fires no-op
    # without needing Postgres (the wrapper runs the same check before its Docker precheck, so a
    # non-session fire never touches Docker either).
    if not is_snapshot_due(asof):
        why = "market open" if calendar.is_open(asof) else "not a trading session"
        return SnapshotReport(ran=False, reason=f"{why} — snapshot is a post-close-on-session job")

    session = calendar.last_completed_session(asof)
    ts = session_ts(session)
    report = SnapshotReport(session=session)

    for agent in ALL_AGENTS:
        aid = agent.value
        # Only ACTIVE agents. A HALTED agent is a manual-intervention state; re-running the
        # kill-switch on it daily would spam CRITICAL events + the global freeze.
        if await persistence.get_status(agent) != AgentStatus.ACTIVE:
            continue
        if await store.equity_snapshot_exists(aid, session):
            report.skipped_existing.append(aid)  # this job or run_agent already did this session
            continue

        book = await persistence.load_book(agent)
        held = [p.ticker for p in book.snapshot(ts).positions if p.qty != 0]
        prices = await runner._live_prices(broker, held)
        missing = [t for t in held if t not in prices]
        if missing:
            # golden rule 8: a missing mark means we cannot value the book honestly → skip, do
            # NOT write a snapshot off stale marks. Surface it loudly.
            await runner._health(
                HealthSeverity.WARN,
                f"snapshot.{aid}",
                f"no live price for {missing} — snapshot skipped for session {session}",
            )
            report.skipped_missing_price.append((aid, missing))
            report.alert(
                f"snapshot SKIPPED — {aid}",
                f"No live price for {', '.join(missing)}; {aid}'s equity was NOT snapshotted for "
                f"{session} (won't guess a mark). The equity curve will have a gap for this day.",
            )
            continue

        result = runner.RunResult(agent=agent, asof=ts, status="snapshot")
        await runner._finalize(agent, book, broker, ts, result, prices=prices)
        report.snapshotted.append(aid)
        if result.halted:
            report.halted.append(aid)
            report.alert(
                f"KILL-SWITCH FIRED — {aid}",
                f"{aid} halted from the daily-snapshot path: equity ${result.equity:,.2f} fell "
                f"below 85% of its high-water mark on marks alone (no trade). Flatten on the next "
                f"run; reactivation is manual only.",
            )

    return report
