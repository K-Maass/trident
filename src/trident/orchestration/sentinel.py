"""SENTINEL orchestration entry points (doc 04 / config/schedule.yaml).

These are the callable targets the Phase-6 scheduler names (`agents.run_sentinel`,
`agents.sentinel_monitor`). They wire the pure pipeline pieces to the generic runner:

  run_sentinel      08:30 ET daily — ensure new items are relevance-judged + FinBERT-scored,
                    (re)persist the trailing sentiment_aggregate signals, then run the agent
                    through the money path with extras={signals, holding_days}.
  sentinel_monitor  hourly 10:00–15:00 ET — held names only; queue an exit for any that has
                    flipped to a knife (sent_z < -2). No new entries.

`holding_days` (how long each name has been continuously held) is computed HERE from the fills
ledger, not in the pure agent — agents never touch the DB.

NOTE: nothing here is on a live schedule yet. Phase 6 wires dispatch; until then these run only
when called explicitly (a dry-run script, or a manual invocation). The trading agent is NOT
deployed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from trident.agents.sentiment import sentinel_agent
from trident.agents.sentiment.agent import ATTN_Z_ENTER, KNIFE_Z, SENT_Z_ENTER
from trident.agents.sentiment.aggregate import (
    PANEL_LOOKBACK_DAYS,
    Z_CLIP,
    aggregate,
    latest_aggregates,
)
from trident.agents.sentiment.relevance import run_relevance_backfill
from trident.agents.sentiment.score import run_score_backfill
from trident.broker.base import Broker
from trident.config import load_universe
from trident.data import calendar as trading_calendar
from trident.data import store
from trident.db import get_engine
from trident.models import AgentName, HealthEvent, HealthSeverity
from trident.orchestration import runner

# PANEL_LOOKBACK_DAYS now lives in aggregate.py (its natural home) and is imported above.


async def log_run_diagnostics(signals: dict[str, dict]) -> dict:
    """Pre-registered NEAR-MISS instrumentation (2026-07-21 deploy) + folded-in report fixes,
    as pure observability — it never influences a decision. Writes ONE `sentinel_nearmiss` INFO
    health event per run carrying:
      • near-misses — every ticker clearing EXACTLY ONE entry gate (sent_z>1.5 XOR attention_z>1.0),
        with which gate and BOTH z-values. This is the data that answers whether sent_z and
        attention_z spikes land on the same day; if joint fires stay zero, it decides the v2
        AND-gate question walk-forward (never re-tuned to this sample).
      • source-mix — how many tickers carry rss / apewisdom vs finnhub-only this session.
      • attention_z sanity — how many rows pin the ±5 clip (its near-zero-baseline volatility).
    Returns the computed summary (also handy for the caller's log).

    TODO (non-blocking):
      1. Persist near-misses as STRUCTURED rows (a table keyed by session+ticker with
         gate/sent_z/attn_z/n_items), not one text INFO event — the signals table already has the
         raw z's, so this is really "materialise the near-miss view" for easy walk-forward queries.
      2. Drop the [:8] truncation in the persisted copy (fine for the alert, lossy for the record).
      3. Move this daily diagnostic OUT of health_events (it's analytics, not an operational event).
    """
    sent_only, attn_only = [], []
    src_rss = src_ape = clip_pins = both = 0
    for tk, s in signals.items():
        sz, az = s.get("sent_z"), s.get("attention_z")
        srcs = s.get("sources_used") or []
        if "rss" in srcs:
            src_rss += 1
        if s.get("apewisdom_rank") is not None:
            src_ape += 1
        if az is not None and abs(az) >= Z_CLIP:
            clip_pins += 1
        if sz is None or az is None:
            continue
        sz_hit, az_hit = sz > SENT_Z_ENTER, az > ATTN_Z_ENTER
        if sz_hit and az_hit:
            both += 1
        elif sz_hit and not az_hit:
            sent_only.append(f"{tk}(sz={sz:.2f},az={az:.2f})")
        elif az_hit and not sz_hit:
            attn_only.append(f"{tk}(sz={sz:.2f},az={az:.2f})")
    summary = {
        "joint_fires": both,
        "sent_only": sent_only,
        "attn_only": attn_only,
        "src_rss": src_rss,
        "src_ape": src_ape,
        "clip_pins": clip_pins,
        "n": len(signals),
    }
    detail = (
        f"joint_fires={both}; sent_only({len(sent_only)})={sent_only[:8]}; "
        f"attn_only({len(attn_only)})={attn_only[:8]}; source_mix: rss={src_rss}/ape={src_ape} "
        f"of {len(signals)}; attention_z clip-pins={clip_pins}"
    )
    await store.write_health_event(
        HealthEvent(
            ts=datetime.now(UTC),
            severity=HealthSeverity.INFO,
            source="sentinel_nearmiss",
            detail=detail,
        )
    )
    return summary


async def _holding_days(agent: AgentName, asof: datetime) -> dict[str, int]:
    """Trading days each currently-held name has been continuously held (entry day = 0),
    reconstructed from the fills event log."""
    async with get_engine().connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT ticker, side, qty, ts FROM fills WHERE agent_id=:a ORDER BY ts, id"),
                {"a": agent.value},
            )
        ).all()
    running: dict[str, float] = {}
    start_ts: dict[str, datetime] = {}
    for r in rows:
        signed = float(r.qty) * (1.0 if r.side == "buy" else -1.0)
        prev = running.get(r.ticker, 0.0)
        new = prev + signed
        if prev <= 0 < new:
            start_ts[r.ticker] = r.ts
        if new <= 0:
            start_ts.pop(r.ticker, None)
        running[r.ticker] = new
    out: dict[str, int] = {}
    for tk, ts in start_ts.items():
        if running.get(tk, 0.0) > 0:
            sess = trading_calendar.sessions(ts.date(), asof.date())
            out[tk] = max(0, len(sess) - 1)
    return out


async def ensure_scored(since: datetime | None = None) -> tuple:
    """Relevance-judge then FinBERT-score any new items (both idempotent / cache-backed)."""
    rel = await run_relevance_backfill(since=since)
    sc = await run_score_backfill(since=since)
    return rel, sc


async def run_sentinel(
    broker: Broker,
    dry_run: bool = True,
    asof: datetime | None = None,
    score: bool = True,
):
    """Daily SENTINEL cycle. `score=True` refreshes relevance/FinBERT/aggregates first (the
    self-contained path used by the dry-run); the scheduled deployment will rely on the
    collector having already scored, and can pass score=False."""
    asof = asof or datetime.now(UTC)
    universe = load_universe()
    target = trading_calendar.previous_session(asof.date())
    if score:
        await ensure_scored()
        await aggregate(
            universe, target - timedelta(days=PANEL_LOOKBACK_DAYS), target, persist=True
        )
    signals = await latest_aggregates(universe, asof)
    try:
        await log_run_diagnostics(signals)  # near-miss instrumentation — observability only
    except Exception as e:  # noqa: BLE001 — diagnostics must NEVER block a trading run (risk-officer)
        print(f"WARN: sentinel diagnostics degraded ({type(e).__name__}: {e})")
    holding_days = await _holding_days(AgentName.SENTINEL, asof)
    return await runner.run_agent(
        sentinel_agent(),
        broker,
        dry_run=dry_run,
        asof=asof,
        extras={"signals": signals, "holding_days": holding_days},
    )


async def sentinel_monitor(broker: Broker, dry_run: bool = True, asof: datetime | None = None):
    """Intraday monitor: exit any HELD name that has flipped to a knife (sent_z < -2). Passes the
    agent only the knife-struck held names, so it emits exits and never new entries."""
    from trident import persistence

    asof = asof or datetime.now(UTC)
    book = await persistence.load_book(AgentName.SENTINEL)
    held = [p.ticker for p in book.snapshot(asof).positions if p.qty > 0]
    if not held:
        return None
    universe = load_universe()
    signals = await latest_aggregates(universe, asof)
    knife = {
        tk: signals[tk]
        for tk in held
        if signals.get(tk, {}).get("sent_z") is not None and signals[tk]["sent_z"] < KNIFE_Z
    }
    if not knife:
        return None
    return await runner.run_agent(
        sentinel_agent(),
        broker,
        dry_run=dry_run,
        asof=asof,
        extras={"signals": knife, "holding_days": {}},
    )
