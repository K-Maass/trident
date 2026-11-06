"""Daily-snapshot safety job — the violation tests the monthly cadence could not give us.

The point of the job is that the kill-switch fires on MARKS, between trades. So the load-bearing
test drives a mark-to-market drawdown past the limit and asserts the halt happens FROM THE
SNAPSHOT PATH (no run_agent, no order). Plus the idempotency guarantee: a second run the same
session writes nothing and does not re-ratchet the HWM.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trident.ledger.book import Book
from trident.models import AgentName, AgentStatus
from trident.orchestration import runner
from trident.orchestration import snapshot as S

ASOF = datetime(2026, 7, 17, 21, 0, tzinfo=UTC)  # a Friday, after the US close


class FakeBroker:
    """Serves whatever prices the test sets; that's how we manufacture a mark drawdown."""

    def __init__(self, prices):
        self._prices = prices

    async def get_latest_price(self, ticker):
        from trident.broker.base import BrokerDataError

        if ticker not in self._prices:
            raise BrokerDataError(f"no price for {ticker}")
        return self._prices[ticker]


@pytest.fixture
def wired(monkeypatch):
    """Stub the calendar (fixed session), persistence and health so the test is offline and we
    can observe the kill-switch + snapshot writes. Everything else is the REAL snapshot/_finalize
    /evaluate_kill_switch path."""
    state = {
        "status": {a: AgentStatus.ACTIVE for a in S.ALL_AGENTS},
        "hwm": {a: 0.0 for a in S.ALL_AGENTS},
        "books": {},
        "snapshots": [],  # (agent_id, session_date)
        "existing": set(),  # (agent_id, session_date) already present
        "health": [],
        "halted": [],
    }

    monkeypatch.setattr(S.calendar, "is_open", lambda asof=None: False)
    monkeypatch.setattr(S.calendar, "last_completed_session", lambda asof=None: ASOF.date())

    async def get_status(agent):
        return state["status"][agent]

    async def set_status(agent, status):
        state["status"][agent] = status
        if status == AgentStatus.HALTED:
            state["halted"].append(agent.value)

    async def snap_exists(aid, session):
        return (aid, session) in state["existing"]

    async def load_book(agent):
        return state["books"][agent]

    async def save_positions(book, ts):
        pass

    async def write_snap(agent, ts, equity, cash):
        state["snapshots"].append((agent.value, ts.date()))
        state["existing"].add((agent.value, ts.date()))  # now it exists (idempotency)

    async def update_hwm(agent, equity):
        state["hwm"][agent] = max(state["hwm"][agent], float(equity))
        return state["hwm"][agent]

    async def health(sev, src, detail):
        state["health"].append((sev, src, detail))

    monkeypatch.setattr(S.store, "equity_snapshot_exists", snap_exists)
    monkeypatch.setattr(S.persistence, "get_status", get_status)
    monkeypatch.setattr(S.persistence, "set_status", set_status)
    monkeypatch.setattr(S.persistence, "load_book", load_book)
    monkeypatch.setattr(runner.persistence, "get_status", get_status)
    monkeypatch.setattr(runner.persistence, "set_status", set_status)
    monkeypatch.setattr(runner.persistence, "save_positions", save_positions)
    monkeypatch.setattr(runner.persistence, "write_equity_snapshot", write_snap)
    monkeypatch.setattr(runner.persistence, "update_high_water_mark", update_hwm)
    monkeypatch.setattr(runner, "_health", health)
    return state


def _book_with(agent, ticker, qty, avg_cost, cash):
    book = Book.new(agent, cash)
    from trident.models import Fill, Side

    book.apply_fill(
        Fill(
            order_id=f"{agent.value}-x",
            agent=agent,
            ticker=ticker,
            side=Side.BUY,
            qty=qty,
            price=avg_cost,
            fees=0.0,
            ts=ASOF,
        )
    )
    return book


async def test_killswitch_fires_from_snapshot_path_on_mark_drawdown(wired):
    """Violation test 4a: an agent holding a name whose MARK falls >15% below the HWM must be
    halted by the snapshot job itself — no run_agent, no order."""
    agent = AgentName.QUANT
    # Bought 100 sh @ $100 with $0 cash left → book value $10,000 at cost.
    wired["books"] = {a: Book.new(a, 30_000) for a in S.ALL_AGENTS}
    wired["books"][agent] = _book_with(agent, "AAA", qty=100, avg_cost=100.0, cash=0.0)
    wired["hwm"][agent] = 10_000.0  # HWM set at cost basis

    # Mark AAA down to $80 → equity $8,000 = 20% drawdown (> 15% limit). Other agents priced flat.
    broker = FakeBroker({"AAA": 80.0})
    report = await S.snapshot_all(broker, asof=ASOF)

    assert agent.value in report.snapshotted  # it WAS snapshotted (equity recorded)
    assert agent.value in report.halted  # and halted...
    assert agent.value in wired["halted"]  # ...via the real set_status(HALTED) call
    # the halt produced an out-of-band alert for the caller to fire
    assert any("KILL-SWITCH" in s for s, _ in report.alerts)


async def test_snapshot_is_idempotent_no_dup_and_no_double_ratchet(wired):
    """Violation test 4b: re-running the same session writes no second snapshot and does not
    re-ratchet the HWM."""
    wired["books"] = {a: Book.new(a, 30_000) for a in S.ALL_AGENTS}
    broker = FakeBroker({})  # all agents flat/cash → no prices needed

    first = await S.snapshot_all(broker, asof=ASOF)
    assert set(first.snapshotted) == {a.value for a in S.ALL_AGENTS}
    hwm_after_first = dict(wired["hwm"])
    n_snaps = len(wired["snapshots"])

    second = await S.snapshot_all(broker, asof=ASOF)  # same session
    assert second.snapshotted == []  # nothing re-written
    assert set(second.skipped_existing) == {a.value for a in S.ALL_AGENTS}
    assert len(wired["snapshots"]) == n_snaps  # no duplicate rows
    assert wired["hwm"] == hwm_after_first  # HWM unchanged on the re-run


async def test_missing_price_skips_agent_and_alerts_not_guesses(wired):
    """Rider 2: a held ticker with no live price → skip that agent's snapshot + alert, never a
    stale mark."""
    agent = AgentName.QUANT
    wired["books"] = {a: Book.new(a, 30_000) for a in S.ALL_AGENTS}
    wired["books"][agent] = _book_with(agent, "AAA", qty=10, avg_cost=100.0, cash=0.0)

    broker = FakeBroker({})  # AAA price MISSING
    report = await S.snapshot_all(broker, asof=ASOF)

    assert agent.value not in report.snapshotted  # not snapshotted off a stale mark
    assert any(aid == agent.value for aid, _ in report.skipped_missing_price)
    assert any("SKIPPED" in s and agent.value in s for s, _ in report.alerts)
    assert (agent.value, ASOF.date()) not in wired["snapshots"]  # no row written


async def test_market_open_is_a_noop(wired, monkeypatch):
    """Never mark mid-session."""
    monkeypatch.setattr(S.calendar, "is_open", lambda asof=None: True)
    report = await S.snapshot_all(FakeBroker({}), asof=ASOF)
    assert report.ran is False and "market open" in report.reason
    assert wired["snapshots"] == []


async def test_non_session_is_a_noop_without_db(wired, monkeypatch):
    """Guard-order fix: a weekend/holiday fire must no-op WITHOUT any DB access (the wrapper skips
    Docker on the matching due-check). Booby-trap load_book so a DB touch would fail the test."""
    monkeypatch.setattr(S.calendar, "is_open", lambda asof=None: False)
    monkeypatch.setattr(S.calendar, "is_session", lambda d: False)  # not a trading session

    async def boom(_agent):
        raise AssertionError("snapshot_all must not touch the DB on a non-session fire")

    monkeypatch.setattr(S.persistence, "get_status", boom)
    report = await S.snapshot_all(FakeBroker({}), asof=ASOF)
    assert report.ran is False and "not a trading session" in report.reason
    assert wired["snapshots"] == []


def test_is_snapshot_due_matrix(monkeypatch):
    """due = post-close (not open) AND a trading session."""
    for is_open, is_sess, expected in [
        (False, True, True),  # trading day, post-close → due
        (True, True, False),  # trading day, intraday → not due
        (False, False, False),  # weekend/holiday → not due
        (True, False, False),  # (can't really happen) not due
    ]:
        monkeypatch.setattr(S.calendar, "is_open", lambda asof=None, v=is_open: v)
        monkeypatch.setattr(S.calendar, "is_session", lambda d, v=is_sess: v)
        assert S.is_snapshot_due(ASOF) is expected
