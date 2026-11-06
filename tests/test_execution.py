"""Execution sizing (pure, offline) + the kill-switch state machine (integration)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trident.execution.service import size_order
from trident.models import AgentName, BookSnapshot, Position, Side

TS = datetime(2026, 7, 6, 15, 0, tzinfo=UTC)


def _book(cash, positions=()):
    return BookSnapshot(agent=AgentName.QUANT, ts=TS, cash=cash, positions=list(positions))


# ------------------------------------------------------------------ sizing
def test_buy_sizes_whole_shares_from_weight_delta():
    book = _book(30_000)  # empty, equity 30k
    o = size_order(AgentName.QUANT, "d1", "AAPL", 0.02, book, price=300.0, limit_bps=50, asof=TS)
    assert o is not None
    assert o.side == Side.BUY
    assert o.qty == 2  # floor(0.02 * 30000 / 300) = floor(2.0)
    assert o.limit_price == round(300.0 * 1.005, 2)  # +50 bps


def test_zero_delta_produces_no_order():
    pos = Position(agent=AgentName.QUANT, ticker="AAPL", qty=2, avg_cost=300, last_price=300)
    book = _book(29_400, [pos])  # equity 30000, AAPL weight 0.02
    assert size_order(AgentName.QUANT, "d", "AAPL", 0.02, book, 300.0, 50, TS) is None


def test_sell_is_capped_at_shares_held_no_shorting():
    pos = Position(agent=AgentName.QUANT, ticker="AAPL", qty=3, avg_cost=300, last_price=300)
    book = _book(29_100, [pos])  # equity 30000, AAPL weight 0.03
    # target 0 -> sell everything, but never more than the 3 held
    o = size_order(AgentName.QUANT, "d", "AAPL", 0.0, book, 300.0, 50, TS)
    assert o is not None and o.side == Side.SELL
    assert o.qty == 3
    assert o.limit_price == round(300.0 * 0.995, 2)  # -50 bps


def test_sub_share_target_rounds_to_no_order():
    book = _book(30_000)
    # 0.5% of 30k = 150 < one 300 share -> floor 0 -> no order
    assert size_order(AgentName.QUANT, "d", "AAPL", 0.005, book, 300.0, 50, TS) is None


def test_zero_price_guarded():
    assert size_order(AgentName.QUANT, "d", "AAPL", 0.02, _book(30_000), 0.0, 50, TS) is None


# ---------------------------------------------------- kill-switch state machine
@pytest.mark.integration
async def test_kill_switch_halts_agent_on_drawdown():
    from sqlalchemy import text

    from trident import persistence
    from trident.data import store
    from trident.models import AgentStatus
    from trident.orchestration.runner import evaluate_kill_switch

    eng = store.get_engine()
    async with eng.begin() as c:  # arrange: active, HWM=100k so 84k equity is a -16% drawdown
        # belt-and-suspenders: never mutate a non-test DB
        _db = (await c.execute(text("SELECT current_database()"))).scalar_one()
        assert str(_db).endswith("_test"), f"refusing to mutate agents on non-test DB '{_db}'"
        await c.execute(
            text("UPDATE agents SET status='active', high_water_mark=100000 WHERE id='quant'")
        )
    try:
        halted = await evaluate_kill_switch(AgentName.QUANT, equity=84_000.0, asof=TS)
        assert halted is True
        assert await persistence.get_status(AgentName.QUANT) == AgentStatus.HALTED
        # a -14% drawdown on a fresh agent must NOT halt
        async with eng.begin() as c:
            await c.execute(
                text("UPDATE agents SET status='active', high_water_mark=100000 WHERE id='quant'")
            )
        assert await evaluate_kill_switch(AgentName.QUANT, equity=86_000.0, asof=TS) is False
        assert await persistence.get_status(AgentName.QUANT) == AgentStatus.ACTIVE
    finally:
        async with eng.begin() as c:  # reset so we don't leave quant halted / a CRITICAL freeze
            await c.execute(
                text("UPDATE agents SET status='active', high_water_mark=0 WHERE id='quant'")
            )
            await c.execute(
                text("DELETE FROM health_events WHERE source='kill_switch' AND ts=:ts"), {"ts": TS}
            )
        await store.dispose_engine()
