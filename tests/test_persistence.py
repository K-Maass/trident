"""Money-path persistence tests. load_book event-sources the Book from fills — the single
source of truth for per-agent P&L — so it gets a known-fills -> expected-Book check.
Integration (needs Postgres); operates on QUANT and restores it in teardown."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import text

from trident import persistence
from trident.data import store
from trident.models import AgentName


async def _assert_test_db(conn):
    """Belt-and-suspenders: a destructive helper refuses to DELETE unless
    it is provably on a *_test database, even if conftest routing were somehow bypassed."""
    name = (await conn.execute(text("SELECT current_database()"))).scalar_one()
    assert str(name).endswith("_test"), f"refusing destructive reset on non-test DB '{name}'"


async def _reset_quant(conn):
    await _assert_test_db(conn)
    for t in ("fills", "orders", "decisions", "positions", "equity_snapshots"):
        await conn.execute(text(f"DELETE FROM {t} WHERE agent_id='quant'"))
    await conn.execute(
        text("UPDATE agents SET status='active', high_water_mark=0 WHERE id='quant'")
    )


@pytest.mark.integration
async def test_load_book_empty_is_start_cash():
    eng = store.get_engine()
    async with eng.begin() as c:
        await _reset_quant(c)
    try:
        book = await persistence.load_book(AgentName.QUANT)
        assert book.cash == Decimal("30000")
        assert book.positions == {}
        assert book.realized_pnl == Decimal("0")
    finally:
        await store.dispose_engine()


@pytest.mark.integration
async def test_load_book_replays_fills_in_order():
    eng = store.get_engine()
    base = datetime(2026, 7, 6, 14, 0, tzinfo=UTC)
    fills = [
        ("AAPL", "buy", 10, 250.0, 0.0, base),  # cash 27500, AAPL 10@250
        ("AAPL", "buy", 10, 270.0, 0.0, base.replace(minute=1)),  # cash 24800, AAPL 20@260
        ("AAPL", "sell", 5, 300.0, 0.0, base.replace(minute=2)),  # cash 26300, AAPL 15@260, pnl 200
    ]
    async with eng.begin() as c:
        await _reset_quant(c)
        for tkr, side, qty, price, fees, ts in fills:
            await c.execute(
                text(
                    "INSERT INTO fills (order_id, agent_id, ticker, side, qty, price, fees, ts) "
                    "VALUES ('t','quant',:t,:s,:q,:p,:f,:ts)"
                ),
                {"t": tkr, "s": side, "q": qty, "p": price, "f": fees, "ts": ts},
            )
    try:
        book = await persistence.load_book(AgentName.QUANT)
        assert book.cash == Decimal("26300")
        assert book.realized_pnl == Decimal("200")  # 5 * (300 - 260)
        lot = book.positions["AAPL"]
        assert lot.qty == Decimal("15")
        assert lot.avg_cost == Decimal("260")  # (10*250 + 10*270)/20, unchanged by the sell
    finally:
        async with eng.begin() as c:
            await _reset_quant(c)
        await store.dispose_engine()
