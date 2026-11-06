"""Ledger accounting — the crown jewels (golden rule 5). Every number here is
hand-computed in the test, not derived from the code under test."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trident.ledger.book import STARTING_CASH, Book, LedgerError
from trident.models import AgentName, Fill, Side

TS = datetime(2026, 7, 3, 15, 0, tzinfo=UTC)


def _fill(ticker, side, qty, price, fees=0.0, agent=AgentName.QUANT) -> Fill:
    return Fill(
        order_id=f"o-{ticker}",
        agent=agent,
        ticker=ticker,
        side=side,
        qty=qty,
        price=price,
        ts=TS,
        fees=fees,
    )


# ------------------------------------------------------------------- setup
def test_starting_cash_defaults_match_schema():
    assert Book.new(AgentName.QUANT).cash == Decimal("30000")
    assert Book.new(AgentName.SENTINEL).cash == Decimal("30000")
    assert Book.new(AgentName.ORACLE).cash == Decimal("30000")
    assert Book.new(AgentName.BENCH).cash == Decimal("10000")
    assert STARTING_CASH[AgentName.BENCH] == Decimal("10000")


# -------------------------------------------------------------------- buys
def test_single_buy_debits_cash_and_opens_position():
    b = Book.new(AgentName.QUANT)  # 30000
    realized = b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))  # cost 2500
    assert realized == Decimal("0")
    assert b.cash == Decimal("27500")
    lot = b.positions["AAPL"]
    assert lot.qty == Decimal("10")
    assert lot.avg_cost == Decimal("250")


def test_second_buy_weights_average_cost():
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))  # cash 27500, avg 250
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 270))  # cost 2700 -> cash 24800
    lot = b.positions["AAPL"]
    assert b.cash == Decimal("24800")
    assert lot.qty == Decimal("20")
    # (10*250 + 10*270) / 20 = 5200/20 = 260
    assert lot.avg_cost == Decimal("260")


# ------------------------------------------------------------------- sells
def test_partial_sell_realizes_pnl_and_keeps_basis():
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))  # cash 27500, avg 250
    realized = b.apply_fill(_fill("AAPL", Side.SELL, 4, 300))  # proceeds 1200
    assert realized == Decimal("200")  # 4 * (300 - 250)
    assert b.realized_pnl == Decimal("200")
    assert b.cash == Decimal("28700")  # 27500 + 1200
    lot = b.positions["AAPL"]
    assert lot.qty == Decimal("6")
    assert lot.avg_cost == Decimal("250")  # basis unchanged on a sale


def test_full_sell_closes_position_and_cash_reconciles():
    b = Book.new(AgentName.QUANT)  # 30000
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))
    realized = b.apply_fill(_fill("AAPL", Side.SELL, 10, 300))
    assert realized == Decimal("500")  # 10 * 50
    assert "AAPL" not in b.positions
    assert b.cash == Decimal("30500")
    # invariant: flat again => cash == start + realized
    assert b.cash == STARTING_CASH[AgentName.QUANT] + b.realized_pnl


def test_oversell_and_naked_sell_raise():
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))
    with pytest.raises(LedgerError):
        b.apply_fill(_fill("AAPL", Side.SELL, 11, 300))  # more than held
    with pytest.raises(LedgerError):
        b.apply_fill(_fill("MSFT", Side.SELL, 1, 100))  # not held at all


def test_buy_after_partial_sell_reaverages_on_reduced_qty():
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))  # qty 10, avg 250, cash 27500
    b.apply_fill(_fill("AAPL", Side.SELL, 4, 300))  # qty 6, avg 250, cash 28700, realized 200
    b.apply_fill(_fill("AAPL", Side.BUY, 6, 260))  # cost 1560 -> cash 27140
    lot = b.positions["AAPL"]
    assert lot.qty == Decimal("12")
    # (6*250 + 6*260) / 12 = (1500 + 1560) / 12 = 3060/12 = 255
    assert lot.avg_cost == Decimal("255")
    assert b.cash == Decimal("27140")
    assert b.realized_pnl == Decimal("200")  # unchanged by the re-buy


# --------------------------------------------------------------- fees & P&L
def test_fees_fold_into_basis_and_reduce_proceeds():
    b = Book.new(AgentName.QUANT)  # 30000
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250, fees=5))  # cost 2505 -> cash 27495
    assert b.cash == Decimal("27495")
    assert b.positions["AAPL"].avg_cost == Decimal("250.5")  # 2505/10
    realized = b.apply_fill(_fill("AAPL", Side.SELL, 10, 300, fees=5))  # proceeds 2995
    # 10*(300 - 250.5) - 5 = 495 - 5 = 490
    assert realized == Decimal("490")
    assert b.cash == Decimal("30490")
    assert b.cash == STARTING_CASH[AgentName.QUANT] + b.realized_pnl  # fee-aware round trip


def test_flat_book_invariant_within_tolerance_on_repeating_basis():
    # With a repeating-decimal average cost, cash == start + realized holds within the money
    # tolerance (avg_cost quantized to 1e-6/share), not to exact Decimal equality. Documents
    # the real guarantee so a future refactor can't silently rely on exact equality.
    b = Book.new(AgentName.QUANT)
    start = b.cash
    b.apply_fill(_fill("AAPL", Side.BUY, 3, 100.0))  # avg 100
    b.apply_fill(_fill("AAPL", Side.BUY, 4, 33.333333))  # avg becomes 61.904762 (repeating)
    b.apply_fill(_fill("AAPL", Side.SELL, 7, 90.0))
    assert "AAPL" not in b.positions
    drift = abs(b.cash - (start + b.realized_pnl))
    assert drift < Decimal("0.01")  # ~2e-6 here; bounded by shares * 1e-6, << $1 recon tolerance


# ----------------------------------------------------------------- dividends
def test_dividend_credits_holder_and_rejects_unheld():
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))  # cash 27500
    b.apply_dividend("AAPL", 23.50)  # 10 shares * $2.35
    assert b.cash == Decimal("27523.5")
    with pytest.raises(LedgerError):
        b.apply_dividend("MSFT", 10.0)  # not held


# --------------------------------------------------------------------- marks
def test_mark_to_market_moves_equity_not_cash():
    b = Book.new(AgentName.QUANT)  # 30000
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))  # cash 27500
    assert b.equity() == Decimal("30000")  # unmarked -> valued at cost, equity flat post-buy
    b.mark_to_market({"AAPL": 300})
    assert b.cash == Decimal("27500")  # marks never touch cash
    assert b.equity() == Decimal("30500")  # 27500 + 10*300


def test_unpriced_name_keeps_prior_mark():
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))
    b.mark_to_market({"AAPL": 300})
    b.mark_to_market({"MSFT": 999})  # AAPL absent -> keeps 300, no invented price
    assert b.equity() == Decimal("30500")


def test_multi_ticker_equity_and_weights():
    b = Book.new(AgentName.QUANT)  # 30000
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))  # -2500
    b.apply_fill(_fill("MSFT", Side.BUY, 20, 100))  # -2000 -> cash 25500
    b.mark_to_market({"AAPL": 260, "MSFT": 110})
    # 25500 + 10*260 + 20*110 = 25500 + 2600 + 2200 = 30300
    assert b.equity() == Decimal("30300")
    assert b.weight("AAPL") == (Decimal("2600") / Decimal("30300")).quantize(Decimal("0.000001"))
    assert b.weight("CASH_ONLY_MISS") == Decimal("0")


def test_ticker_case_is_normalized_across_ops():
    # Buy lowercase, mark/dividend/weight in other cases — all must agree (no silent miss).
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("aapl", Side.BUY, 10, 250))  # stored as AAPL
    assert "AAPL" in b.positions and "aapl" not in b.positions
    b.mark_to_market({"AApL": 300})  # would silently no-op before the fix
    assert b.equity() == Decimal("30500")  # mark actually applied
    b.apply_dividend("aApl", 10.0)  # holder recognized regardless of case
    assert b.cash == Decimal("27510")
    assert b.weight("aapl") == b.weight("AAPL")


# --------------------------------------------------------------------- BENCH
def test_bench_spy_buy_and_hold():
    b = Book.new(AgentName.BENCH)  # 10000
    b.apply_fill(_fill("SPY", Side.BUY, 25, 400, agent=AgentName.BENCH))  # 10000 -> cash 0
    assert b.cash == Decimal("0")
    b.mark_to_market({"SPY": 440})
    assert b.equity() == Decimal("11000")  # 25 * 440


# ------------------------------------------------------------------ snapshot
def test_snapshot_mirrors_book():
    b = Book.new(AgentName.QUANT)
    b.apply_fill(_fill("AAPL", Side.BUY, 10, 250))
    b.mark_to_market({"AAPL": 300})
    snap = b.snapshot(TS)
    assert snap.agent == AgentName.QUANT
    assert snap.cash == 27500.0
    assert len(snap.positions) == 1
    assert snap.positions[0].ticker == "AAPL"
    assert snap.positions[0].last_price == 300.0
    assert abs(snap.equity - float(b.equity())) < 1e-6
