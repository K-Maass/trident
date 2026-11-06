"""Backtester tests — two permanent tripwires (offline, synthetic bars):
  1. look-ahead: `_LaggedPrices` can never return a bar dated >= session T.
  2. golden-file: one hand-computed equity curve; any driver/fill-model change fails loud.
Synthetic bars are passed to the backtester directly."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from trident.agents.base import TradingAgent
from trident.data import calendar as cal
from trident.evaluation import backtest as bt
from trident.models import AgentName, Decision


def _make_bars(eval_map: dict, warmup_start: date, end: date) -> pd.DataFrame:
    """One synthetic ticker AAPL: flat 100 / 10M volume on warmup sessions (so ADV clears the
    liquidity floor and price-sanity passes), custom (open, close) on eval sessions."""
    rows = []
    for d in cal.sessions(warmup_start, end):
        o, c = eval_map.get(d, (100.0, 100.0))
        rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": "AAPL",
                "open": o,
                "high": max(o, c),
                "low": min(o, c),
                "close": c,
                "volume": 10_000_000,
            }
        )
    return pd.DataFrame(rows)


class _BuyOnceHold(TradingAgent):
    name = AgentName.QUANT

    async def generate_decisions(self, ctx):
        if ctx.book.positions:  # already invested -> hold (no rebalance -> hand-computable)
            return []
        return [
            Decision(
                agent=self.name,
                asof=ctx.asof,
                ticker="AAPL",
                target_weight=0.10,
                rationale={"r": "buy once"},
            )
        ]


async def test_backtest_golden_equity_curve(monkeypatch):
    evs = cal.sessions(date(2024, 3, 1), date(2024, 3, 10))[:3]
    bars = _make_bars(
        {evs[0]: (100.0, 102.0), evs[1]: (102.0, 104.0), evs[2]: (104.0, 101.0)},
        evs[0] - timedelta(days=90),
        evs[2],
    )

    def unexpected_history(*args, **kwargs):
        pytest.fail("in-memory bars must bypass market history")

    monkeypatch.setattr(bt.market, "get_history", unexpected_history)
    res = await bt.run_backtest(
        _BuyOnceHold(), ["AAPL"], evs[0], evs[2], start_cash=10_000, slippage_bps=5, bars=bars
    )
    # Session 1: flat -> buy int(0.10*10000/100)=10 sh @ 100*(1+5bps)=100.05 -> cash 8999.50;
    #   mark @102 -> 8999.50 + 10*102 = 10019.50
    # Session 2: hold; mark @104 -> 8999.50 + 1040 = 10039.50
    # Session 3: hold; mark @101 -> 8999.50 + 1010 = 10009.50
    assert [round(v, 2) for v in res.equity.tolist()] == [10019.50, 10039.50, 10009.50]
    assert res.trades == 1
    (fill,) = res.fills
    assert (fill.ticker, fill.qty, fill.price) == ("AAPL", 10, 100.05)


async def test_lagged_prices_never_leaks_session_t():
    evs = cal.sessions(date(2024, 3, 1), date(2024, 3, 28))
    df = _make_bars({}, evs[0] - timedelta(days=90), evs[-1])
    bars = bt._Bars(df)
    for asof in evs[5:]:  # every session with enough history behind it
        got = bt._LaggedPrices(bars, asof).get(["AAPL"], 30)
        assert not got.empty
        assert (got["date"].dt.date < asof).all(), f"look-ahead leak at {asof}"
