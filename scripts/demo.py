"""Run a three-session backtest using synthetic prices and a fixed buy-once policy."""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pandas as pd

from trident.agents.base import MarketContext, TradingAgent
from trident.data.calendar import sessions
from trident.evaluation.backtest import run_backtest
from trident.models import AgentName, Decision

TICKER = "AAPL"
START, END = date(2024, 3, 1), date(2024, 3, 5)


class BuyOnce(TradingAgent):
    name = AgentName.QUANT

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        if ctx.book.positions:
            return []
        return [
            Decision(
                agent=self.name,
                asof=ctx.asof,
                ticker=TICKER,
                target_weight=0.10,
                rationale={"policy": "Buy once at a 10% target weight"},
            )
        ]


def synthetic_bars() -> pd.DataFrame:
    prices = dict(zip(sessions(START, END), [(100, 102), (102, 104), (104, 101)], strict=True))
    rows = []
    for day in sessions(START - timedelta(days=90), END):
        open_price, close = prices.get(day, (100, 100))
        rows.append(
            {
                "date": pd.Timestamp(day),
                "ticker": TICKER,
                "open": open_price,
                "high": max(open_price, close),
                "low": min(open_price, close),
                "close": close,
                "volume": 10_000_000,
            }
        )
    return pd.DataFrame(rows)


async def main() -> None:
    result = await run_backtest(
        BuyOnce(), [TICKER], START, END, start_cash=10_000, bars=synthetic_bars()
    )
    print("Synthetic prices (AAPL) | Starting equity: $10,000.00 | Slippage: 5 bps")
    for fill in result.fills:
        print(
            f"{fill.ts.date()}  {fill.side.value.upper()} {fill.qty:g} {fill.ticker}"
            f" @ ${fill.price:,.2f}"
        )
    print("Session       Closing equity")
    for day, equity in result.equity.items():
        print(f"{day.date()}    ${equity:>12,.2f}")


if __name__ == "__main__":
    asyncio.run(main())
