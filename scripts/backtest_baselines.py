"""Phase 3 M3: run the 4 rule baselines + SPY buy-and-hold over the walk-forward window at
5 and 10 bps, and write the report. This is the bar the ML must beat after costs.

Reproduce: uv run python scripts/backtest_baselines.py
"""

import asyncio

from trident.agents.quant.baselines import BASELINES
from trident.config import load_universe
from trident.evaluation import backtest as bt
from trident.evaluation.report import render_backtest_report

START, END = "2016-06-01", "2026-06-30"  # ~10y, spans the 2020 and 2022 drawdowns
START_CASH = 30_000.0


async def main() -> None:
    universe = load_universe()
    results: dict[str, dict[int, bt.BacktestResult]] = {}
    for name, cls in BASELINES.items():
        results[name] = {}
        for bps in (5, 10):
            print(f"running {name} @ {bps}bps ...", flush=True)
            results[name][bps] = await bt.run_backtest(
                cls(), universe, START, END, start_cash=START_CASH, slippage_bps=float(bps)
            )
    bench = bt.buy_and_hold_equity("SPY", START, END, START_CASH)

    path = render_backtest_report(
        question=(
            "Do the 4 doc-03 rule baselines (MA-cross, 12-1 momentum, RSI-2, Markov regime) "
            "beat SPY buy-and-hold on Sharpe, after 5-10 bps costs, 2016-2026?"
        ),
        window=(START, END),
        results=results,
        benchmark_equity=bench,
        benchmark_name="SPY",
        reproduce="uv run python scripts/backtest_baselines.py",
        out_name=f"baselines-{END}",
    )
    print(f"\nreport: {path}")


if __name__ == "__main__":
    asyncio.run(main())
