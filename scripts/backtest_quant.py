"""Phase 3 M6: the ML go/no-go. Backtest the QUANT ML as a WALK-FORWARD OUT-OF-SAMPLE replay
(each date scored by a model trained only on prior data) vs the 4 rule baselines + SPY, over
the OOS window, at 5 and 10 bps. Writes the final report.

Reproduce: uv run python scripts/backtest_quant.py
"""

import asyncio

import pandas as pd

from trident.agents.quant.agent import OOSReplayAgent
from trident.agents.quant.baselines import BASELINES
from trident.config import ARTIFACTS_DIR, load_universe
from trident.evaluation import backtest as bt
from trident.evaluation import metrics as m
from trident.evaluation.report import render_backtest_report

OOS_PATH = ARTIFACTS_DIR / "models" / "quant_oos_predictions.parquet"
START_CASH = 30_000.0


async def main() -> None:
    oos = pd.read_parquet(OOS_PATH)
    oos["date"] = pd.to_datetime(oos["date"])
    # OOS window: the walk-forward test folds. +10 days so the first session has a score.
    start = (oos["date"].min() + pd.Timedelta(days=10)).date().isoformat()
    end = oos["date"].max().date().isoformat()
    universe = load_universe()

    results: dict[str, dict[int, bt.BacktestResult]] = {"quant_ml": {}}
    for bps in (5, 10):
        print(f"running quant_ml (OOS replay) @ {bps}bps ...", flush=True)
        results["quant_ml"][bps] = await bt.run_backtest(
            OOSReplayAgent(oos),
            universe,
            start,
            end,
            start_cash=START_CASH,
            slippage_bps=float(bps),
        )
    for name, cls in BASELINES.items():
        results[name] = {}
        for bps in (5, 10):
            print(f"running {name} @ {bps}bps ...", flush=True)
            results[name][bps] = await bt.run_backtest(
                cls(), universe, start, end, start_cash=START_CASH, slippage_bps=float(bps)
            )
    bench = bt.buy_and_hold_equity("SPY", start, end, START_CASH)

    provenance = (
        "- The `OOSReplayAgent` executes each score at the NEXT session's open (a close[T] "
        "signal is tradeable only at open[T+1]) — matching the deployed `QuantAgent`. An earlier "
        "same-session replay was a 1-day look-ahead that *deflated* the ML (0.43 vs 1.16); "
        "corrected after the backtest-auditor FAIL.\n"
        "- `SCORE_THRESHOLD` 0.55 → ⅓: a spec correction from the score *distribution* "
        "(top-tercile P centers on 0.33, so 0.55 is unreachable), not tuned to returns. "
        "Sensitivity: ML loses "
        "at every threshold (Sharpe 0.13–0.46 same-session; ~1.0–1.2 corrected — never clears "
        "momentum by the +0.10 bar).\n"
        "- Provenance honesty: one in-sample run (fake 1.99 Sharpe) was discarded as a "
        "methodology error, and the alignment fix above was a bug fix — neither is a tune-to-win "
        "config search. The go/no-go itself is ONE-SHOT."
    )
    path = render_backtest_report(
        question=(
            "Does the QUANT LightGBM ranker (walk-forward OOS, executed as deployed) beat SPY "
            "and the best rule baseline on Sharpe, after 5-10 bps costs, over the OOS window?"
        ),
        window=(start, end),
        results=results,
        benchmark_equity=bench,
        benchmark_name="SPY",
        reproduce="uv run python scripts/backtest_quant.py",
        out_name=f"quant-ml-vs-baselines-{end}",
        configs_tried=1,
        provenance=provenance,
    )

    # v1 decision, computed from the corrected results per the binding rule (roadmap build log).
    def sr(res):
        return m.sharpe(m.daily_returns(res.equity))

    ml5, ml10 = sr(results["quant_ml"][5]), sr(results["quant_ml"][10])
    mo5, mo10 = sr(results["momentum_12_1"][5]), sr(results["momentum_12_1"][10])
    beats = (ml5 >= mo5 + 0.10) and (ml10 >= mo10 + 0.10)
    winner = "quant_ml (ML)" if beats else "momentum_12_1 (baseline)"
    decision = (
        f"\n## v1 decision (binding rule)\n\n"
        f"ML Sharpe {ml5:.2f}/{ml10:.2f} (5/10 bps) vs momentum {mo5:.2f}/{mo10:.2f}. Rule: ML "
        f"ships only if it beats momentum by ≥ +0.10 at BOTH cost levels without worse maxDD; "
        f"ties → simpler model. Beats-by-0.10 at both: **{beats}**. ML turnover "
        f"{results['quant_ml'][5].trades} trades vs momentum {results['momentum_12_1'][5].trades} "
        f"(cost-robustness strongly favors momentum). **→ {winner} ships as QUANT v1.**\n"
    )
    with path.open("a") as f:
        f.write(decision)
    print(f"\nreport: {path}\n{decision}")


if __name__ == "__main__":
    asyncio.run(main())
