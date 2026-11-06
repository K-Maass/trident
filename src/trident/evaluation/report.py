"""Backtest report writer (backtest-protocol step 8).

Renders a strategy suite (each run at 5 and 10 bps) into a Markdown report under
artifacts/backtests/: per-year table (return, Sharpe, maxDD, turnover), aggregate Sharpe ±
SE, and the mandatory limitations + reproduce line. Honest by construction — SR is always
shown with its standard error (doc 07).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trident.config import ARTIFACTS_DIR
from trident.evaluation import metrics as m
from trident.evaluation.backtest import BacktestResult

BACKTEST_DIR = ARTIFACTS_DIR / "backtests"

LIMITATIONS = (
    "- **Survivorship bias**: the universe is *today's* 40 names; delisted/removed names are "
    "absent. Inflates results; v2 fix is a point-in-time universe (doc 03).\n"
    "- **Corp-action adjustment** is to *today's* split/dividend history (`market.py` bars) — a "
    "mild look-ahead in absolute prices; returns are the honest quantity.\n"
    "- **Dividends unmodelled**: `get_activities → []` (backtest AND paper), so total returns "
    "are understated ~0.3%/quarter. Must be wired before the Phase 8 evaluation window opens.\n"
    "- **Paper/backtest fills are optimistic** (open ± slippage, no queue/impact) — treat "
    "Sharpe as an upper bound (doc 07)."
)


def per_year(equity: pd.Series, traded: pd.Series) -> pd.DataFrame:
    rows = []
    for year, eq_y in equity.groupby(equity.index.year):
        r = m.daily_returns(eq_y)
        traded_y = traded[traded.index.year == year]
        rows.append(
            {
                "year": int(year),
                "return": m.total_return(eq_y),
                "sharpe": m.sharpe(r),
                "maxDD": m.max_drawdown(eq_y),
                "turnover": m.turnover(traded_y, eq_y),
            }
        )
    return pd.DataFrame(rows)


def _fmt_year_table(df: pd.DataFrame) -> str:
    head = "| Year | Return | Sharpe | MaxDD | Turnover |\n|---|---|---|---|---|"
    lines = [
        f"| {r.year} | {r['return'] * 100:.1f}% | {r.sharpe:.2f} | "
        f"{r.maxDD * 100:.1f}% | {r.turnover:.1f}x |"
        for _, r in df.iterrows()
    ]
    return "\n".join([head, *lines])


def _aggregate_line(equity: pd.Series) -> str:
    r = m.daily_returns(equity)
    sr = m.sharpe(r)
    se = m.sharpe_standard_error(sr, len(r))
    sig = "significant" if se and abs(sr) / se > 2 else "NOT significant (|SR|/SE ≤ 2)"
    return (
        f"total return {m.total_return(equity) * 100:.1f}%, CAGR {m.cagr(equity) * 100:.1f}%, "
        f"Sharpe {sr:.2f} ± {se:.2f} ({sig}), MaxDD {m.max_drawdown(equity) * 100:.1f}%, "
        f"Calmar {m.calmar(equity):.2f}"
    )


def _red_flags(results: dict[str, dict[int, BacktestResult]]) -> list[str]:
    """Auto-detect the backtest-protocol red flags instead of leaving a boilerplate note."""
    flags: list[str] = []
    for strat, by_bps in results.items():
        eq5, eq10 = by_bps[5].equity, by_bps[10].equity
        sr5, sr10 = m.sharpe(m.daily_returns(eq5)), m.sharpe(m.daily_returns(eq10))
        if sr5 > 2.5:
            flags.append(f"⚠️ {strat}: aggregate Sharpe {sr5:.2f} > 2.5 — investigate for a bug.")
        if sr5 - sr10 > 0.3:
            flags.append(
                f"⚠️ {strat}: cost-fragile — Sharpe {sr5:.2f}→{sr10:.2f} from 5→10 bps "
                "(turnover-implied costs eat the edge)."
            )
        pj = per_year(eq5, by_bps[5].traded_value)
        last_year = pj["year"].max()
        for _, row in pj.iterrows():
            if row.sharpe > 2.5:
                note = (
                    " (partial year — high SE, not an aggregate breach)"
                    if row.year == last_year
                    else ""
                )
                flags.append(
                    f"ℹ️ {strat} {int(row.year)}: per-year Sharpe {row.sharpe:.2f} > 2.5{note}."
                )
    return flags or ["No red flags: no aggregate Sharpe > 2.5, no cost-fragility beyond 0.3."]


def _chart(results, benchmark_equity, benchmark_name, out_name: str) -> Path | None:
    """Equity + drawdown chart vs benchmark (protocol step 8). PNG next to the report."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 — chart is best-effort
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    series = {f"{benchmark_name} (BENCH)": benchmark_equity}
    series.update({s: by[5].equity for s, by in results.items()})
    for name, eq in series.items():
        norm = eq / eq.iloc[0]
        ax1.plot(norm.index, norm.values, label=name, lw=1.3)
        dd = eq / eq.cummax() - 1.0
        ax2.plot(dd.index, dd.values * 100, lw=1.0)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.set_title(out_name)
    ax2.set_ylabel("drawdown %")
    ax2.axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    path = BACKTEST_DIR / f"{out_name}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def render_backtest_report(
    question: str,
    window: tuple[str, str],
    results: dict[str, dict[int, BacktestResult]],
    benchmark_equity: pd.Series,
    benchmark_name: str,
    reproduce: str,
    out_name: str,
    configs_tried: int = 1,
    provenance: str = "",
) -> Path:
    """`results[strategy][bps] = BacktestResult`. Writes the report (+ chart), returns its path."""
    md: list[str] = [f"# Backtest — {out_name}", ""]
    md += [f"**Question (frozen):** {question}", ""]
    md += [f"**Window:** {window[0]} → {window[1]}", f"**Reproduce:** `{reproduce}`", ""]
    md += [
        f"**Costs:** run at 5 bps and 10 bps slippage/side. **Configs tried:** {configs_tried}.",
        "",
    ]
    if provenance:
        md += ["## Model selection / provenance", "", provenance, ""]

    chart = _chart(results, benchmark_equity, benchmark_name, out_name)
    if chart is not None:
        md += ["## Equity & drawdown vs benchmark", "", f"![equity & drawdown]({chart.name})", ""]

    # Summary table across strategies (5 bps)
    md += ["## Summary (5 bps, vs benchmark)", ""]
    md += [
        "| Strategy | Return | Sharpe ± SE | MaxDD | Calmar | Trades |",
        "|---|---|---|---|---|---|",
    ]
    bench_r = m.daily_returns(benchmark_equity)
    bench_sr = m.sharpe(bench_r)
    md.append(
        f"| {benchmark_name} (BENCH) | {m.total_return(benchmark_equity) * 100:.1f}% | "
        f"{bench_sr:.2f} ± {m.sharpe_standard_error(bench_sr, len(bench_r)):.2f} | "
        f"{m.max_drawdown(benchmark_equity) * 100:.1f}% | {m.calmar(benchmark_equity):.2f} | — |"
    )
    for strat, by_bps in results.items():
        res = by_bps[5]
        r = m.daily_returns(res.equity)
        sr = m.sharpe(r)
        md.append(
            f"| {strat} | {m.total_return(res.equity) * 100:.1f}% | "
            f"{sr:.2f} ± {m.sharpe_standard_error(sr, len(r)):.2f} | "
            f"{m.max_drawdown(res.equity) * 100:.1f}% | {m.calmar(res.equity):.2f} | {res.trades} |"
        )
    md.append("")

    # Per-strategy detail: 5 vs 10 bps aggregate + per-year table
    for strat, by_bps in results.items():
        md += [f"## {strat}", ""]
        md += [f"- **5 bps:** {_aggregate_line(by_bps[5].equity)}"]
        md += [f"- **10 bps:** {_aggregate_line(by_bps[10].equity)}", ""]
        md += [
            "Per-year (5 bps):",
            "",
            _fmt_year_table(per_year(by_bps[5].equity, by_bps[5].traded_value)),
            "",
        ]

    md += ["## Limitations", "", LIMITATIONS, ""]
    md += ["## Red-flag scan (auto)", ""]
    md += [f"- {f}" for f in _red_flags(results)]
    md += [""]

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    path = BACKTEST_DIR / f"{out_name}.md"
    path.write_text("\n".join(md))
    return path
