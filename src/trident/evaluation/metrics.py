"""Performance metrics (doc 07) — pure functions on a daily equity/returns series.

Golden rule 5: money math gets tests. Every function here is deterministic and
hand-checkable; `tests/test_metrics.py` pins them against hand-computed fixtures.
Undefined results (zero variance, empty series) return NaN rather than a fake 0 — the
weekly report is required to print SR ± SE and refuse the word "significant" unless
|SR|/SE > 2, so honesty about undefined-ness matters.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    """Simple daily returns r_t = E_t/E_{t-1} - 1, first row dropped."""
    return equity.astype(float).pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    e = equity.astype(float)
    if len(e) < 2 or e.iloc[0] == 0:
        return float("nan")
    return float(e.iloc[-1] / e.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    e = equity.astype(float)
    n = len(e) - 1
    if n < 1 or e.iloc[0] <= 0 or e.iloc[-1] <= 0:
        return float("nan")
    years = n / periods_per_year
    return float((e.iloc[-1] / e.iloc[0]) ** (1.0 / years) - 1.0)


def _excess(returns: pd.Series, rf_annual: float, periods_per_year: int) -> pd.Series:
    return returns.astype(float) - rf_annual / periods_per_year


def sharpe(
    returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    ex = _excess(returns, rf_annual, periods_per_year)
    sd = ex.std(ddof=1)
    if len(ex) < 2 or sd == 0 or math.isnan(sd):
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(periods_per_year))


def sharpe_standard_error(
    sharpe_ratio: float, n_periods: int, periods_per_year: int = TRADING_DAYS
) -> float:
    """SE(Sharpe) ≈ sqrt((1 + SR²/2) / T_years) (doc 07). With ~60 days SE ≈ 2.1 —
    so a measured SR of 1.0 is indistinguishable from 0."""
    years = n_periods / periods_per_year
    if years <= 0 or math.isnan(sharpe_ratio):
        return float("nan")
    return float(math.sqrt((1.0 + sharpe_ratio**2 / 2.0) / years))


def sortino(
    returns: pd.Series, rf_annual: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> float:
    ex = _excess(returns, rf_annual, periods_per_year)
    downside = ex[ex < 0]
    dd = math.sqrt((downside**2).mean()) if len(downside) else 0.0
    if len(ex) < 2 or dd == 0:
        return float("nan")
    return float(ex.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Most negative peak-to-trough: min(E_t / running_max - 1). Returns ≤ 0."""
    e = equity.astype(float)
    if len(e) < 2:
        return float("nan")
    return float((e / e.cummax() - 1.0).min())


def calmar(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(equity)
    g = cagr(equity, periods_per_year)
    if mdd == 0 or math.isnan(mdd) or math.isnan(g):
        return float("nan")
    return float(g / abs(mdd))


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """OLS r_agent - rf = α + β(r_bench - rf). Returns annualized α, its t-stat, and β."""
    a = _excess(returns, rf_annual, periods_per_year)
    b = _excess(benchmark_returns, rf_annual, periods_per_year)
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return {"alpha_annual": float("nan"), "alpha_tstat": float("nan"), "beta": float("nan")}
    y, x = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
    reg = stats.linregress(x, y)
    tstat = reg.intercept / reg.intercept_stderr if reg.intercept_stderr else float("nan")
    return {
        "alpha_annual": float(reg.intercept * periods_per_year),
        "alpha_tstat": float(tstat),
        "beta": float(reg.slope),
    }


def hit_rate(returns: pd.Series) -> float:
    r = returns.astype(float).dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def turnover(
    traded_value: pd.Series, equity: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualized Σ|traded value| / average equity."""
    avg_eq = equity.astype(float).mean()
    n = len(traded_value)
    if avg_eq == 0 or n == 0:
        return float("nan")
    return float(traded_value.abs().sum() / avg_eq / (n / periods_per_year))


def exposure(invested_value: pd.Series, equity: pd.Series) -> float:
    """Average invested fraction (1 - cash weight)."""
    joined = pd.concat([invested_value.astype(float), equity.astype(float)], axis=1).dropna()
    joined = joined[joined.iloc[:, 1] != 0]
    if joined.empty:
        return float("nan")
    return float((joined.iloc[:, 0] / joined.iloc[:, 1]).mean())


def summary(
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    """Convenience roll-up for reports. SR is reported with its standard error."""
    r = daily_returns(equity)
    sr = sharpe(r, rf_annual, periods_per_year)
    out = {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "sharpe": sr,
        "sharpe_se": sharpe_standard_error(sr, len(r), periods_per_year),
        "sortino": sortino(r, rf_annual, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity, periods_per_year),
        "hit_rate": hit_rate(r),
    }
    if benchmark_equity is not None:
        out.update(alpha_beta(r, daily_returns(benchmark_equity), rf_annual, periods_per_year))
    return out


__all__ = [
    "daily_returns",
    "total_return",
    "cagr",
    "sharpe",
    "sharpe_standard_error",
    "sortino",
    "max_drawdown",
    "calmar",
    "alpha_beta",
    "hit_rate",
    "turnover",
    "exposure",
    "summary",
    "TRADING_DAYS",
]
