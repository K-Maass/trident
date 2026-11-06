"""QUANT feature engineering (doc 03) — cross-sectional, strictly lag-safe.

Every feature is computable from data available at decision time: it uses ONLY bars with
date <= the row's own date (backward-looking rolling windows, backward log returns, and
same-date cross-sectional statistics). This is the anti-look-ahead seam — proven by the
shifted-copy test in tests/test_quant_features.py (deleting future rows must not change any
past feature value) and by a bounded-lookback test (a window that starts late reproduces the
same row, so the M6 agent can compute identical features from a short price history).

`compute_features(prices, asof)` returns ONE row per ticker (index=ticker) of the feature
vector AS OF the latest date in `prices` — the shape the M6 QuantAgent scores each day.
`compute_feature_panel(prices, spy_prices)` returns the whole tidy history — the shape
train.py assembles a matrix from. Both share the same math, so training and inference agree,
and `FEATURE_NAMES` fixes the column order they must both use.

SPY is NOT in the 40-name universe (it is BENCH-only, universe.yaml). Cross-asset / regime
features need it, so it is fetched separately (data.market.get_history(["SPY"], ...)) and
passed in — or fetched here on demand when a caller omits it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered feature list — training and inference MUST use this exact order.
FEATURE_NAMES: list[str] = [
    # returns (log)
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "ret_21d_z",  # 21d return z-scored cross-sectionally vs the universe
    # mean-reversion
    "rsi_2",
    "rsi_14",
    "dist_sma20_atr",  # (close - SMA20) in ATR(14) units
    # trend
    "sma20_sma100_ratio",
    "linreg_slope_63",  # OLS slope of log-price over 63d (per-day, already normalized)
    # volatility
    "realized_vol_21",
    "vol_of_vol",
    "atr_pct",  # ATR(14) / close
    # volume
    "dollar_vol_ratio_z",  # z of (5d avg $vol / 63d avg $vol)
    "obv_slope",  # 21d OLS slope of OBV, scaled by 21d avg volume (level-invariant)
    # calendar
    "dow",  # day-of-week 0=Mon..4=Fri
    "days_to_month_end",  # calendar days to month end
    # cross-asset regime context (same for every ticker on a given date)
    "spy_ret_5d",
    "spy_vol_21",  # SPY 21d realized vol — VIX proxy
    "spy_regime_bull",
    "spy_regime_bear",
    "spy_regime_sideways",
    "spy_days_in_regime",
]

_REGIME_THRESH = 0.02  # ±2% on the 20d return — fixed a priori (same labeler as baseline 4)
_REGIME_WINDOW = 20


# ------------------------------------------------------------------ helpers
def _as_ts(d) -> pd.Timestamp:
    """`asof` (datetime/date/str, tz-aware or not) -> a tz-naive midnight Timestamp, so it
    compares against the tz-naive `date` column that market.get_history produces."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Tidy long bars -> a (date x ticker) matrix, sorted by date. Duplicate (date,ticker)
    pairs are not expected (get_history guarantees one row each) but aggfunc='last' is a
    defensive no-op if any slip through."""
    return df.pivot_table(index="date", columns="ticker", values=value, aggfunc="last").sort_index()


def _rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Connors-style RSI on simple rolling averages (matches baselines._rsi, vectorized over
    tickers). An all-up window (avg_loss==0) is 100 (overbought); warmup rows stay NaN."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # loss==0 with some gain -> 100; leave warmup NaN (avg_gain NaN) untouched so it drops out
    return rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)


def _atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Average True Range on a simple rolling mean of true range (backward-only)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    return tr.rolling(period).mean()


def _rolling_ols_slope(y: pd.DataFrame, window: int) -> pd.DataFrame:
    """OLS slope of each column of `y` on a unit-spaced time index, over a trailing `window`.

    Closed form via rolling sums (no per-window apply): with x = 0,1,2,... (global session
    index, which only shifts by a constant inside any window and so does not change the
    slope), slope = [Sxy - Sx*Sy/n] / [Sxx - Sx^2/n]. Fully vectorized, backward-only."""
    n = window
    x = pd.Series(np.arange(len(y), dtype="float64"), index=y.index)
    sx = x.rolling(n).sum()
    sxx = (x * x).rolling(n).sum()
    sy = y.rolling(n).sum()
    sxy = y.mul(x, axis=0).rolling(n).sum()
    denom = sxx - sx * sx / n  # a per-date Series (variance of n consecutive ints * n)
    num = sxy.sub(sy.mul(sx, axis=0) / n)
    return num.div(denom, axis=0)


def _xsection_z(mat: pd.DataFrame) -> pd.DataFrame:
    """Row-wise (per-date, across tickers) z-score. Uses same-date values only."""
    mu = mat.mean(axis=1)
    sd = mat.std(axis=1, ddof=0).replace(0, np.nan)
    return mat.sub(mu, axis=0).div(sd, axis=0)


def _broadcast(series: pd.Series, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """A per-date Series -> a (date x ticker) matrix, same value across all tickers."""
    aligned = series.reindex(index)
    return pd.DataFrame(
        np.repeat(aligned.to_numpy()[:, None], len(columns), axis=1),
        index=index,
        columns=columns,
    )


def _spy_context(spy_prices: pd.DataFrame) -> pd.DataFrame:
    """Per-date SPY regime context (indexed by date): 5d return, 21d realized vol (VIX proxy),
    Bull/Bear/Sideways one-hot from the ±2% 20d-return labeler (baseline 4), and days in the
    current regime. All trailing-window only -> lag-safe by construction (doc 03)."""
    spy = spy_prices.sort_values("date")
    close = spy.set_index("date")["close"].astype(float)
    logc = np.log(close)
    ret_1d = logc.diff(1)

    ret_20 = close / close.shift(_REGIME_WINDOW) - 1.0
    label = pd.Series(
        np.where(
            ret_20 > _REGIME_THRESH, "bull", np.where(ret_20 < -_REGIME_THRESH, "bear", "side")
        ),
        index=close.index,
    ).where(ret_20.notna())  # keep warmup as NaN
    run_id = (label != label.shift(1)).cumsum()
    days_in_regime = label.groupby(run_id).cumcount() + 1.0
    days_in_regime = days_in_regime.where(label.notna())

    return pd.DataFrame(
        {
            "spy_ret_5d": logc.diff(5),
            "spy_vol_21": ret_1d.rolling(21).std(ddof=0),
            "spy_regime_bull": (label == "bull").astype(float).where(label.notna()),
            "spy_regime_bear": (label == "bear").astype(float).where(label.notna()),
            "spy_regime_sideways": (label == "side").astype(float).where(label.notna()),
            "spy_days_in_regime": days_in_regime,
        }
    )


# --------------------------------------------------------------- public API
def compute_feature_panel(prices: pd.DataFrame, spy_prices: pd.DataFrame) -> pd.DataFrame:
    """Tidy feature panel [date, ticker, *FEATURE_NAMES] over the whole history in `prices`.

    `prices`: tidy long [date, ticker, open, high, low, close, volume] for the universe.
    `spy_prices`: same schema for SPY only. Rows in the warmup window (any feature still NaN)
    are kept here — callers (train.py) drop NaN after joining labels; single-date inference
    slices the final complete row.
    """
    if prices.empty:
        return pd.DataFrame(columns=["date", "ticker", *FEATURE_NAMES])

    prices = prices.sort_values(["ticker", "date"])
    close = _pivot(prices, "close")
    high = _pivot(prices, "high")
    low = _pivot(prices, "low")
    volume = _pivot(prices, "volume")
    idx, cols = close.index, close.columns

    logc = np.log(close)
    ret_21d = logc.diff(21)

    dollar_vol = close * volume
    dv_ratio = dollar_vol.rolling(5).mean() / dollar_vol.rolling(63).mean()

    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    obv_slope = _rolling_ols_slope(obv, 21) / volume.rolling(21).mean()

    sma20 = close.rolling(20).mean()
    atr14 = _atr(high, low, close, 14)
    realized_vol_21 = logc.diff(1).rolling(21).std(ddof=0)

    # calendar (per-date scalars broadcast to every ticker)
    dow = pd.Series(idx.dayofweek.astype("float64"), index=idx)
    days_to_month_end = pd.Series((idx.days_in_month - idx.day).astype("float64"), index=idx)

    spy_ctx = _spy_context(spy_prices)

    # ordered dict -> guarantees the panel's column order matches FEATURE_NAMES
    mats: dict[str, pd.DataFrame] = {
        "ret_1d": logc.diff(1),
        "ret_5d": logc.diff(5),
        "ret_21d": ret_21d,
        "ret_63d": logc.diff(63),
        "ret_126d": logc.diff(126),
        "ret_21d_z": _xsection_z(ret_21d),
        "rsi_2": _rsi(close, 2),
        "rsi_14": _rsi(close, 14),
        "dist_sma20_atr": (close - sma20) / atr14,
        "sma20_sma100_ratio": sma20 / close.rolling(100).mean(),
        "linreg_slope_63": _rolling_ols_slope(logc, 63),
        "realized_vol_21": realized_vol_21,
        "vol_of_vol": realized_vol_21.rolling(21).std(ddof=0),
        "atr_pct": atr14 / close,
        "dollar_vol_ratio_z": _xsection_z(dv_ratio),
        "obv_slope": obv_slope,
        "dow": _broadcast(dow, idx, cols),
        "days_to_month_end": _broadcast(days_to_month_end, idx, cols),
        "spy_ret_5d": _broadcast(spy_ctx["spy_ret_5d"], idx, cols),
        "spy_vol_21": _broadcast(spy_ctx["spy_vol_21"], idx, cols),
        "spy_regime_bull": _broadcast(spy_ctx["spy_regime_bull"], idx, cols),
        "spy_regime_bear": _broadcast(spy_ctx["spy_regime_bear"], idx, cols),
        "spy_regime_sideways": _broadcast(spy_ctx["spy_regime_sideways"], idx, cols),
        "spy_days_in_regime": _broadcast(spy_ctx["spy_days_in_regime"], idx, cols),
    }

    long = pd.concat({name: m.stack(future_stack=True) for name, m in mats.items()}, axis=1)
    long.index.set_names(["date", "ticker"], inplace=True)
    panel = long.reset_index()
    return (
        panel[["date", "ticker", *FEATURE_NAMES]]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def compute_features(
    prices: pd.DataFrame, asof, spy_prices: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One feature row per ticker (index=ticker, columns=FEATURE_NAMES) AS OF the latest date
    in `prices` (guaranteed by the caller to be <= asof — no look-ahead).

    `spy_prices` is fetched from the market cache over the same date range when omitted, so
    the M6 agent can just pass the universe bars. Passing it explicitly (tests, or when the
    caller already holds it) avoids the data-layer round trip.
    """
    if prices.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)

    asof_ts = _as_ts(asof)
    prices = prices[prices["date"] <= asof_ts]
    if prices.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)

    if spy_prices is None:
        from trident.data import market  # local import: keeps the feature math I/O-free

        lo, hi = prices["date"].min().date(), prices["date"].max().date()
        spy_prices = market.get_history(["SPY"], lo, hi)
    spy_prices = spy_prices[spy_prices["date"] <= asof_ts]

    panel = compute_feature_panel(prices, spy_prices)
    if panel.empty:
        return pd.DataFrame(columns=FEATURE_NAMES)

    last = panel["date"].max()
    row = panel[panel["date"] == last].set_index("ticker")
    return row[FEATURE_NAMES]


__all__ = ["FEATURE_NAMES", "compute_feature_panel", "compute_features"]
