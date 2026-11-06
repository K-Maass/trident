"""QUANT feature/label tests — the mandatory leakage self-test (backtest-protocol step 4,
the roadmap's "shifted-copy trick") plus feature and label sanity.

All offline: a deterministic synthetic OHLCV panel (seeded GBM) stands in for the market
cache, so nothing here touches the network or DB. The leakage test is the load-bearing one:
deleting every row AFTER date t must leave every feature value at t byte-for-byte identical —
if it doesn't, a feature is peeking at the future and the whole edge is a mirage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trident.agents.quant.features import (
    FEATURE_NAMES,
    compute_feature_panel,
    compute_features,
)
from trident.agents.quant.labels import make_labels

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
N_DAYS = 500


def _synth_prices(tickers=TICKERS, n=N_DAYS, seed=7) -> pd.DataFrame:
    """Deterministic positive OHLCV via independent geometric random walks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n, name="date")
    frames = []
    for i, tkr in enumerate(tickers):
        rets = rng.normal(0.0004, 0.017, size=n)
        close = 100.0 * (1.0 + i * 0.1) * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0.0, 0.006, size=n)) * close
        high = close + intraday
        low = close - intraday
        open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n))
        volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": tkr,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _spy_prices(n=N_DAYS, seed=99) -> pd.DataFrame:
    return _synth_prices(["SPY"], n=n, seed=seed)


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _synth_prices()


@pytest.fixture(scope="module")
def spy() -> pd.DataFrame:
    return _spy_prices()


@pytest.fixture(scope="module")
def full_panel(prices, spy) -> pd.DataFrame:
    return compute_feature_panel(prices, spy).set_index(["date", "ticker"])


def _sample_dates(prices) -> list[pd.Timestamp]:
    dates = np.array(sorted(prices["date"].unique()))
    # spread across history, all well past the 126d warmup
    return [pd.Timestamp(dates[i]) for i in (200, 280, 360, 430, 480)]


# ============================================================ LEAKAGE (mandatory)
def test_deleting_future_rows_does_not_change_past_features(prices, spy, full_panel):
    """Shifted-copy trick: for 5 sample dates, features at t computed on the FULL history must
    be byte-for-byte identical to features at t computed on history truncated at t. Any
    difference == look-ahead leakage."""
    for t in _sample_dates(prices):
        p_t = prices[prices["date"] <= t]
        s_t = spy[spy["date"] <= t]
        trunc = compute_feature_panel(p_t, s_t).set_index(["date", "ticker"])

        a = full_panel.xs(t, level="date")[FEATURE_NAMES].sort_index()
        b = trunc.xs(t, level="date")[FEATURE_NAMES].sort_index()
        # exact equality — every feature is a backward-only function of past bars
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_bounded_lookback_late_start_reproduces_row(prices, spy, full_panel):
    """The M6 agent computes features from a SHORT price history. A window that starts only
    ~200 sessions before t must reproduce the same feature row (all features have bounded
    lookback). `spy_days_in_regime` can legitimately differ if a regime run predates the
    window, so it is excluded from this equality (it IS covered exactly by the future-
    truncation test above)."""
    dates = np.array(sorted(prices["date"].unique()))
    cols = [c for c in FEATURE_NAMES if c != "spy_days_in_regime"]
    for t in _sample_dates(prices):
        ti = int(np.searchsorted(dates, np.datetime64(t)))
        start = pd.Timestamp(dates[ti - 200])
        p_w = prices[(prices["date"] <= t) & (prices["date"] >= start)]
        s_w = spy[(spy["date"] <= t) & (spy["date"] >= start)]
        win = compute_feature_panel(p_w, s_w).set_index(["date", "ticker"])

        a = full_panel.xs(t, level="date")[cols].sort_index()
        b = win.xs(t, level="date")[cols].sort_index()
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_compute_features_matches_panel_last_row(prices, spy, full_panel):
    """The single-date inference API returns exactly the panel's final-date row per ticker."""
    last = pd.Timestamp(sorted(prices["date"].unique())[-1])
    row = compute_features(prices, last, spy_prices=spy)
    assert row.index.name == "ticker"
    assert list(row.columns) == FEATURE_NAMES
    expected = full_panel.xs(last, level="date")[FEATURE_NAMES].sort_index()
    pd.testing.assert_frame_equal(row.sort_index(), expected, check_exact=True)


# ============================================================ FEATURE SANITY
def test_final_feature_row_has_no_nan(prices, spy):
    """A well-covered universe on the final date must yield a complete feature vector."""
    last = pd.Timestamp(sorted(prices["date"].unique())[-1])
    row = compute_features(prices, last, spy_prices=spy)
    assert len(row) == len(TICKERS)
    assert not row.isna().any().any()


def test_rsi_bounded_0_100(full_panel):
    for col in ("rsi_2", "rsi_14"):
        vals = full_panel[col].dropna()
        assert vals.min() >= 0.0
        assert vals.max() <= 100.0


def test_regime_onehot_is_mutually_exclusive(full_panel):
    onehot = full_panel[["spy_regime_bull", "spy_regime_bear", "spy_regime_sideways"]].dropna()
    # exactly one regime active per row
    assert np.allclose(onehot.sum(axis=1).to_numpy(), 1.0)
    assert set(np.unique(onehot.to_numpy())) <= {0.0, 1.0}


def test_cross_sectional_z_is_standardized(full_panel):
    """ret_21d_z is a per-date z-score across tickers -> each date has mean ~0, std ~1."""
    per_date = full_panel["ret_21d_z"].dropna().groupby(level="date")
    means = per_date.mean()
    stds = per_date.std(ddof=0)
    assert means.abs().max() < 1e-8
    # a handful of dates can have a near-degenerate spread; the vast majority are ~1
    assert stds.median() == pytest.approx(1.0, abs=1e-6)


def test_days_in_regime_is_monotone_within_a_run(full_panel):
    """days_in_regime counts consecutive same-regime sessions -> increments by 1 or resets."""
    s = full_panel["spy_days_in_regime"].dropna().xs(TICKERS[0], level="ticker").sort_index()
    diffs = s.diff().dropna()
    # each step is either +1 (run continues) or a reset to 1 (=> diff = 1 - prev <= 0)
    assert ((diffs == 1.0) | (diffs <= 0.0)).all()
    assert (s >= 1.0).all()


# ============================================================ LABEL SANITY
def test_labels_are_top_tercile_cross_sectional():
    """Hand-built frame: 6 tickers, a single scorable date. The top third by 5d forward return
    (2 of 6) get label 1, the rest 0."""
    dates = pd.bdate_range("2020-01-02", periods=7, name="date")
    # forward 5d return is decided by close[t] vs close[t+5]; order winners deterministically
    fwd_rank = {"T1": 6, "T2": 5, "T3": 4, "T4": 3, "T5": 2, "T6": 1}  # T1 best, T6 worst
    rows = []
    for tkr, mult in fwd_rank.items():
        closes = [100.0] * 7
        closes[5] = 100.0 * (1.0 + 0.01 * mult)  # day index 5 = t0 + 5 sessions
        for d, c in zip(dates, closes, strict=True):
            rows.append({"date": d, "ticker": tkr, "close": c})
    bars = pd.DataFrame(rows)

    labels = make_labels(bars, horizon=5)
    day0 = labels[labels["date"] == dates[0]].set_index("ticker")["label"]
    # top tercile of 6 = 2 names: the two highest forward returns
    assert set(day0[day0 == 1].index) == {"T1", "T2"}
    assert day0.sum() == 2


def test_labels_drop_tail_without_forward_return():
    bars = _synth_prices(["ZZZ"], n=30, seed=1)
    labels = make_labels(bars, horizon=5)
    last_close_date = bars["date"].max()
    # the final `horizon` dates have no forward return -> excluded
    assert labels["date"].max() < last_close_date
    assert labels["date"].max() == sorted(bars["date"].unique())[-6]


def test_label_base_rate_near_one_third(prices):
    labels = make_labels(prices, horizon=5)
    assert 0.25 < labels["label"].mean() < 0.40
