"""Metrics tests — hand-computed fixtures (golden rule 5). Every expected value is
written literally, not recomputed by the code under test."""

from __future__ import annotations

import math

import pandas as pd

from trident.evaluation import metrics as m

SQRT252 = math.sqrt(252)


def test_total_return():
    assert abs(m.total_return(pd.Series([100, 110, 121])) - 0.21) < 1e-9  # 121/100 - 1


def test_cagr_doubles_in_one_year():
    eq = pd.Series([100.0] * 252 + [200.0])  # 253 points -> n=252 periods, x2
    assert abs(m.cagr(eq) - 1.0) < 1e-9  # (200/100)^(252/252) - 1 = 1.0


def test_max_drawdown():
    # running max 100,120,120,120 -> dd at 90 = 90/120-1 = -0.25 (the worst)
    assert m.max_drawdown(pd.Series([100, 120, 90, 110])) == -0.25


def test_sharpe_hand_computed():
    # returns mean 0.02, std(ddof=1) 0.01 -> 0.02/0.01 * sqrt(252) = 2*sqrt(252)
    sr = m.sharpe(pd.Series([0.01, 0.02, 0.03]))
    assert abs(sr - 2 * SQRT252) < 1e-9


def test_sharpe_standard_error():
    # SE(SR=1, T=1yr) = sqrt((1 + 1/2)/1) = sqrt(1.5)
    assert abs(m.sharpe_standard_error(1.0, 252) - math.sqrt(1.5)) < 1e-12
    # ~60 days under the null (SR=0): SE = sqrt(252/60) ~ 2.05 (the doc-07 honesty point)
    assert abs(m.sharpe_standard_error(0.0, 60) - math.sqrt(252 / 60)) < 1e-12


def test_sortino_hand_computed():
    # mean 0.005; downside dev = sqrt((0.01^2+0.02^2)/2)=0.01581139 -> 0.31622777*sqrt(252)
    s = m.sortino(pd.Series([0.02, -0.01, 0.03, -0.02]))
    assert abs(s - 5.0199602) < 1e-6


def test_calmar():
    eq = pd.Series([100, 120, 90, 110])  # maxdd -0.25
    expected = m.cagr(eq) / 0.25
    assert abs(m.calmar(eq) - expected) < 1e-12


def test_alpha_beta_recovers_known_line():
    bench = pd.Series([0.01, -0.02, 0.03, 0.00, -0.01, 0.02])
    agent = 2 * bench + 0.001  # beta 2, daily alpha 0.001
    ab = m.alpha_beta(agent, bench)
    assert abs(ab["beta"] - 2.0) < 1e-9
    assert abs(ab["alpha_annual"] - 0.001 * 252) < 1e-9


def test_hit_rate():
    assert m.hit_rate(pd.Series([0.01, -0.01, 0.02, 0.0, -0.03])) == 0.4  # 2 of 5 > 0


def test_turnover_annualized():
    traded = pd.Series([1000, 500, 1500])  # sum 3000 over 3 days
    eq = pd.Series([10000, 10000, 10000])  # avg 10000
    # 3000/10000 / (3/252) = 0.3 * 84 = 25.2
    assert abs(m.turnover(traded, eq) - 25.2) < 1e-9


def test_exposure():
    assert m.exposure(pd.Series([5000, 6000]), pd.Series([10000, 10000])) == 0.55


def test_undefined_returns_nan_not_zero():
    assert math.isnan(m.sharpe(pd.Series([0.01, 0.01, 0.01])))  # zero variance
    assert math.isnan(m.total_return(pd.Series([100])))  # single point
    assert math.isnan(m.max_drawdown(pd.Series([100])))


def test_summary_includes_se_and_benchmark():
    eq = pd.Series([100, 101, 102, 101, 103, 105])
    bench = pd.Series([100, 100.5, 101, 100.5, 101.5, 102])
    s = m.summary(eq, bench)
    assert set(["sharpe", "sharpe_se", "max_drawdown", "alpha_annual", "beta"]).issubset(s)
    assert not math.isnan(s["sharpe_se"])
