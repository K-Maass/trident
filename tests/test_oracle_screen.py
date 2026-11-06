"""Deterministic screen — ranks by composite z-score, always re-underwrites held names, records
how many factors actually landed, and refuses a mis-scaled share count. Offline."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from trident.agents.analyst import screen as S
from trident.data import fundamentals as f

ASOF = datetime(2026, 7, 21, tzinfo=UTC)


def _row(fy, filed, **vals):
    base = {name: None for name, _, _ in f._LINE_ITEMS}
    return base | {"fy": fy, "end": f"{fy}-12-31", "filed": filed, "form": "10-K"} | vals


def _fin(*rows):
    return {
        "ticker": "X",
        "fetched_at": "2026-07-21T00:00:00+00:00",
        "fiscal_years": list(rows),
        "concepts_used": {},
        "missing_concepts": [],
    }


def _profile(q: float, shares: float = 10_000_000.0):
    """q scales profitability: HIGHQ gets 1.0, LOWQ 0.2."""
    return dict(
        revenue=1000.0,
        net_income=250.0 * q,
        operating_income=300.0 * q,
        stockholders_equity=800.0,
        operating_cash_flow=300.0 * q,
        capital_expenditures=100.0,
        income_tax_expense=50.0,
        cash_and_equivalents=100.0,
        long_term_debt=200.0,
        diluted_shares=shares,
    )


def _setup(monkeypatch, shares_by_ticker=None):
    shares_by_ticker = shares_by_ticker or {}

    def fake_financials(ticker, years=5, **k):
        t = ticker.upper()
        q = {"HIGHQ": 1.0, "LOWQ": 0.2}.get(t, 0.5)
        sh = shares_by_ticker.get(t, 10_000_000.0)
        return _fin(
            _row(2025, "2026-02-15", **_profile(q, sh)),
            _row(2024, "2025-02-15", **(_profile(q, sh) | {"revenue": 800.0})),
        )

    monkeypatch.setattr(f, "financials", fake_financials)
    monkeypatch.setattr(
        S.market,
        "get_history",
        lambda tickers, start, end, **k: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-07-20"]),
                "ticker": [(tickers[0] if isinstance(tickers, list) else tickers).upper()],
                "open": [99.0],
                "high": [101.0],
                "low": [98.0],
                "close": [100.0],
                "volume": [1e7],
            }
        ),
    )


def test_screen_ranks_higher_quality_first_and_returns_the_full_table(monkeypatch):
    _setup(monkeypatch)
    res = S.run_screen(["HIGHQ", "LOWQ"], ASOF, held=[], top_n=1)
    assert len(res.table) == 2  # the full ranked table stays inspectable
    assert res.table.iloc[0]["ticker"] == "HIGHQ"
    assert res.candidates == ["HIGHQ"]


def test_held_names_are_always_candidates_even_below_the_cut(monkeypatch):
    _setup(monkeypatch)
    res = S.run_screen(["HIGHQ", "LOWQ"], ASOF, held=["LOWQ"], top_n=1)
    assert "HIGHQ" in res.candidates and "LOWQ" in res.candidates  # holdings re-underwritten


def test_n_factors_records_how_much_evidence_each_row_rests_on(monkeypatch):
    _setup(monkeypatch)
    res = S.run_screen(["HIGHQ", "LOWQ"], ASOF, top_n=2)
    assert set(res.table["n_factors"]) == {4}  # all four factors landed for this fixture


def test_a_mis_scaled_share_count_cannot_hijack_the_ranking(monkeypatch):
    """A count tagged in millions would post an earnings yield ~1e9x too high and take rank 1.
    The guard drops the factor instead, leaving the name ranked on its other three."""
    _setup(monkeypatch, shares_by_ticker={"LOWQ": 716.4})
    res = S.run_screen(["HIGHQ", "LOWQ"], ASOF, top_n=2)
    lowq = res.table.set_index("ticker").loc["LOWQ"]
    assert pd.isna(lowq["earnings_yield"]) and lowq["n_factors"] == 3
    assert res.table.iloc[0]["ticker"] == "HIGHQ"  # the bad row did not seize the top


def test_a_ticker_with_no_visible_filings_scores_neutral_not_negative(monkeypatch):
    """Every factor NaN -> z-scores of 0 -> composite 0. Absent evidence must not read as a
    strong negative signal (nor a positive one)."""
    _setup(monkeypatch)
    monkeypatch.setattr(
        f,
        "financials",
        lambda ticker, years=5, **k: (
            _fin() if ticker.upper() == "DARK" else _fin(_row(2025, "2026-02-15", **_profile(1.0)))
        ),
    )
    res = S.run_screen(["HIGHQ", "DARK"], ASOF, top_n=2)
    dark = res.table.set_index("ticker").loc["DARK"]
    assert dark["n_factors"] == 0 and dark["composite"] == pytest.approx(0.0)
