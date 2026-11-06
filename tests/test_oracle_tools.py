"""ORACLE research tools — anti-look-ahead binding, the never-guess fail-safe, staleness,
provenance pass-through, and the share-count plausibility guard. Fully offline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from trident.agents.analyst import tools as T
from trident.agents.analyst.tools import ToolContext
from trident.data import fundamentals as f

ASOF = datetime(2026, 7, 21, tzinfo=UTC)


def _row(fy, filed, **vals):
    base = {name: None for name, _, _ in f._LINE_ITEMS}
    return base | {"fy": fy, "end": f"{fy}-12-31", "filed": filed, "form": "10-K"} | vals


def _fin(*rows, fetched="2026-07-21T00:00:00+00:00"):
    return {
        "ticker": "TEST",
        "fetched_at": fetched,
        "fiscal_years": list(rows),
        "concepts_used": {"revenue": ["Revenues"]},
        "missing_concepts": [],
    }


_FULL = dict(
    revenue=1000.0,
    net_income=250.0,
    operating_income=300.0,
    stockholders_equity=800.0,
    operating_cash_flow=300.0,
    capital_expenditures=100.0,
    depreciation_amortization=50.0,
    diluted_shares=10_000_000.0,
)


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    # FY2025 was filed before asof; FY2026 was filed AFTER and must never be visible.
    fin = _fin(
        _row(2026, "2027-02-15", **_FULL),
        _row(2025, "2026-02-15", **_FULL),
        _row(2024, "2025-02-15", **(_FULL | {"revenue": 800.0})),
    )
    monkeypatch.setattr(f, "financials", lambda ticker, years=5, **k: fin)
    monkeypatch.setattr(
        T.market,
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


async def test_financials_hide_filings_made_after_asof():
    fin = await T.get_financials(ToolContext(ASOF), "test")
    fys = [r["fy"] for r in fin["fiscal_years"]]
    assert fin["available"] and 2025 in fys and 2026 not in fys  # future filing excluded


async def test_growth_is_computed_only_from_visible_years():
    """The asof filter runs BEFORE derived_metrics, so YoY growth compares 2025 vs 2024 — never
    against the invisible 2026 filing."""
    r = await T.get_ratios(ToolContext(ASOF), "test")
    latest = r["quality_by_year"][0]
    assert latest["fy"] == 2025
    g = latest["metrics"]["revenue_growth_yoy"]
    assert g["inputs"]["revenue_prior"] == 800.0  # the 2024 row
    assert g["value"] == pytest.approx(0.25)


async def test_missing_fundamentals_returns_unavailable(monkeypatch):
    monkeypatch.setattr(f, "financials", lambda ticker, years=5, **k: _fin())
    out = await T.get_financials(ToolContext(ASOF), "zzz")
    assert out["available"] is False and "reason" in out


async def test_stale_flag_set_for_old_cache(monkeypatch):
    old = (datetime.now(UTC) - timedelta(days=T.STALE_DAYS + 5)).isoformat()
    monkeypatch.setattr(
        f,
        "financials",
        lambda ticker, years=5, **k: _fin(_row(2025, "2026-02-15", **_FULL), fetched=old),
    )
    r = await T.get_ratios(ToolContext(ASOF), "test")
    assert r["available"] and r["stale"] is True


async def test_ratios_carry_provenance_and_are_json_clean():
    r = await T.get_ratios(ToolContext(ASOF), "test")
    json.dumps(r, allow_nan=False)  # the runner refuses non-JSON extras
    nm = r["quality_by_year"][0]["metrics"]["net_margin"]
    assert set(nm) == {"value", "formula", "inputs"} and nm["value"] == pytest.approx(0.25)
    pe = r["valuation"]["pe"]
    assert set(pe) == {"value", "formula", "inputs"}  # tool-computed metrics too
    assert pe["value"] == pytest.approx(100.0 * 10_000_000.0 / 250.0)


async def test_implausible_share_count_suppresses_valuation(monkeypatch):
    """A count tagged in millions (MCD FY2025: 716.4) would turn a $191B market cap into $191K.
    We refuse the number rather than infer the scale — golden rule 8."""
    monkeypatch.setattr(
        f,
        "financials",
        lambda ticker, years=5, **k: _fin(
            _row(2025, "2026-02-15", **(_FULL | {"diluted_shares": 716.4}))
        ),
    )
    r = await T.get_ratios(ToolContext(ASOF), "test")
    assert r["available"]
    assert r["valuation"]["market_cap"]["value"] is None
    assert r["valuation"]["pe"]["value"] is None
    assert r["data_quality"] and "716.4" in r["data_quality"][0]
    # quality ratios are share-independent and must survive
    assert r["quality_by_year"][0]["metrics"]["net_margin"]["value"] == pytest.approx(0.25)


async def test_recent_news_binds_until_asof_and_tails_n(monkeypatch):
    captured = {}

    async def fake_get_news(tickers=None, until=None, only_relevant=False, **k):
        captured.update(until=until, only_relevant=only_relevant)
        return [
            {
                "title": f"n{i}",
                "source": "rss",
                "url": f"u{i}",
                "published_at": datetime(2026, 5, i + 1, tzinfo=UTC),
            }
            for i in range(5)
        ]

    monkeypatch.setattr(T.store, "get_news", fake_get_news)
    out = await T.get_recent_news(ToolContext(ASOF), "test", n=2)
    assert captured["until"] == ASOF and captured["only_relevant"] is True
    assert out["count"] == 2 and out["items"][0]["title"] == "n4"  # newest first, tail-2
