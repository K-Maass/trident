"""Freshness-job tests. Market/fundamentals checks run offline against tmp caches with a
fixed asof; the news check + orchestrator need Postgres (integration)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from trident.data import freshness, market

UTC = UTC
# After the 2026-07-06 close (20:00 UTC), the last completed session is 2026-07-06.
ASOF = datetime(2026, 7, 6, 21, 0, tzinfo=UTC)


def _write_bars(cache_dir, ticker, last_day):
    idx = pd.bdate_range("2026-05-01", last_day, name="date")
    df = pd.DataFrame(
        {c: range(len(idx)) for c in ["open", "high", "low", "close", "volume"]}, index=idx
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(market._cache_path(ticker, cache_dir))


# --------------------------------------------------------------- market bars
def test_market_bars_fresh_when_current(tmp_path):
    _write_bars(tmp_path, "AAPL", "2026-07-06")  # through the last completed session
    r = freshness.check_market_bars(["AAPL"], cache_dir=tmp_path, asof=ASOF)
    assert r.ok, r.detail


def test_market_bars_stale_when_old(tmp_path):
    _write_bars(tmp_path, "AAPL", "2026-06-10")  # weeks behind
    r = freshness.check_market_bars(["AAPL"], cache_dir=tmp_path, asof=ASOF)
    assert not r.ok
    assert "AAPL" in r.detail


def test_market_bars_missing_cache_is_stale(tmp_path):
    r = freshness.check_market_bars(["ZZZZ"], cache_dir=tmp_path, asof=ASOF)
    assert not r.ok
    assert "no-cache" in r.detail


# --------------------------------------------------------------- fundamentals
def test_fundamentals_fresh(tmp_path):
    (tmp_path / "AAPL.json").write_text(json.dumps({"_fetched_at": datetime.now(UTC).isoformat()}))
    r = freshness.check_fundamentals(["AAPL"], cache_dir=tmp_path)
    assert r.ok, r.detail


def test_fundamentals_missing_and_stale(tmp_path):
    r_missing = freshness.check_fundamentals(["AAPL"], cache_dir=tmp_path)
    assert not r_missing.ok and "missing" in r_missing.detail
    old = (datetime.now(UTC) - timedelta(days=200)).isoformat()
    (tmp_path / "MSFT.json").write_text(json.dumps({"_fetched_at": old}))
    r_stale = freshness.check_fundamentals(["MSFT"], cache_dir=tmp_path, max_stale_days=120)
    assert not r_stale.ok and "stale" in r_stale.detail


# ------------------------------------------------------------- orchestrator
@pytest.mark.integration
async def test_run_freshness_check_writes_health_event(tmp_path):
    # Empty caches -> market+fundamentals stale -> WARNs written; news reflects live DB.
    from trident.data import store

    try:
        report = await freshness.run_freshness_check(asof=ASOF, tickers=["AAPL"])
        sources = {r.source for r in report.results}
        assert sources == {"market_bars", "fundamentals", "news"}
        assert not report.all_ok  # no market/fundamentals cache in this tmp run
    finally:
        # clean the WARN rows this test emitted
        eng = store.get_engine()
        from sqlalchemy import text

        async with eng.begin() as conn:
            await conn.execute(
                text("DELETE FROM health_events WHERE ts = :ts AND source LIKE 'freshness%'"),
                {"ts": ASOF},
            )
        await store.dispose_engine()
