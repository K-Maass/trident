"""SENTINEL ingestion tests (doc 04). Parsing + dedupe run fully offline against
synthetic fixtures. The DB cache/dedupe
check is integration-marked (needs Postgres; CI excludes it)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trident.data.news import dedupe_fuzzy, parse_finnhub, parse_rss
from trident.models import NewsItem

FIX = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------- parsing
def test_parse_finnhub_normalizes_to_utc_newsitems():
    rows = json.loads((FIX / "finnhub_demo.json").read_text())
    items = parse_finnhub(rows, "AAPL")
    assert items, "fixture should yield items"
    assert len(items) == len(rows)  # every fixture row has url + datetime
    it = items[0]
    assert it.source == "finnhub"
    assert it.ticker == "AAPL"
    assert it.published_at.tzinfo is not None  # tz-aware
    assert it.published_at.utcoffset() == UTC.utcoffset(None)
    assert it.url_hash  # auto-computed, non-empty (dedup key)


def test_parse_finnhub_skips_rows_missing_url_or_datetime():
    good = {"datetime": 1783151400, "headline": "H", "summary": "S", "url": "https://x/1"}
    no_url = {"datetime": 1783151400, "headline": "H", "url": ""}
    no_dt = {"datetime": None, "headline": "H", "url": "https://x/2"}
    items = parse_finnhub([good, no_url, no_dt], "MSFT")
    assert len(items) == 1
    assert items[0].url == "https://x/1"


def test_parse_rss_reads_title_link_and_time():
    content = (FIX / "rss_demo.xml").read_bytes()
    items = parse_rss(content, "yahoo", "AAPL")
    assert len(items) >= 5
    assert all(i.source == "rss:yahoo" for i in items)
    assert all(i.ticker == "AAPL" for i in items)
    assert all(i.url and i.title for i in items)
    assert all(i.published_at.tzinfo is not None for i in items)


# ---------------------------------------------------------------------- dedupe
def _item(title: str, url: str, ticker: str | None = "AAPL") -> NewsItem:
    return NewsItem(
        source="finnhub",
        ticker=ticker,
        title=title,
        url=url,
        published_at=datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
    )


def test_dedupe_fuzzy_drops_near_duplicate_titles_same_ticker():
    batch = [
        _item("Apple beats Q3 earnings estimates on strong iPhone sales", "https://a/1"),
        _item(
            "Apple beats Q3 earnings estimates on strong iPhone sales!", "https://a/2"
        ),  # near-dup
        _item("Apple unveils new Vision Pro headset at WWDC", "https://a/3"),  # distinct
    ]
    kept = dedupe_fuzzy(batch, existing=[])
    titles = {k.title for k in kept}
    assert len(kept) == 2
    assert "Apple unveils new Vision Pro headset at WWDC" in titles


def test_dedupe_fuzzy_respects_ticker_boundary():
    # Same headline text, different tickers => NOT a duplicate (no cross-name merge).
    batch = [
        _item("Fed decision sends markets higher", "https://a/1", ticker="AAPL"),
        _item("Fed decision sends markets higher", "https://a/2", ticker="MSFT"),
    ]
    assert len(dedupe_fuzzy(batch, existing=[])) == 2


def test_dedupe_fuzzy_checks_against_existing_db_titles():
    existing = [("AAPL", "Apple beats Q3 earnings estimates on strong iPhone sales")]
    batch = [_item("Apple beats Q3 earnings estimates on strong iPhone sales.", "https://a/9")]
    assert dedupe_fuzzy(batch, existing=existing) == []  # already banked


# ------------------------------------------------------------------ DB cache
@pytest.mark.integration
async def test_news_items_url_hash_dedupe_at_db():
    """Inserting the same item twice yields one row — url_hash PK is the cache key."""
    from trident.data import store

    item = NewsItem(
        source="test",
        ticker="AAPL",
        title="phase1 cache dedupe probe",
        url="https://trident.test/phase1-dedupe-probe",
        published_at=datetime.now(UTC),
    )
    try:
        first = await store.insert_news_items([item])
        second = await store.insert_news_items([item])  # same url_hash
        assert first == 1
        assert second == 0  # ON CONFLICT DO NOTHING
    finally:
        from sqlalchemy import text

        eng = store.get_engine()
        async with eng.begin() as conn:
            await conn.execute(
                text("DELETE FROM news_items WHERE url_hash = :h"), {"h": item.url_hash}
            )
        await store.dispose_engine()


@pytest.mark.integration
async def test_cross_ticker_collision_is_logged_not_lost():
    """Same URL, different ticker → the discarded (url, ticker) pairing is recorded in
    news_url_collisions (attribution-bias data for the 2026-07-22 check), and news_items keeps
    the FIRST ticker (url_hash PK is URL-only)."""
    from sqlalchemy import text

    from trident.data import store

    url = "https://trident.test/multi-company-collision-probe"
    a = NewsItem(
        source="finnhub", ticker="MSFT", title="A buys B", url=url, published_at=datetime.now(UTC)
    )
    b = NewsItem(
        source="finnhub", ticker="AMZN", title="A buys B", url=url, published_at=datetime.now(UTC)
    )
    assert a.url_hash == b.url_hash  # URL-only hash → same key, different ticker
    try:
        assert await store.insert_news_items([a]) == 1
        assert await store.insert_news_items([b]) == 0  # conflict: AMZN pairing discarded
        stored = await store.ticker_for_hashes([url_h := a.url_hash])
        assert stored[url_h] == "MSFT"  # first writer won

        eng = store.get_engine()
        async with eng.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT stored_ticker, attempted_ticker FROM news_url_collisions "
                        "WHERE url_hash = :h"
                    ),
                    {"h": a.url_hash},
                )
            ).all()
        assert (("MSFT", "AMZN")) in [(r[0], r[1]) for r in rows]
    finally:
        eng = store.get_engine()
        async with eng.begin() as conn:
            await conn.execute(
                text("DELETE FROM news_url_collisions WHERE url_hash = :h"), {"h": a.url_hash}
            )
            await conn.execute(
                text("DELETE FROM news_items WHERE url_hash = :h"), {"h": a.url_hash}
            )
        await store.dispose_engine()
