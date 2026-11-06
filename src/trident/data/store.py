"""Data-layer persistence (Phase 1 ingestion writes only).

Scope is deliberately narrow: news_items, aggregator signal snapshots, and
health_events. The money-path stores (ledger/orders/fills) are Phase 2 and live
elsewhere — this module never touches them.

Cache policy (doc 02): event-like data → Postgres with fetched_at, cache-first,
dedupe on write. url_hash is the news_items PK, so ON CONFLICT DO NOTHING gives us
exact-duplicate rejection for free; fuzzy-title dedupe happens upstream in news.py.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

from sqlalchemy import text

from trident.db import dispose_engine, get_engine  # re-exported for existing callers
from trident.models import HealthEvent, HealthSeverity, NewsItem

__all__ = [
    "get_engine",
    "dispose_engine",
    "insert_news_items",
    "recent_news_titles",
    "insert_signal_snapshots",
    "write_health_event",
    "get_news",
    "news_needing_relevance",
    "news_needing_sentiment",
    "set_news_relevance",
    "set_news_sentiment",
    "get_signals",
    "delete_signals",
]


_INSERT_NEWS = text(
    """
    INSERT INTO news_items (url_hash, source, ticker, title, body, url, published_at)
    VALUES (:url_hash, :source, :ticker, :title, :body, :url, :published_at)
    ON CONFLICT (url_hash) DO NOTHING
    """
)


_SELECT_KNOWN_HASHES = text("SELECT url_hash, ticker FROM news_items WHERE url_hash = ANY(:hashes)")
_INSERT_COLLISION = text(
    """
    INSERT INTO news_url_collisions (url_hash, stored_ticker, attempted_ticker, attempted_source)
    VALUES (:url_hash, :stored_ticker, :attempted_ticker, :attempted_source)
    """
)


async def insert_news_items(items: list[NewsItem], log_collisions: bool = True) -> int:
    """Insert deduped-on-url_hash; returns the count of NEW rows (conflicts skipped).

    Because url_hash keys on the URL alone, an article that already exists under one ticker will
    conflict when it arrives under a DIFFERENT ticker — and ON CONFLICT DO NOTHING would discard
    that second (url, ticker) pairing without a trace. When `log_collisions`, we record those
    cross-ticker pairings to news_url_collisions first (prospective attribution-bias data for the
    pre-registered 2026-07-22 check). Detection covers both already-stored rows and duplicates
    that appear twice within this same batch.
    """
    if not items:
        return 0
    engine = get_engine()
    hashes = list({it.url_hash for it in items})
    collisions: list[dict] = []
    inserted = 0
    async with engine.begin() as conn:
        rows = (await conn.execute(_SELECT_KNOWN_HASHES, {"hashes": hashes})).all()
        known: dict[str, str | None] = {r[0]: r[1] for r in rows}  # url_hash -> stored ticker
        for it in items:
            prior = known.get(it.url_hash)
            if prior is not None and it.ticker and prior != it.ticker:
                collisions.append(
                    {
                        "url_hash": it.url_hash,
                        "stored_ticker": prior,
                        "attempted_ticker": it.ticker,
                        "attempted_source": it.source,
                    }
                )
            res = await conn.execute(
                _INSERT_NEWS,
                {
                    "url_hash": it.url_hash,
                    "source": it.source,
                    "ticker": it.ticker,
                    "title": it.title,
                    "body": it.body,
                    "url": it.url,
                    "published_at": it.published_at,
                },
            )
            n = res.rowcount or 0
            if n and it.url_hash not in known:
                known[it.url_hash] = it.ticker  # first-writer wins; later dups compare to it
            inserted += n

    # Collisions are observational, not trading data — persist best-effort in a separate txn so a
    # logging hiccup can never roll back a good ingest (golden rule 8: degrade, don't break).
    if log_collisions and collisions:
        try:
            async with engine.begin() as conn:
                await conn.execute(_INSERT_COLLISION, collisions)
        except Exception as e:  # noqa: BLE001
            await write_health_event(
                HealthEvent(
                    ts=datetime.now(UTC),
                    severity=HealthSeverity.WARN,
                    source="news.collisions",
                    detail=f"failed to log {len(collisions)} url collisions: {type(e).__name__}",
                )
            )
    return inserted


_RECENT_TITLES = text(
    "SELECT ticker, title FROM news_items WHERE published_at >= :since ORDER BY published_at DESC"
)


async def recent_news_titles(since: datetime) -> list[tuple[str | None, str]]:
    """(ticker, title) for items since `since` — the corpus fuzzy dedupe checks against."""
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(_RECENT_TITLES, {"since": since})).all()
    return [(r[0], r[1]) for r in rows]


async def ticker_for_hashes(hashes: list[str]) -> dict[str, str | None]:
    """url_hash -> stored ticker, for the hashes present in news_items. Used by the dry
    collision re-fetch to compare an item's fetched ticker against the one it is stored under."""
    if not hashes:
        return {}
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (await conn.execute(_SELECT_KNOWN_HASHES, {"hashes": hashes})).all()
    return {r[0]: r[1] for r in rows}


async def equity_snapshot_exists(agent: str, session: date) -> bool:
    """True if any equity snapshot is already stored for this agent on this session date.
    The daily-snapshot job uses this to stay one-per-(agent, session) and to defer to the
    richer post-trade snapshot that run_agent writes on rebalance/deploy days."""
    async with get_engine().connect() as conn:
        n = (
            await conn.execute(
                text("SELECT count(*) FROM equity_snapshots WHERE agent_id=:a AND ts::date=:d"),
                {"a": agent, "d": session},
            )
        ).scalar_one()
    return int(n) > 0


async def collision_stats() -> dict:
    """Summarise logged cross-ticker URL collisions (prospective, from news_url_collisions)."""
    engine = get_engine()
    async with engine.connect() as conn:
        total = (await conn.execute(text("SELECT count(*) FROM news_url_collisions"))).scalar_one()
        distinct_urls = (
            await conn.execute(text("SELECT count(DISTINCT url_hash) FROM news_url_collisions"))
        ).scalar_one()
    return {"logged_collisions": int(total), "distinct_urls": int(distinct_urls)}


_INSERT_SIGNAL = text(
    """
    INSERT INTO signals (agent_id, ts, ticker, score, metadata)
    VALUES (:agent_id, :ts, :ticker, :score, CAST(:metadata AS JSONB))
    """
)


async def insert_signal_snapshots(rows: list[dict]) -> int:
    """Aggregator snapshots (ApeWisdom/Tradestie) → signals.metadata with fetched_at (doc 04).

    Each row: {agent_id, ts, ticker, score, metadata(dict)}. attention_only sources pass
    score=0.0 (placeholder); the attention payload lives in metadata.
    """
    if not rows:
        return 0
    engine = get_engine()
    async with engine.begin() as conn:
        for r in rows:
            await conn.execute(
                _INSERT_SIGNAL,
                {
                    "agent_id": r["agent_id"],
                    "ts": r["ts"],
                    "ticker": r["ticker"],
                    "score": r["score"],
                    "metadata": json.dumps(r.get("metadata", {})),
                },
            )
    return len(rows)


_INSERT_HEALTH = text(
    "INSERT INTO health_events (ts, severity, source, detail) "
    "VALUES (:ts, :severity, :source, :detail)"
)


async def write_health_event(event: HealthEvent) -> None:
    """Fail-safe telemetry (golden rule 8). Ingestion degrades to a WARN, never a crash."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(
            _INSERT_HEALTH,
            {
                "ts": event.ts,
                "severity": event.severity.value,
                "source": event.source,
                "detail": event.detail,
            },
        )


# ------------------------------------------------------ SENTINEL reads/scoring (Phase 4)
_NEWS_COLS = "url_hash, source, ticker, title, body, url, published_at, relevant, sentiment"


def _news_rows(rows) -> list[dict]:
    return [dict(r._mapping) for r in rows]


async def get_news(
    tickers: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    only_relevant: bool = False,
) -> list[dict]:
    """Full news_items rows for a ticker set + published_at window (uses ix_news_ticker_ts).
    `only_relevant=True` returns just items the Haiku filter passed (relevant IS TRUE)."""
    where = ["TRUE"]
    params: dict = {}
    if tickers is not None:
        where.append("ticker = ANY(:tickers)")
        params["tickers"] = [t.upper() for t in tickers]
    if since is not None:
        where.append("published_at >= :since")
        params["since"] = since
    if until is not None:
        where.append("published_at < :until")
        params["until"] = until
    if only_relevant:
        where.append("relevant IS TRUE")
    sql = (
        f"SELECT {_NEWS_COLS} FROM news_items WHERE {' AND '.join(where)} ORDER BY published_at ASC"
    )
    async with get_engine().connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return _news_rows(rows)


async def news_needing_relevance(
    since: datetime | None = None, limit: int | None = None
) -> list[dict]:
    """raw_text items the Haiku filter hasn't judged yet (relevant IS NULL). All news_items
    are raw_text (aggregators write to signals, not here), so no source filter is needed."""
    where = ["relevant IS NULL"]
    params: dict = {}
    if since is not None:
        where.append("published_at >= :since")
        params["since"] = since
    sql = (
        f"SELECT {_NEWS_COLS} FROM news_items WHERE {' AND '.join(where)} ORDER BY published_at ASC"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    async with get_engine().connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return _news_rows(rows)


async def news_needing_sentiment(
    since: datetime | None = None, limit: int | None = None
) -> list[dict]:
    """Relevant items FinBERT hasn't scored yet (relevant IS TRUE AND sentiment IS NULL)."""
    where = ["relevant IS TRUE", "sentiment IS NULL"]
    params: dict = {}
    if since is not None:
        where.append("published_at >= :since")
        params["since"] = since
    sql = (
        f"SELECT {_NEWS_COLS} FROM news_items WHERE {' AND '.join(where)} ORDER BY published_at ASC"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    async with get_engine().connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return _news_rows(rows)


async def set_news_relevance(verdicts: dict[str, bool]) -> int:
    """Backfill news_items.relevant by url_hash (the Haiku-verdict cache). Returns row count."""
    if not verdicts:
        return 0
    payload = [{"h": h, "v": bool(v)} for h, v in verdicts.items()]
    async with get_engine().begin() as conn:
        await conn.execute(text("UPDATE news_items SET relevant = :v WHERE url_hash = :h"), payload)
    return len(payload)


async def set_news_sentiment(scores: dict[str, float]) -> int:
    """Backfill news_items.sentiment by url_hash (the FinBERT-score cache). Returns row count."""
    if not scores:
        return 0
    payload = [{"h": h, "s": float(s)} for h, s in scores.items()]
    async with get_engine().begin() as conn:
        await conn.execute(
            text("UPDATE news_items SET sentiment = :s WHERE url_hash = :h"), payload
        )
    return len(payload)


async def get_signals(
    agent: str,
    tickers: list[str] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict]:
    """Read persisted signals (attention snapshots + SENTINEL aggregates) for one agent."""
    where = ["agent_id = :a"]
    params: dict = {"a": agent}
    if tickers is not None:
        where.append("ticker = ANY(:tickers)")
        params["tickers"] = [t.upper() for t in tickers]
    if since is not None:
        where.append("ts >= :since")
        params["since"] = since
    if until is not None:
        where.append("ts < :until")
        params["until"] = until
    sql = (
        "SELECT ts, ticker, score, metadata FROM signals "
        f"WHERE {' AND '.join(where)} ORDER BY ts ASC"
    )
    async with get_engine().connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return [dict(r._mapping) for r in rows]


async def sample_for_review(
    n_per_class: int = 10, seed: str = "weekly", exclude_labeled: bool = True
) -> list[dict]:
    """Stratified sample for the weekly relevance spot-check: n predicted-YES + n predicted-NO.

    Stratified, not random: a plain random draw is ~58% NO and would under-sample the YES class,
    which is the expensive failure (noise reaching FinBERT and moving sent_raw).

    Only items the MODEL actually judged (ticker IS NOT NULL) — ticker-less items are auto-NO
    with no API call, so grading them would measure nothing. `exclude_labeled` keeps successive
    weeks from re-drawing items a human has already adjudicated, so the sample accumulates
    rather than resampling the same rows.
    """
    base = """
        SELECT url_hash, ticker, source, title, url, relevant AS model_verdict
        FROM news_items
        WHERE relevant IS {want} AND ticker IS NOT NULL
    """
    if exclude_labeled:
        base += " AND url_hash NOT IN (SELECT url_hash FROM relevance_labels)"
    base += " ORDER BY md5(url_hash || :seed) LIMIT :n"

    out: list[dict] = []
    async with get_engine().connect() as conn:
        for want in ("TRUE", "FALSE"):
            rows = (
                await conn.execute(
                    text(base.replace("{want}", want)), {"seed": seed, "n": n_per_class}
                )
            ).all()
            out.extend(dict(r._mapping) for r in rows)
    return out


async def insert_relevance_labels(rows: list[dict]) -> int:
    """Persist human adjudications. NEVER touches news_items.relevant — the model's verdict is
    history. Re-labelling the same (url_hash, reviewer) updates that reviewer's own label only."""
    if not rows:
        return 0
    sql = text(
        """
        INSERT INTO relevance_labels
            (url_hash, reviewer, label, model_verdict, note, sample_id)
        VALUES (:url_hash, :reviewer, :label, :model_verdict, :note, :sample_id)
        ON CONFLICT (url_hash, reviewer) DO UPDATE
            SET label = EXCLUDED.label, note = EXCLUDED.note, labeled_at = now()
        """
    )
    async with get_engine().begin() as conn:
        for r in rows:
            await conn.execute(sql, {"note": None, **r})
    return len(rows)


async def get_relevance_labels(sample_id: str | None = None) -> list[dict]:
    sql = "SELECT * FROM relevance_labels"
    params: dict = {}
    if sample_id:
        sql += " WHERE sample_id = :s"
        params["s"] = sample_id
    async with get_engine().connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return [dict(r._mapping) for r in rows]


async def delete_signals(
    agent: str,
    kind: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> int:
    """Delete signal rows of a given metadata 'kind' (e.g. 'sentiment_aggregate') for an agent
    in a ts window — the delete half of aggregation's idempotent delete-then-insert."""
    where = ["agent_id = :a", "metadata->>'kind' = :k"]
    params: dict = {"a": agent, "k": kind}
    if since is not None:
        where.append("ts >= :since")
        params["since"] = since
    if until is not None:
        where.append("ts < :until")
        params["until"] = until
    sql = f"DELETE FROM signals WHERE {' AND '.join(where)}"
    async with get_engine().begin() as conn:
        res = await conn.execute(text(sql), params)
    return res.rowcount or 0
