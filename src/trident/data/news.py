"""SENTINEL ingestion (doc 04) — the Phase 1 "start collecting now" collector.

Banks raw media from day one so SENTINEL has backtest history long before it trades
(Phase 4). This module ONLY fetches, normalizes, dedupes, and persists:
  raw_text (Finnhub, RSS)            -> news_items   (dedupe url_hash + fuzzy title)
  aggregator snapshots (ApeWisdom)   -> signals      (metadata + fetched_at)
Scoring (FinBERT) and the Haiku relevance filter are Phase 4 — items land unscored
(news_items.relevant / .sentiment stay NULL).

Source registry is config/sentinel_sources.yaml (doc 04): a source that is disabled,
missing its env keys, or erroring contributes nothing and a WARN health event fires —
never a crash, never a silently-reused stale value.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import feedparser
import httpx
from rapidfuzz import fuzz
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trident.config import get_settings, load_sentinel_sources, load_universe
from trident.data import store
from trident.models import HealthEvent, HealthSeverity, NewsItem

# Retry these (transient); NOT HTTPStatusError — a 4xx/5xx is a reject, don't hammer it.
NETWORK_ERRORS = (httpx.TransportError, httpx.TimeoutException)
FUZZY_TITLE_THRESHOLD = 90  # rapidfuzz token_set_ratio > 90 => same story (doc 04)


# --------------------------------------------------------------------------- utils
class AsyncRateLimiter:
    """Minimum-interval limiter (token-bucket at ~1 permit / interval). 80%-of-limit
    ceilings come from sentinel_sources.yaml, not hardcoded here."""

    def __init__(self, per_min: float) -> None:
        self._interval = 60.0 / per_min if per_min and per_min > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next = now + self._interval


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception_type(NETWORK_ERRORS),
    reraise=True,
)
async def _get(client: httpx.AsyncClient, url: str, params: dict | None = None) -> httpx.Response:
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r


def _rss_time(entry: dict) -> datetime:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    if st:
        # feedparser normalizes to UTC struct_time
        return datetime(*st[:6], tzinfo=UTC)
    return datetime.now(UTC)


def _feed_name(url: str) -> str:
    host = httpx.URL(url).host or "feed"
    return host.replace("www.", "").split(".")[0]


def parse_rss(content: bytes, source_name: str, ticker: str | None) -> list[NewsItem]:
    """Parse raw feed bytes into NewsItems. Pure + offline-testable (fixture-friendly)."""
    feed = feedparser.parse(content)
    out: list[NewsItem] = []
    for e in feed.entries:
        link = e.get("link")
        title = e.get("title")
        if not link or not title:
            continue
        out.append(
            NewsItem(
                source=f"rss:{source_name}",
                ticker=ticker,
                title=title,
                body=e.get("summary") or None,
                url=link,
                published_at=_rss_time(e),
            )
        )
    return out


def parse_finnhub(rows: list[dict], ticker: str) -> list[NewsItem]:
    """Normalize Finnhub /company-news rows for one ticker. Pure + offline-testable."""
    out: list[NewsItem] = []
    for row in rows:
        ts = row.get("datetime")
        url = row.get("url")
        if not ts or not url:
            continue
        out.append(
            NewsItem(
                source="finnhub",
                ticker=ticker,
                title=row.get("headline") or "(no headline)",
                body=row.get("summary") or None,
                url=url,
                published_at=datetime.fromtimestamp(int(ts), tz=UTC),
            )
        )
    return out


def dedupe_fuzzy(
    items: list[NewsItem],
    existing: list[tuple[str | None, str]],
    threshold: int = FUZZY_TITLE_THRESHOLD,
) -> list[NewsItem]:
    """Drop items whose (ticker, title) fuzzily matches an already-seen title.

    Complements the DB's exact url_hash dedupe: catches the same story republished at a
    different URL. Compared only within the same ticker to avoid cross-name false merges.
    `existing` is (ticker, title) already in the DB for the window.
    """
    seen = list(existing)
    kept: list[NewsItem] = []
    for it in items:
        dup = any(
            tk == it.ticker and fuzz.token_set_ratio(it.title, ttl) > threshold for tk, ttl in seen
        )
        if not dup:
            kept.append(it)
            seen.append((it.ticker, it.title))
    return kept


# ------------------------------------------------------------------------- sources
async def _fetch_finnhub(
    client: httpx.AsyncClient,
    tickers: list[str],
    key: str,
    frm: str,
    to: str,
    lim: AsyncRateLimiter,
) -> tuple[list[NewsItem], list[str]]:
    base = "https://finnhub.io/api/v1/company-news"
    items: list[NewsItem] = []
    errors: list[str] = []
    for t in tickers:
        await lim.acquire()
        try:
            r = await _get(client, base, params={"symbol": t, "from": frm, "to": to, "token": key})
            items.extend(parse_finnhub(r.json(), t))
        except Exception as e:  # noqa: BLE001 — per-ticker resilience
            errors.append(f"{t}: {type(e).__name__}")
    return items, errors


async def _fetch_rss(
    client: httpx.AsyncClient, feeds: list[str], tickers: list[str]
) -> tuple[list[NewsItem], list[str]]:
    items: list[NewsItem] = []
    errors: list[str] = []
    per_ticker = [f for f in feeds if "{ticker}" in f]
    market_wide = [f for f in feeds if "{ticker}" not in f]
    for url in market_wide:
        try:
            r = await _get(client, url)
            items.extend(parse_rss(r.content, _feed_name(url), None))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{_feed_name(url)}: {type(e).__name__}")
    for tmpl in per_ticker:
        name = _feed_name(tmpl)
        for t in tickers:
            try:
                r = await _get(client, tmpl.format(ticker=t))
                items.extend(parse_rss(r.content, name, t))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{name}:{t}: {type(e).__name__}")
    return items, errors


async def _fetch_apewisdom(
    client: httpx.AsyncClient,
    base_url: str,
    pages: int,
    universe: set[str],
    asof: datetime,
    lim: AsyncRateLimiter,
) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for page in range(1, pages + 1):
        await lim.acquire()
        try:
            r = await _get(client, base_url.format(page=page))
            results = r.json().get("results", [])
        except Exception as e:  # noqa: BLE001
            errors.append(f"page{page}: {type(e).__name__}")
            break
        for res in results:
            tk = (res.get("ticker") or "").upper()
            if tk not in universe:
                continue
            rows.append(
                {
                    "agent_id": "sentinel",
                    "ts": asof,
                    "ticker": tk,
                    "score": 0.0,  # attention_only: no sentiment; payload is in metadata
                    "metadata": {
                        "source": "apewisdom",
                        "kind": "attention_snapshot",
                        "rank": res.get("rank"),
                        "mentions": res.get("mentions"),
                        "upvotes": res.get("upvotes"),
                        "mentions_24h_ago": res.get("mentions_24h_ago"),
                        "name": res.get("name"),
                        "fetched_at": asof.isoformat(),
                    },
                }
            )
    return rows, errors


# --------------------------------------------------------------------- orchestrator
@dataclass
class SourceResult:
    name: str
    fetched: int = 0
    persisted: int = 0
    status: str = "ok"  # ok | skipped | error
    detail: str = ""


@dataclass
class CollectionReport:
    asof: datetime
    news_fetched: int = 0
    news_inserted: int = 0
    signals_inserted: int = 0
    sources: list[SourceResult] = field(default_factory=list)


async def collect(
    tickers: list[str] | None = None,
    asof: datetime | None = None,
    lookback_hours: int = 24,
    apewisdom_pages: int = 2,
    dry_run: bool = False,
) -> CollectionReport:
    """Run one ingestion cycle across the enabled registry sources.

    dry_run=True skips ALL DB writes (fetch + parse + dedupe only) — used by tests and
    for a no-side-effect connectivity check.
    """
    settings = get_settings()
    registry = load_sentinel_sources()
    universe = [t.upper() for t in (tickers or load_universe())]
    universe_set = set(universe)
    asof = asof or datetime.now(UTC)
    since = asof - timedelta(hours=lookback_hours)

    report = CollectionReport(asof=asof)
    news: list[NewsItem] = []
    signals: list[dict] = []

    async with httpx.AsyncClient(
        headers={"User-Agent": settings.news_user_agent}, timeout=15.0, follow_redirects=True
    ) as client:
        # --- Finnhub (raw_text; needs a key) ---
        cfg = registry.get("finnhub_news", {})
        if cfg.get("enabled"):
            if settings.finnhub_api_key:
                lim = AsyncRateLimiter(cfg.get("rate_limit_per_min", 48))
                items, errs = await _fetch_finnhub(
                    client,
                    universe,
                    settings.finnhub_api_key,
                    since.date().isoformat(),
                    asof.date().isoformat(),
                    lim,
                )
                news.extend(items)
                report.sources.append(
                    SourceResult(
                        "finnhub_news",
                        fetched=len(items),
                        status="error" if errs else "ok",
                        detail="; ".join(errs[:5]),
                    )
                )
            else:
                report.sources.append(
                    SourceResult("finnhub_news", status="skipped", detail="FINNHUB_API_KEY missing")
                )

        # --- RSS (raw_text; no key) ---
        cfg = registry.get("rss", {})
        if cfg.get("enabled"):
            items, errs = await _fetch_rss(client, cfg.get("feeds", []), universe)
            news.extend(items)
            report.sources.append(
                SourceResult(
                    "rss",
                    fetched=len(items),
                    status="error" if errs else "ok",
                    detail="; ".join(errs[:5]),
                )
            )

        # --- ApeWisdom (attention_only; no key) ---
        cfg = registry.get("apewisdom", {})
        if cfg.get("enabled"):
            base_url = cfg.get("base_url", "")
            lim = AsyncRateLimiter(60)  # gentle; polls_per_day governs cadence, not burst
            rows, errs = await _fetch_apewisdom(
                client, base_url, apewisdom_pages, universe_set, asof, lim
            )
            signals.extend(rows)
            report.sources.append(
                SourceResult(
                    "apewisdom",
                    fetched=len(rows),
                    status="error" if errs else "ok",
                    detail="; ".join(errs[:5]),
                )
            )

    # Never silently drop an enabled source: flag any that this MVP collector doesn't
    # yet handle (e.g. tradestie_wsb — implemented in the next increment; also currently
    # unreachable). It shows as skipped and emits a WARN on real runs.
    handled = {"finnhub_news", "rss", "apewisdom"}
    reported = {s.name for s in report.sources}
    for name, scfg in registry.items():
        if scfg.get("enabled") and name not in handled and name not in reported:
            report.sources.append(
                SourceResult(name, status="skipped", detail="enabled but not wired in Phase 1 MVP")
            )

    report.news_fetched = len(news)

    # Fuzzy dedupe against what's already banked for the window, then persist.
    existing = [] if dry_run else await store.recent_news_titles(since)
    deduped = dedupe_fuzzy(news, existing)

    if not dry_run:
        report.news_inserted = await store.insert_news_items(deduped)
        report.signals_inserted = await store.insert_signal_snapshots(signals)
        for sr in report.sources:
            sr.persisted = (
                report.news_inserted if sr.name != "apewisdom" else report.signals_inserted
            )
        await _emit_health(report)
    else:
        report.news_inserted = len(deduped)
        report.signals_inserted = len(signals)

    return report


async def _emit_health(report: CollectionReport) -> None:
    """One WARN per degraded source (golden rule 8). INFO summary on a clean run."""
    now = datetime.now(UTC)
    degraded = [s for s in report.sources if s.status in ("error", "skipped")]
    for s in degraded:
        await store.write_health_event(
            HealthEvent(
                ts=now,
                severity=HealthSeverity.WARN,
                source=f"data.news.{s.name}",
                detail=f"{s.status}: {s.detail or 'no detail'} (renormalized away this cycle)",
            )
        )
    if not degraded:
        await store.write_health_event(
            HealthEvent(
                ts=now,
                severity=HealthSeverity.INFO,
                source="data.news",
                detail=(
                    f"ingest ok: {report.news_inserted} new items, "
                    f"{report.signals_inserted} attention rows"
                ),
            )
        )
