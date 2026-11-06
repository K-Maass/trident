"""Recover the universe-wide Finnhub history gap (2026-06-22 → 07-01).

WHY THIS EXISTS
---------------
The 2x/day collector fetched Finnhub `/company-news` with a wide date range, and the endpoint
returns only the ~250 MOST RECENT items per request. For a high-volume name that quota is
consumed entirely by the last few days, so nothing older than ~2026-07-02 was ever banked —
for ANY of the 40 tickers. Everything in the DB before that date is RSS-only.

That is not just a coverage hole: `sent_raw` is a source-weighted mean, so every ticker's
baseline had a SOURCE-MIX REGIME BREAK at Jul 2 (RSS-only before, Finnhub+RSS after), which
contaminates its own trailing sent_z baseline.

The endpoint DOES serve history when the window is narrow enough that the quota isn't the
binding constraint. So we re-fetch PER DAY, which gives each day its own quota.

SAFETY
------
Idempotent by construction: `news_items.url_hash` is the PK and the insert is
`ON CONFLICT DO NOTHING`, so re-pulled items that overlap existing rows are skipped and their
existing relevance/sentiment verdicts are preserved (never reset, never double-counted).

Read-only w.r.t. decisions: no agent has ever traded on these rows.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, date, datetime, timedelta

import httpx

from trident.config import get_settings, load_sentinel_sources, load_universe
from trident.data import store
from trident.data.news import AsyncRateLimiter, parse_finnhub
from trident.models import HealthEvent, HealthSeverity

BASE = "https://finnhub.io/api/v1/company-news"
GAP_START = date(2026, 6, 22)
GAP_END = date(2026, 7, 1)  # inclusive; Jul 2 onward is already banked


def _days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


async def backfill(
    tickers: list[str], start: date, end: date, dry_run: bool = False
) -> tuple[int, int, list[str]]:
    """Per-day Finnhub fetch over [start, end]. Returns (fetched, inserted, errors)."""
    settings = get_settings()
    settings.require("finnhub_api_key")
    reg = load_sentinel_sources()
    per_min = float(reg.get("finnhub_news", {}).get("rate_limit_per_min", 55))
    lim = AsyncRateLimiter(per_min)

    days = _days(start, end)
    fetched = inserted = 0
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for t in tickers:
            t_items = []
            for d in days:
                await lim.acquire()
                iso = d.isoformat()
                try:
                    r = await client.get(
                        BASE,
                        params={
                            "symbol": t,
                            "from": iso,
                            "to": iso,
                            "token": settings.finnhub_api_key,
                        },
                    )
                    r.raise_for_status()
                    t_items.extend(parse_finnhub(r.json(), t))
                except Exception as e:  # noqa: BLE001 — per-(ticker,day) resilience
                    errors.append(f"{t}@{iso}: {type(e).__name__}")
            fetched += len(t_items)
            n = 0 if dry_run else await store.insert_news_items(t_items)
            inserted += n
            print(f"  {t:6s} fetched={len(t_items):5d} inserted_new={n:5d}")

    if errors:
        await store.write_health_event(
            HealthEvent(
                ts=datetime.now(UTC),
                severity=HealthSeverity.WARN,
                source="finnhub.backfill",
                detail=f"{len(errors)} per-day fetch error(s); first={errors[0][:160]}",
            )
        )
    return fetched, inserted, errors


async def probe_beyond(tickers: list[str], start: date, end: date) -> None:
    """Rider (d): how far back does the free tier actually serve? Probe only — never ingest.
    The panel start stays at 2026-06-22."""
    settings = get_settings()
    lim = AsyncRateLimiter(55)
    print(f"\n=== probe BEYOND the gap ({start}..{end}) — measurement only, NOT ingested ===")
    async with httpx.AsyncClient(timeout=30.0) as client:
        for t in tickers:
            total = 0
            for d in _days(start, end):
                await lim.acquire()
                try:
                    r = await client.get(
                        BASE,
                        params={
                            "symbol": t,
                            "from": d.isoformat(),
                            "to": d.isoformat(),
                            "token": settings.finnhub_api_key,
                        },
                    )
                    total += len(r.json()) if r.status_code == 200 else 0
                except Exception:  # noqa: BLE001
                    pass
            print(f"  {t:6s} items available {start}..{end}: {total}")


async def measure_collisions(tickers: list[str], start: date, end: date) -> None:
    """Rider (1a): quantify cross-ticker URL collisions WITHOUT inserting anything.

    ON CONFLICT DO NOTHING already discarded the historical (url, ticker) pairings, so the only
    way to recover them is to re-fetch and compare each fetched item's ticker against the ticker
    the URL is actually stored under. Reports:
      * collision share  — of fetched items whose URL is already stored, how many are stored
        under a DIFFERENT ticker (the attribution-bias magnitude).
      * fetch-order (in)stability — a URL seen here under >1 ticker is an item whose stored
        attribution is decided purely by ingest order; we count how many distinct URLs are
        multi-ticker within this re-fetch.
    """
    settings = get_settings()
    settings.require("finnhub_api_key")
    lim = AsyncRateLimiter(55)

    # url_hash -> set of tickers it was fetched under (in THIS re-fetch)
    fetched_under: dict[str, set[str]] = {}
    fetched_items = 0
    print(f"\n=== collision measurement {start}..{end} — {len(tickers)} tickers, NO inserts ===")
    async with httpx.AsyncClient(timeout=30.0) as client:
        for t in tickers:
            for d in _days(start, end):
                await lim.acquire()
                try:
                    r = await client.get(
                        BASE,
                        params={
                            "symbol": t,
                            "from": d.isoformat(),
                            "to": d.isoformat(),
                            "token": settings.finnhub_api_key,
                        },
                    )
                    items = parse_finnhub(r.json(), t) if r.status_code == 200 else []
                except Exception:  # noqa: BLE001
                    items = []
                for it in items:
                    fetched_items += 1
                    fetched_under.setdefault(it.url_hash, set()).add(t)

    stored = await store.ticker_for_hashes(list(fetched_under))
    # cross-ticker vs the STORED attribution
    stored_present = cross_ticker = 0
    for h, tick_set in fetched_under.items():
        st = stored.get(h)
        if st is None:
            continue
        stored_present += 1
        if any(tk != st for tk in tick_set):
            cross_ticker += 1
    # multi-ticker WITHIN this re-fetch = order-decided attribution
    multi_ticker_urls = sum(1 for s in fetched_under.values() if len(s) > 1)

    uniq = len(fetched_under)
    print(f"  fetched items ............ {fetched_items}")
    print(f"  distinct URLs ............ {uniq}")
    print(f"  URLs already stored ...... {stored_present}")
    pct = (100.0 * cross_ticker / stored_present) if stored_present else 0.0
    print(
        f"  stored under a DIFFERENT ticker than a re-fetch attributed it to: "
        f"{cross_ticker} ({pct:.1f}% of stored)"
    )
    mpct = (100.0 * multi_ticker_urls / uniq) if uniq else 0.0
    print(
        f"  URLs fetched under >1 ticker (order-decided attribution): "
        f"{multi_ticker_urls} ({mpct:.1f}% of distinct)"
    )
    logged = await store.collision_stats()
    print(f"  prospectively logged so far (news_url_collisions): {logged}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=GAP_START.isoformat())
    ap.add_argument("--end", default=GAP_END.isoformat())
    ap.add_argument("--tickers", default="", help="comma-separated; default = full universe")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe-beyond", action="store_true", help="rider (d): measure, don't ingest")
    ap.add_argument(
        "--measure-collisions",
        action="store_true",
        help="rider (1a): dry re-fetch, compare fetched vs stored ticker, NO inserts",
    )
    args = ap.parse_args()

    tickers = (
        [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if args.tickers
        else load_universe()
    )
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # Measurement mode is exclusive — it must NEVER ingest (it is the "measure, don't act" path).
    if args.measure_collisions:
        await measure_collisions(tickers, start, end)
        return

    print(f"=== Finnhub per-day backfill {start}..{end} — {len(tickers)} tickers ===")
    fetched, inserted, errors = await backfill(tickers, start, end, dry_run=args.dry_run)
    print(f"\nfetched={fetched} inserted_new={inserted} dupes_skipped={fetched - inserted}")
    if errors:
        print(f"errors={len(errors)} (first 5): {errors[:5]}")

    if args.probe_beyond:
        await probe_beyond(tickers[:3], start - timedelta(days=7), start - timedelta(days=1))


if __name__ == "__main__":
    asyncio.run(main())
