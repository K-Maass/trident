"""SENTINEL ingestion + scoring cycle.

  uv run python scripts/collect_news.py               # ingest, then score+aggregate new items
  uv run python scripts/collect_news.py --no-score    # ingest only (Phase-1 behaviour)
  uv run python scripts/collect_news.py --dry-run      # fetch+parse+dedupe, no DB writes, no score
  uv run python scripts/collect_news.py --tickers AAPL,MSFT,NVDA

A real run banks raw news into news_items and ApeWisdom attention into signals, then (unless
--no-score) runs the SENTINEL scoring pipeline over any NEW items — Haiku relevance filter
(cached), FinBERT sentiment (cached), and per-ticker aggregation persisted to signals — so the
sentiment-history baseline keeps maturing daily. This scores DATA only; the trading agent is not
run here. Wired into the 2×/day launchd collector (doc 04; Phase-6 scheduler later).
"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from trident.data import store
from trident.data.news import collect
from trident.models import HealthEvent, HealthSeverity


async def _score_and_aggregate() -> None:
    """Post-ingest: judge+score new items and (re)persist the trailing sentiment aggregates.
    Guarded — ingestion already succeeded, so a scoring failure degrades to a WARN, never a
    crash (golden rule 8). The imports are INSIDE the try on purpose: an import error is a
    scoring failure too, and must not take down a good ingest (it did, 2026-07-18)."""
    try:
        from trident.agents.sentiment.aggregate import PANEL_LOOKBACK_DAYS, aggregate
        from trident.config import load_universe
        from trident.data import calendar as trading_calendar
        from trident.orchestration.sentinel import ensure_scored

        rel, sc = await ensure_scored()
        universe = load_universe()
        target = trading_calendar.previous_session(datetime.now(UTC).date())
        await aggregate(
            universe, target - timedelta(days=PANEL_LOOKBACK_DAYS), target, persist=True
        )
        print(
            f"score: relevance judged={rel.judged} (api_calls={rel.api_calls}, "
            f"relevant={rel.relevant}), finbert scored={sc.scored}; "
            f"aggregates persisted through session {target}"
        )
    except Exception as e:  # noqa: BLE001 — never break a good ingest
        await store.write_health_event(
            HealthEvent(
                ts=datetime.now(UTC),
                severity=HealthSeverity.WARN,
                source="data.news.scoring",
                detail=f"post-ingest scoring degraded: {type(e).__name__}: {e}",
            )
        )
        print(f"WARN: post-ingest scoring degraded ({type(e).__name__}: {e})")


async def _run(args: argparse.Namespace) -> None:
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None
    report = await collect(tickers=tickers, dry_run=args.dry_run)
    print(f"asof={report.asof.isoformat()}  dry_run={args.dry_run}")
    for s in report.sources:
        line = f"  {s.name:14s} status={s.status:8s} fetched={s.fetched:4d}"
        if s.detail:
            line += f"  [{s.detail}]"
        print(line)
    print(
        f"news: fetched={report.news_fetched}  inserted(new)={report.news_inserted}  "
        f"attention_signals={report.signals_inserted}"
    )
    if not args.dry_run and not args.no_score:
        await _score_and_aggregate()
    await store.dispose_engine()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="no DB writes; implies no scoring")
    p.add_argument("--no-score", action="store_true", help="ingest only, skip SENTINEL scoring")
    p.add_argument("--tickers", default="", help="comma-separated subset (default: full universe)")
    asyncio.run(_run(p.parse_args()))
