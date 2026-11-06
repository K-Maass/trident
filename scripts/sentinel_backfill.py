"""Phase 4 SENTINEL backfill (riders a + c).

Runs the scoring pipeline over ALL banked news so a sentiment-history baseline exists
immediately, then reports threshold REACHABILITY (not a tuning step) and source coverage:

  1. Haiku relevance filter over every unjudged item (cached in news_items.relevant → $0 on
     re-runs).
  2. FinBERT sentiment over every relevant-unscored item (cached in news_items.sentiment).
  3. Aggregate the full banked session range and persist the sentiment_aggregate signals.
  4. Print the sent_z / attention_z distributions and whether the doc-04 cutoffs (1.5 / 1.0) are
     REACHABLE on the banked window — a Phase-3-style "is the threshold even hittable" check.
     We do NOT tune the cutoffs to this data.
  5. Report ApeWisdom coverage and re-probe Tradestie's live + ?date= endpoints.
  6. --verdict-sample N prints a balanced YES/NO sample of relevance verdicts for human review.

Usage:
  uv run python scripts/sentinel_backfill.py                    # full backfill + report
  uv run python scripts/sentinel_backfill.py --dry-run          # score in memory, persist nothing
  uv run python scripts/sentinel_backfill.py --verdict-sample 20
"""

import argparse
import asyncio
from datetime import UTC, datetime

import httpx
import pandas as pd
from sqlalchemy import text

from trident.agents.sentiment.aggregate import aggregate
from trident.agents.sentiment.relevance import run_relevance_backfill
from trident.agents.sentiment.score import run_score_backfill
from trident.config import load_sentinel_sources, load_universe
from trident.data import calendar as trading_calendar
from trident.data import store
from trident.db import get_engine


async def _news_date_bounds() -> tuple[datetime, datetime] | None:
    async with get_engine().connect() as conn:
        row = (
            await conn.execute(text("SELECT min(published_at), max(published_at) FROM news_items"))
        ).first()
    if not row or row[0] is None:
        return None
    return row[0], row[1]


async def _apewisdom_coverage() -> str:
    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT count(*), count(distinct ticker), min(ts), max(ts) FROM signals "
                    "WHERE agent_id='sentinel' AND metadata->>'kind'='attention_snapshot'"
                )
            )
        ).first()
    return f"rows={row[0]} tickers={row[1]} span={row[2]}..{row[3]}"


async def _reprobe_tradestie() -> str:
    reg = load_sentinel_sources().get("tradestie_wsb", {})
    url = reg.get("base_url", "")
    out = []
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as c:
        for label, params in (("live", None), ("?date=07-01-2026", {"date": "07-01-2026"})):
            try:
                r = await c.get(url, params=params)
                n = len(r.json()) if r.status_code == 200 else 0
                out.append(f"{label}: reachable status={r.status_code} rows={n}")
            except Exception as e:  # noqa: BLE001
                out.append(f"{label}: UNREACHABLE {type(e).__name__}")
    return " | ".join(out)


def _dist(name: str, s: pd.Series, cutoff: float) -> str:
    s = s.dropna()
    if s.empty:
        return f"{name}: no finite values (baseline too immature to compute z)"
    qs = s.quantile([0.5, 0.9, 0.95, 0.99]).round(3).to_dict()
    n_above = int((s > cutoff).sum())
    reach = "REACHABLE" if s.max() > cutoff else "NOT reachable"
    return (
        f"{name}: n={len(s)} max={s.max():.3f} p50={qs[0.5]} p90={qs[0.9]} "
        f"p95={qs[0.95]} p99={qs[0.99]} | >{cutoff}: {n_above} rows → cutoff {reach}"
    )


async def _print_verdict_sample(n: int) -> None:
    half = max(1, n // 2)
    async with get_engine().connect() as conn:
        yes = (
            await conn.execute(
                text(
                    "SELECT ticker, source, title FROM news_items WHERE relevant IS TRUE "
                    "ORDER BY random() LIMIT :n"
                ),
                {"n": half},
            )
        ).all()
        no = (
            await conn.execute(
                text(
                    "SELECT ticker, source, title FROM news_items WHERE relevant IS FALSE "
                    "AND ticker IS NOT NULL ORDER BY random() LIMIT :n"
                ),
                {"n": n - half},
            )
        ).all()
    print(f"\n=== relevance verdict sample ({len(yes) + len(no)}) — for human review ===")
    for verdict, rows in (("YES", yes), ("NO ", no)):
        for r in rows:
            print(f"  [{verdict}] {r.ticker:5s} {r.source:16s} {r.title[:74]}")


async def main(args: argparse.Namespace) -> None:
    universe = load_universe()
    print("=== 1. relevance (Haiku, cached) ===")
    rel = await run_relevance_backfill(dry_run=args.dry_run)
    print(
        f"judged={rel.judged} relevant={rel.relevant} irrelevant={rel.irrelevant} "
        f"api_calls={rel.api_calls} tickerless_auto_no={rel.tickerless_auto_no} "
        f"parse_misses={rel.parse_misses} errored_items={rel.errored_items}"
    )

    print("=== 2. FinBERT sentiment (cached) ===")
    sc = await run_score_backfill(dry_run=args.dry_run)
    print(f"scored={sc.scored} mean={sc.mean:.3f} min={sc.min:.3f} max={sc.max:.3f}")

    print("=== 3. aggregate full banked range ===")
    bounds = await _news_date_bounds()
    if bounds is None:
        print("no news banked — nothing to aggregate")
        await store.dispose_engine()
        return
    start = bounds[0].date()
    end = min(bounds[1].date(), trading_calendar.previous_session(datetime.now(UTC).date()))
    panel = await aggregate(universe, start, end, persist=not args.dry_run)
    sessions = sorted(panel["date"].unique()) if not panel.empty else []
    print(
        f"range {start}..{end}  sessions={len(sessions)}  ticker-session rows={len(panel)}  "
        f"persisted={'no (dry-run)' if args.dry_run else 'yes'}"
    )

    print("=== 4. threshold REACHABILITY (rider c — no tuning) ===")
    if panel.empty:
        print("empty panel")
    else:
        print("  " + _dist("sent_z (enter>1.5)", panel["sent_z"], 1.5))
        print("  " + _dist("attention_z (enter>1.0)", panel["attention_z"], 1.0))
        immature = int(panel["immature"].sum())
        print(
            f"  NOTE: {immature}/{len(panel)} rows immature (<21-session baseline). With ~"
            f"{len(sessions)} banked sessions this is EXPECTED; z-scores firm up as history grows."
        )

    print("=== 5. source coverage ===")
    print("  apewisdom banked:", await _apewisdom_coverage())
    print("  tradestie reprobe:", await _reprobe_tradestie())

    if args.verdict_sample and not args.dry_run:
        await _print_verdict_sample(args.verdict_sample)

    await store.dispose_engine()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="score in memory, persist nothing")
    p.add_argument("--verdict-sample", type=int, default=0, help="print N verdicts for review")
    asyncio.run(main(p.parse_args()))
