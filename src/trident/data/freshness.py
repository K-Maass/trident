"""Data freshness health job (doc 02) — the 07:30 ET canary.

Asserts staleness bounds per data source and writes health_events. A stale/dead source
degrades to a WARN (downstream renormalizes it away), never a crash (golden rule 8).
Wired into the scheduler in Phase 6; callable now for ad-hoc checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from trident.config import load_universe
from trident.data import calendar as trading_calendar
from trident.data import fundamentals as fund
from trident.data import market, store
from trident.models import HealthEvent, HealthSeverity


@dataclass
class FreshnessResult:
    source: str
    ok: bool
    detail: str


@dataclass
class FreshnessReport:
    asof: datetime
    results: list[FreshnessResult] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return all(r.ok for r in self.results)


# ---------------------------------------------------------------- market bars
def check_market_bars(
    tickers: list[str] | None = None,
    cache_dir: Path = market.BARS_DIR,
    asof: datetime | None = None,
    max_stale_sessions: int = 2,
) -> FreshnessResult:
    """Each ticker's cached bars should reach the last completed session (± a small gap)."""
    tickers = [t.upper() for t in (tickers or load_universe())]
    last = trading_calendar.last_completed_session(asof)
    stale: list[str] = []
    for t in tickers:
        path = market._cache_path(t, cache_dir)
        if not path.exists():
            stale.append(f"{t}:no-cache")
            continue
        idx = pd.read_parquet(path, columns=[]).index
        if len(idx) == 0:
            stale.append(f"{t}:empty")
            continue
        max_date = idx.max().date()
        gap = sum(1 for d in trading_calendar.sessions(max_date, last) if d > max_date)
        if gap > max_stale_sessions:
            stale.append(f"{t}@{max_date}")
    ok = not stale
    detail = f"fresh through {last}" if ok else f"last={last}; stale({len(stale)}): {stale[:5]}"
    return FreshnessResult("market_bars", ok, detail)


# --------------------------------------------------------------- fundamentals
def check_fundamentals(
    tickers: list[str] | None = None,
    cache_dir: Path = fund.FUND_DIR,
    max_stale_days: int = 120,
) -> FreshnessResult:
    """Fundamentals update ~quarterly; flag missing or very old cached companyfacts."""
    tickers = [t.upper() for t in (tickers or load_universe())]
    now = datetime.now(UTC)
    problems: list[str] = []
    for t in tickers:
        path = cache_dir / f"{t}.json"
        if not path.exists():
            problems.append(f"{t}:missing")
            continue
        try:
            import json

            fetched = json.loads(path.read_text()).get("_fetched_at")
            if fetched and (now - datetime.fromisoformat(fetched)).days > max_stale_days:
                problems.append(f"{t}:stale")
        except Exception:  # noqa: BLE001 — unreadable cache counts as a problem, not a crash
            problems.append(f"{t}:unreadable")
    ok = not problems
    detail = (
        f"{len(tickers)} tickers cached & fresh"
        if ok
        else f"issues({len(problems)}): {problems[:5]}"
    )
    return FreshnessResult("fundamentals", ok, detail)


# ----------------------------------------------------------------------- news
async def check_news(min_items_24h: int = 1) -> FreshnessResult:
    """Ingestion is alive iff some news landed in the last 24h (doc 02/04)."""
    engine = store.get_engine()
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT count(*) FILTER (WHERE fetched_at > now() - interval '24 hours'), "
                    "max(published_at) FROM news_items"
                )
            )
        ).one()
    last24, newest = int(row[0]), row[1]
    ok = last24 >= min_items_24h
    detail = f"{last24} items in last 24h; newest={newest}"
    return FreshnessResult("news", ok, detail)


# ------------------------------------------------------------------ orchestrator
async def run_freshness_check(
    asof: datetime | None = None, write: bool = True, tickers: list[str] | None = None
) -> FreshnessReport:
    """Run all checks; write one WARN health_event per stale source (INFO if all fresh)."""
    asof = asof or datetime.now(UTC)
    report = FreshnessReport(asof=asof)
    report.results.append(check_market_bars(tickers=tickers, asof=asof))
    report.results.append(check_fundamentals(tickers=tickers))
    try:
        report.results.append(await check_news())
    except Exception as e:  # noqa: BLE001 — DB down shouldn't crash the health job
        report.results.append(FreshnessResult("news", False, f"check failed: {type(e).__name__}"))

    if write:
        stale = [r for r in report.results if not r.ok]
        for r in stale:
            await store.write_health_event(
                HealthEvent(
                    ts=asof,
                    severity=HealthSeverity.WARN,
                    source=f"freshness.{r.source}",
                    detail=r.detail,
                )
            )
        if not stale:
            await store.write_health_event(
                HealthEvent(
                    ts=asof,
                    severity=HealthSeverity.INFO,
                    source="freshness",
                    detail="; ".join(f"{r.source} ok" for r in report.results),
                )
            )
    return report
