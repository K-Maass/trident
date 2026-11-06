"""SENTINEL relevance filter (doc 04, pipeline stage 2) — the load-bearing noise gate.

Batches raw_text items 20/call to `bulk_model` (Haiku, temperature 0) and asks, per item,
whether it is MATERIALLY about the given ticker's business/stock. Kills ticker-symbol false
positives ("A", "ALL") and meme noise before FinBERT ever runs.

Caching is the `news_items.relevant` column itself: we only ever ask about items whose verdict
is NULL, so a re-run over already-judged items issues ZERO API calls — the "$0 LLM on cache
hits" acceptance. Ticker-less market-wide items (CNBC/MarketWatch/Seeking-Alpha) can't attach to
a ticker for per-ticker aggregation, so they are auto-NO with no API call.

Failure policy (golden rule 8): a whole-batch API error leaves those items UNjudged (relevant
stays NULL → retried next cycle), never silently NO. A parse miss inside a good response is a
conservative NO (we got an answer, the model just didn't format that line) and is counted +
WARNed for the weekly spot-check.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from trident.config import get_settings
from trident.data import store
from trident.models import HealthEvent, HealthSeverity

BATCH_SIZE = 20
MAX_TOKENS = 1024
_TITLE_CLIP = 240
MAX_RETRIES = 6  # SDK-level exponential backoff on 429 / connection errors
CLIENT_TIMEOUT = 60.0
THROTTLE_SECONDS = 0.35  # gentle inter-batch pacing so a long backfill doesn't exhaust the pool

_SYSTEM = (
    "You are a financial-news relevance filter. Each numbered item gives a stock TICKER and a "
    "headline. Decide whether the item is MATERIALLY about that ticker's business or stock — "
    "NOT a passing mention, a broad market/index round-up, or a ticker-symbol coincidence "
    "(e.g. the letter 'A' or the word 'ALL'). Reply with EXACTLY one line per item, same order, "
    "in the form: '<n>. YES - <short reason>' or '<n>. NO - <short reason>'. Nothing else."
)

_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(YES|NO)\b", re.IGNORECASE)


@dataclass
class RelevanceReport:
    judged: int = 0  # verdicts we are writing back (excludes API-errored batches)
    relevant: int = 0
    irrelevant: int = 0
    api_calls: int = 0
    tickerless_auto_no: int = 0
    parse_misses: int = 0
    errored_items: int = 0  # left NULL for retry
    errors: list[str] = field(default_factory=list)


def _build_user_prompt(batch: list[dict]) -> str:
    lines = []
    for i, it in enumerate(batch, 1):
        title = (it.get("title") or "").replace("\n", " ").strip()[:_TITLE_CLIP]
        lines.append(f'{i}. [{it["ticker"]}] "{title}"')
    return "\n".join(lines)


def _parse_verdicts(text_out: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for line in text_out.splitlines():
        m = _LINE.match(line)
        if m:
            out[int(m.group(1))] = m.group(2).upper() == "YES"
    return out


async def _judge_batch(client, model: str, batch: list[dict]) -> dict[str, bool]:
    """Return {url_hash: verdict} for a good response; parse misses become conservative NO.
    Raises on API error so the caller can leave the batch unjudged for retry."""
    resp = await client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _build_user_prompt(batch)}],
    )
    text_out = "".join(
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
    )
    parsed = _parse_verdicts(text_out)
    verdicts: dict[str, bool] = {}
    misses = 0
    for i, it in enumerate(batch, 1):
        if i in parsed:
            verdicts[it["url_hash"]] = parsed[i]
        else:
            verdicts[it["url_hash"]] = False  # conservative NO on a parse miss
            misses += 1
    verdicts["__misses__"] = misses  # smuggled out; stripped by the caller
    return verdicts


async def filter_relevance(
    items: list[dict],
    *,
    client=None,
    model: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, bool], RelevanceReport]:
    """Judge raw_text items for per-ticker materiality. Returns ({url_hash: bool}, report).

    Pass items straight from `store.news_needing_relevance()` — everything here is assumed to
    need a verdict, so the caller controls the cache boundary and $0-on-cache follows for free.
    """
    settings = get_settings()
    model = model or settings.bulk_model
    report = RelevanceReport()
    verdicts: dict[str, bool] = {}

    tickered: list[dict] = []
    for it in items:
        if not it.get("ticker"):
            verdicts[it["url_hash"]] = False
            report.tickerless_auto_no += 1
        else:
            tickered.append(it)

    if tickered:
        throttle = False
        if client is None:
            settings.require("anthropic_api_key")
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                max_retries=MAX_RETRIES,
                timeout=CLIENT_TIMEOUT,
            )
            throttle = True  # real client → pace batches; a test's fake client runs full speed
        for start in range(0, len(tickered), batch_size):
            batch = tickered[start : start + batch_size]
            try:
                raw = await _judge_batch(client, model, batch)
                report.api_calls += 1
                report.parse_misses += raw.pop("__misses__", 0)
                verdicts.update(raw)
            except Exception as e:  # noqa: BLE001 — leave the batch unjudged (retry next cycle)
                report.errored_items += len(batch)
                report.errors.append(f"batch@{start}: {type(e).__name__}: {e}")
            if throttle and start + batch_size < len(tickered):
                await asyncio.sleep(THROTTLE_SECONDS)

    report.judged = len(verdicts)
    report.relevant = sum(1 for v in verdicts.values() if v)
    report.irrelevant = report.judged - report.relevant
    if report.errors:
        await store.write_health_event(
            HealthEvent(
                ts=datetime.now(UTC),
                severity=HealthSeverity.WARN,
                source="sentinel.relevance",
                detail=(
                    f"{len(report.errors)} batch error(s), {report.errored_items} items left "
                    f"unjudged for retry. first={report.errors[0][:160]}"
                ),
            )
        )
    return verdicts, report


async def run_relevance_backfill(
    since: datetime | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    client=None,
) -> RelevanceReport:
    """Fetch unjudged items → Haiku → write verdicts back (unless dry_run). Idempotent: a
    re-run finds nothing NULL and makes zero API calls."""
    items = await store.news_needing_relevance(since=since, limit=limit)
    verdicts, report = await filter_relevance(items, client=client)
    if not dry_run and verdicts:
        await store.set_news_relevance(verdicts)
    return report
