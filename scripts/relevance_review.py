"""The weekly SENTINEL relevance spot-check (doc 04), as an accumulating validation loop.

  --new-sample   draw a stratified 10 predicted-YES / 10 predicted-NO batch and write a review
                 sheet (no grading, no commentary — the reviewer fills in the LABEL column)
  --ingest FILE  persist a filled-in review sheet to relevance_labels
  --report       cumulative YES/NO precision with Wilson 95% CIs over ALL labels ever collected

Human labels are persisted to `relevance_labels`, NEVER to `news_items.relevant`: the model's
verdict is history and is not overwritten by a human (same never-edit-history rule as the schema).

Review sheet format (a markdown table; the reviewer edits only the LABEL and NOTE columns):
  | # | ticker | source | headline | LABEL | NOTE |
LABEL is YES/NO — what the item ACTUALLY is. The model's verdict is deliberately NOT shown, so
the reviewer is not anchored by it; it is recovered from url_hash at ingest time.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

from trident.agents.sentiment.validation import precision_from_labels
from trident.data import store

SHEET_ROW = re.compile(r"^\|\s*(\d+)\s*\|.*?\|\s*(YES|NO)\s*\|\s*(.*?)\s*\|\s*$", re.IGNORECASE)
HASH_LINE = re.compile(r"^<!--\s*(\d+)\s*=\s*([0-9a-f]+)\s*-->\s*$")


def _clean(s: str | None) -> str:
    return (s or "").replace("|", "/").replace("\n", " ").strip()


async def new_sample(sample_id: str, n_per_class: int, path: str) -> None:
    rows = await store.sample_for_review(n_per_class=n_per_class, seed=sample_id)
    if not rows:
        print("no unlabeled items available")
        return
    lines = [
        f"# SENTINEL relevance review — sample `{sample_id}`",
        "",
        f"Drawn {datetime.now(UTC):%Y-%m-%d}. Stratified {n_per_class} predicted-YES + "
        f"{n_per_class} predicted-NO, from items the model actually judged.",
        "",
        "**Question:** is the item *materially about that ticker's business or stock*? Fill in",
        "**LABEL** (YES/NO) and an optional NOTE. The model's verdict is hidden on purpose —",
        "it is recovered at ingest, so you are not anchored by it.",
        "",
        "| # | ticker | source | headline | LABEL | NOTE |",
        "|---|---|---|---|---|---|",
    ]
    hashes = []
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['ticker']} | {r['source']} | {_clean(r['title'])} |  |  |")
        hashes.append(f"<!-- {i} = {r['url_hash']} -->")
    lines += ["", "<!-- do not edit below: row -> url_hash map -->", *hashes, ""]
    await asyncio.to_thread(Path(path).write_text, "\n".join(lines))
    print(f"wrote {len(rows)} rows -> {path}")


async def ingest(path: str, reviewer: str, sample_id: str) -> None:
    text_in = await asyncio.to_thread(Path(path).read_text)

    idx_to_hash = {
        int(m.group(1)): m.group(2) for m in map(HASH_LINE.match, text_in.splitlines()) if m
    }
    verdicts: dict[int, tuple[bool, str]] = {}
    for line in text_in.splitlines():
        m = SHEET_ROW.match(line)
        if m:
            verdicts[int(m.group(1))] = (m.group(2).upper() == "YES", m.group(3))

    if not verdicts:
        print("no filled-in LABEL cells found — nothing to ingest")
        return

    model = {
        r["url_hash"]: r["relevant"]
        for r in await store.get_news(tickers=None, since=None, until=None)
        if r["url_hash"] in set(idx_to_hash.values())
    }

    rows = []
    for i, (label, note) in sorted(verdicts.items()):
        h = idx_to_hash.get(i)
        if h is None or model.get(h) is None:
            print(f"  row {i}: no url_hash / model verdict — skipped")
            continue
        rows.append(
            {
                "url_hash": h,
                "reviewer": reviewer,
                "label": label,
                "model_verdict": bool(model[h]),
                "note": note or None,
                "sample_id": sample_id,
            }
        )
    n = await store.insert_relevance_labels(rows)
    agree = sum(1 for r in rows if r["label"] == r["model_verdict"])
    print(
        f"ingested {n} labels from {reviewer} (sample={sample_id}) — "
        f"agreed with model on {agree}/{n}"
    )


async def report() -> None:
    labels = await store.get_relevance_labels()
    rep = precision_from_labels(labels)
    print(rep.render())
    if rep.n_labels == 0:
        print("\n(no human labels yet — run --new-sample, review it, then --ingest)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-sample", action="store_true")
    ap.add_argument("--ingest", metavar="FILE")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--reviewer", help="Reviewer label required for ingest")
    ap.add_argument("--sample-id", default=f"weekly-{datetime.now(UTC):%Y-%m-%d}")
    ap.add_argument("--n", type=int, default=10, help="per class")
    ap.add_argument("--out", default="artifacts/relevance_review.md")
    args = ap.parse_args()

    if args.new_sample:
        await new_sample(args.sample_id, args.n, args.out)
    if args.ingest:
        if not args.reviewer:
            ap.error("ingest requires --reviewer")
        await ingest(args.ingest, args.reviewer, args.sample_id)
    if args.report or not (args.new_sample or args.ingest):
        await report()


if __name__ == "__main__":
    asyncio.run(main())
