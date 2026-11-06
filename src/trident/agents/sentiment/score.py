"""SENTINEL sentiment scorer (doc 04, pipeline stage 3) — FinBERT, local, $0.

`ProsusAI/finbert` over `title + first 512 chars of body`, on CPU, batch 32, producing
`p(pos) - p(neg) ∈ [-1, 1]` per item. Only relevant, unscored items are scored (the
`news_items.sentiment` column is the cache), so a re-run makes no forward passes for items
already scored. pre_scored aggregator sources (Tradestie) don't go through FinBERT — they use
`clamp_pre_scored` on their provider score; that path is ready for when Tradestie returns.
"""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass
from datetime import datetime

from trident.data import store

MODEL_ID = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_TOKENS = 512
BODY_CHAR_CLIP = 512  # "title + first 512 chars" (doc 04); tokenizer still truncates to 512 tok


def clamp_pre_scored(x: float) -> float:
    """Clamp a provider-supplied score to [-1, 1] (doc 04 pre_scored rule)."""
    return max(-1.0, min(1.0, float(x)))


def item_text(it: dict) -> str:
    """title + first 512 chars of body — the FinBERT input (doc 04)."""
    title = (it.get("title") or "").strip()
    body = (it.get("body") or "").strip()[:BODY_CHAR_CLIP]
    return f"{title}. {body}".strip()


@functools.lru_cache(maxsize=1)
def _load_model():
    """Lazy, cached (holds the ~440MB model). Import torch/transformers only when scoring so
    the rest of SENTINEL runs without the heavy extra installed."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    pos_idx = next(i for i, lab in id2label.items() if lab.startswith("pos"))
    neg_idx = next(i for i, lab in id2label.items() if lab.startswith("neg"))
    return tok, model, pos_idx, neg_idx


def score_texts(texts: list[str], batch_size: int = BATCH_SIZE) -> list[float]:
    """FinBERT p(pos) - p(neg) per text, clamped to [-1, 1]. Synchronous / CPU-bound."""
    if not texts:
        return []
    import torch

    tok, model, pos_idx, neg_idx = _load_model()
    out: list[float] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, padding=True)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        diffs = (probs[:, pos_idx] - probs[:, neg_idx]).tolist()
        out.extend(clamp_pre_scored(d) for d in diffs)
    return out


@dataclass
class ScoreReport:
    scored: int = 0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0


async def run_score_backfill(
    since: datetime | None = None, limit: int | None = None, dry_run: bool = False
) -> ScoreReport:
    """FinBERT-score all relevant-but-unscored items → write news_items.sentiment. Idempotent:
    a re-run finds nothing (sentiment already set) and does no forward passes."""
    items = await store.news_needing_sentiment(since=since, limit=limit)
    if not items:
        return ScoreReport()
    texts = [item_text(it) for it in items]
    scores = await asyncio.to_thread(score_texts, texts)
    mapping = {it["url_hash"]: s for it, s in zip(items, scores, strict=True)}
    if not dry_run:
        await store.set_news_sentiment(mapping)
    vals = list(mapping.values())
    return ScoreReport(
        scored=len(vals),
        mean=sum(vals) / len(vals),
        min=min(vals),
        max=max(vals),
    )
