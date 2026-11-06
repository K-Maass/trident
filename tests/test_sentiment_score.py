"""SENTINEL FinBERT scorer tests.

Offline: the pure clamp + text-builder. The actual model forward pass is an `integration` test
(needs the ~440MB ProsusAI/finbert download) that just asserts sign ordering — positive news >
neutral > negative — so a mislabeled class index would be caught.
"""

from __future__ import annotations

import pytest

from trident.agents.sentiment import score as S


def test_clamp_pre_scored_bounds():
    assert S.clamp_pre_scored(2.0) == 1.0
    assert S.clamp_pre_scored(-5.0) == -1.0
    assert S.clamp_pre_scored(0.3) == 0.3


def test_item_text_is_title_plus_clipped_body():
    it = {"title": "Headline", "body": "x" * 1000}
    text = S.item_text(it)
    assert text.startswith("Headline. ")
    # title + first 512 chars of body (+ ". " joiner)
    assert len(text) == len("Headline. ") + S.BODY_CHAR_CLIP


def test_item_text_handles_missing_body():
    assert S.item_text({"title": "Only title", "body": None}) == "Only title."


@pytest.mark.integration
def test_finbert_orders_sentiment():
    scores = S.score_texts(
        [
            "Company reports record profit and raises full-year guidance",
            "The company held its annual shareholder meeting on Tuesday",
            "Firm slashes forecast amid plunging demand and mounting losses",
        ]
    )
    pos, neutral, neg = scores
    assert pos > neutral > neg
    assert all(-1.0 <= s <= 1.0 for s in scores)
