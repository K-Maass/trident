"""Wilson CIs + cumulative precision for the accumulating relevance spot-check."""

from __future__ import annotations

import pytest

from trident.agents.sentiment.validation import (
    PrecisionReport,
    precision_from_labels,
    wilson_interval,
)


def test_wilson_perfect_score_is_not_certainty():
    """The whole reason we use Wilson: 20/20 must NOT imply 100% precision. A naive normal
    interval collapses to [1,1] here and would let a good week masquerade as proof."""
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0
    assert lo < 1.0
    assert 0.80 < lo < 0.90  # ~83.9% — honest lower bound on a 20/20 run


def test_wilson_tightens_as_labels_accumulate():
    """The point of accumulating: the same 95% score gets a tighter CI with more labels."""
    lo_small, hi_small = wilson_interval(19, 20)
    lo_big, hi_big = wilson_interval(190, 200)
    assert (hi_big - lo_big) < (hi_small - lo_small)
    assert lo_big > lo_small


def test_wilson_no_labels_is_total_ignorance_not_a_crash():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_stays_in_unit_interval():
    for k, n in [(0, 5), (5, 5), (1, 3), (99, 100)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(6, 5)


def test_precision_stratifies_on_the_MODEL_verdict():
    """Precision must be computed within the class we SAMPLED (the model's verdict). Folding on
    the human label instead would silently answer a recall question a stratified sample cannot
    estimate."""
    labels = [
        {"model_verdict": True, "label": True},  # YES, correct
        {"model_verdict": True, "label": True},  # YES, correct
        {"model_verdict": True, "label": False},  # YES, WRONG (noise reached FinBERT)
        {"model_verdict": False, "label": False},  # NO, correct
        {"model_verdict": False, "label": True},  # NO, WRONG (real signal dropped)
    ]
    rep = precision_from_labels(labels)
    assert (rep.yes_correct, rep.yes_total) == (2, 3)
    assert (rep.no_correct, rep.no_total) == (1, 2)
    assert rep.yes_precision == pytest.approx(2 / 3)
    assert rep.no_precision == pytest.approx(0.5)
    assert rep.n_labels == 5


def test_precision_report_renders_without_labels():
    rep = PrecisionReport()
    assert rep.yes_precision is None and rep.no_precision is None
    assert "no labels yet" in rep.render()
