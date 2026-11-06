"""Citation enforcement — the anti-hallucination teeth.

Hard checks (missing tag, unresolved tag) must fail the thesis. The soft classification must
distinguish a raw REPORTED figure from a DERIVED one, and — the load-bearing property — a
derived number is only evidence when it is RECOMPUTED from the metric's own {formula, inputs}
provenance. A fabricated `value` field must not be able to verify itself.
"""

from __future__ import annotations

import copy

from trident.agents.analyst.citations import check_citations, eval_formula
from trident.data import fundamentals as f

# A real derived_metrics payload. Values are chosen so each assertion has ONE explanation:
#   net_margin = 250/1000 = 0.25   roe = 250/800 = 0.3125
#   fcf        = 300-100  = 200    fcf_margin = 200/1000 = 0.20
# (net_income is 250, not 200, so a cited "200" can only be fcf — never a raw line item.)
_FIN = {
    "ticker": "TEST",
    "fetched_at": "2026-07-21T00:00:00+00:00",
    "fiscal_years": [
        {name: None for name, _, _ in f._LINE_ITEMS}
        | {
            "fy": 2025,
            "end": "2025-12-31",
            "filed": "2026-02-15",
            "form": "10-K",
            "revenue": 1000.0,
            "net_income": 250.0,
            "operating_cash_flow": 300.0,
            "capital_expenditures": 100.0,
            "stockholders_equity": 800.0,
        }
    ],
    "concepts_used": {},
    "missing_concepts": [],
}
_ROW = f.derived_metrics(_FIN)["fiscal_years"][0]
TRACE = {"T1": {"id": "T1", "tool": "get_ratios", "result": _ROW}}


def _bases(text, trace=TRACE):
    r = check_citations(text, trace)
    return r, (r.citations[0].bases if r.citations else [])


# ------------------------------------------------------------------- hard checks
def test_numeric_sentence_without_a_tag_hard_fails():
    r = check_citations("Revenue grew 12% last year. Net margin is 25% [T1].", TRACE)
    assert not r.hard_ok
    assert any("no [Tn] citation" in v for v in r.violations)


def test_unresolvable_tag_hard_fails():
    r = check_citations("Net margin is 25% [T9].", TRACE)
    assert not r.hard_ok
    assert any("[T9]" in v for v in r.violations)


def test_bare_year_and_fiscal_label_are_exempt():
    r = check_citations("In 2024 and FY2024 the business compounded well [T1].", TRACE)
    assert r.hard_ok and r.n_numeric_sentences == 0


# ------------------------------------------------------- reported vs derived
def test_reported_input_is_classified_reported():
    r, bases = _bases("Revenue was $1,000 [T1].")
    assert r.hard_ok and bases == ["reported"]


def test_derived_metric_is_verified_by_recomputation():
    """net_margin is confirmed by recomputing net_income / revenue from the raw inputs."""
    r, bases = _bases("Net margin is 25% [T1].")
    assert bases == ["derived"] and r.unverified_rate == 0.0
    assert r.counts["derived"] == 1


def test_a_fabricated_metric_value_cannot_verify_itself():
    """THE load-bearing property. The stored `value` is tampered to 0.99; recomputing
    net_income/revenue from the untouched inputs yields 0.20, so the cited 99% matches nothing
    and the tool-integrity fault is recorded separately."""
    bad = copy.deepcopy(_ROW)
    bad["metrics"]["net_margin"]["value"] = 0.99
    r, bases = _bases("Net margin is 99% [T1].", {"T1": {"id": "T1", "tool": "x", "result": bad}})
    assert bases == ["unverified"] and r.unverified_rate == 1.0
    assert r.provenance_mismatches and "net_margin" in r.provenance_mismatches[0]


def test_incomplete_provenance_is_unverifiable_not_verified():
    """An input the issuer never tagged makes the ratio uncheckable — surfaced, never counted
    as a match."""
    holed = copy.deepcopy(_ROW)
    holed["metrics"]["net_margin"]["inputs"]["revenue"] = None
    holed["metrics"]["net_margin"]["value"] = 0.25
    r, bases = _bases("Net margin is 25% [T1].", {"T1": {"id": "T1", "tool": "x", "result": holed}})
    assert bases == ["unverifiable"] and r.counts["unverifiable"] == 1


def test_a_number_found_nowhere_is_unverified():
    r, bases = _bases("Net margin is 41.3% [T1].")
    assert bases == ["unverified"] and r.unverified_rate == 1.0
    assert r.hard_ok  # soft signal only — never a hard gate


# ------------------------------------------------------------- unit discipline
def test_magnitude_suffix_matches_the_right_scale():
    # fcf = 300 - 100 = 200 (dollars in this fixture); "$200" must match, "$200B" must not.
    _, ok = _bases("Free cash flow was $200 [T1].")
    _, wrong = _bases("Free cash flow was $200B [T1].")
    assert ok == ["derived"] and wrong == ["unverified"]


def test_a_percentage_cannot_verify_against_a_dollar_magnitude():
    """Regression: a blind scale sweep let '99%' confirm itself against an unrelated $98.8B
    figure. Matching is unit-aware, so a percent only ever checks against a ratio."""
    _, bases = _bases("Equity is 800% [T1].")  # 800.0 is a reported dollar figure in the payload
    assert bases == ["unverified"]


# ------------------------------------------------------------------ formula engine
def test_eval_formula_computes_and_refuses_non_arithmetic():
    assert eval_formula("a / b", {"a": 10.0, "b": 4.0}) == 2.5
    assert eval_formula("(a - b) / b", {"a": 12.0, "b": 10.0}) == 0.2
    assert eval_formula("a / b", {"a": 1.0, "b": 0.0}) is None  # zero denominator
    assert eval_formula("a / b", {"a": 1.0, "b": None}) is None  # missing input
    # formula strings are DATA: they never get to call functions or reach attributes
    assert eval_formula("__import__('os').system('true')", {}) is None
    assert eval_formula("a.__class__", {"a": 1.0}) is None
