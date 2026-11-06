"""Citation enforcement (doc 05) — the anti-hallucination teeth.

The thesis prompt requires an inline `[T3]`-style tag on every numeric claim, where `T3` names a
tool call recorded in THIS ticker's trace. This module checks that discipline.

HARD (fails the thesis): every sentence containing a number carries >=1 tag (bare four-digit
years and FY labels exempt), and every tag resolves to a real recorded call. A hard failure gets
one repair round-trip; still failing -> the ticker is dropped (no thesis -> no position).

SOFT (surfaced, never a gate): every cited number is classified against the cited payload.
Because `data/fundamentals.py` ships each ratio as {"value", "formula", "inputs"} provenance, we
do not have to take the tool's own summary on faith:

  reported     — the number is a RAW figure the filing or market reported (a line item, a price,
                 or one of a metric's `inputs`).
  derived      — the number matches a metric RECOMPUTED here from its formula and inputs. The
                 tool's own `value` field is deliberately NOT trusted as evidence; a derived
                 claim is only verified if the raw inputs arithmetically imply it.
  unverifiable — the metric's provenance is incomplete (some input is None, as happens when an
                 issuer never tagged the concept), so no arithmetic check exists. The model may
                 have quoted the tool faithfully; we simply cannot confirm it.
  unverified   — matched nothing. This is the hallucination signal.

`provenance_mismatches` records any metric whose stored `value` disagrees with the recomputation
— that is a TOOL-integrity fault, not a model fault, and it must never be silent.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

from trident.models import Citation

_TAG = re.compile(r"\[T(\d+)\]")
# A numeric token, NOT preceded by a letter/digit (so the digits inside FY2024 / Q3 are not
# "numbers"), with its UNIT captured: percent, multiple, or a magnitude word/letter.
# The word-boundary belongs INSIDE the alternation, on the alphabetic suffixes only: "%" and
# the following space are both non-word characters, so a trailing \b would never match after a
# percent sign and every percentage would silently degrade to its bare reading.
_NUM = re.compile(
    r"(?<![A-Za-z0-9])([-+]?\$?\d[\d,]*(?:\.\d+)?)"
    r"\s*(%|x\b|bn\b|billion\b|million\b|thousand\b|trillion\b|[bmkt]\b)?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"^(?:19|20)\d{2}$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_METRIC_KEYS = {"value", "formula", "inputs"}  # the fundamentals._metric provenance shape
_REL_TOL = 0.01

# Magnitude suffix -> multiplier. Matching is UNIT-AWARE rather than a blind scale sweep: the
# cited text states its own unit, so "$98.8B" is checked as 9.88e10 and "99%" as 0.99 or 99 —
# never as each other. A free scale sweep (1e-9 … 1e9) lets a percentage "verify" against an
# unrelated dollar magnitude, which is a false confirmation in an anti-hallucination check.
_MAGNITUDES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}

REPORTED, DERIVED, UNVERIFIABLE, UNVERIFIED = "reported", "derived", "unverifiable", "unverified"


@dataclass
class CitationResult:
    hard_ok: bool = True
    violations: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    unverified_rate: float = 0.0
    n_numeric_sentences: int = 0
    n_checked_numbers: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    provenance_mismatches: list[str] = field(default_factory=list)


# --------------------------------------------------------------- formula recompute
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: None if b == 0 else a / b,
}


def eval_formula(formula: str, inputs: dict) -> float | None:
    """Evaluate a provenance `formula` over its `inputs`. Returns None when any referenced input
    is missing/None, on a zero denominator, or on a non-finite result.

    Deliberately an AST walk over a whitelist (+ - * / and unary minus, names, numeric literals)
    rather than eval(): formula strings are data, and data never gets to call functions or reach
    attributes. Anything outside the whitelist evaluates to None rather than raising.
    """
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            v = node.value
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        if isinstance(node, ast.Name):
            v = inputs.get(node.id)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = ev(node.operand)
            return None if v is None else -v
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = ev(node.left), ev(node.right)
            if a is None or b is None:
                return None
            return _BINOPS[type(node.op)](a, b)
        return None  # calls, attributes, subscripts, comparisons: refused

    try:
        out = ev(tree)
    except (TypeError, ZeroDivisionError, RecursionError):
        return None
    return out if (out is not None and math.isfinite(out)) else None


# ------------------------------------------------------------------- payload harvest
@dataclass
class _Evidence:
    reported: set[float] = field(default_factory=set)
    derived: set[float] = field(default_factory=set)
    unverifiable: set[float] = field(default_factory=set)
    mismatches: list[str] = field(default_factory=list)


def _add_num(acc: set[float], v) -> None:
    if isinstance(v, bool) or v is None:
        return
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        acc.add(float(v))
    elif isinstance(v, str):
        try:
            f = float(v.replace(",", ""))
        except ValueError:
            return
        if math.isfinite(f):
            acc.add(f)


def _is_metric(obj) -> bool:
    return isinstance(obj, dict) and set(obj) == _METRIC_KEYS and isinstance(obj["inputs"], dict)


def _harvest(obj, ev: _Evidence, path: str = "") -> None:
    """Walk a tool result, splitting numbers into reported vs recomputed-derived evidence."""
    if _is_metric(obj):
        inputs = obj["inputs"]
        for k, v in inputs.items():  # a metric's inputs ARE raw reported figures
            _add_num(ev.reported, v)
            del k
        recomputed = eval_formula(obj["formula"], inputs)
        stored = obj["value"]
        if recomputed is None:
            # No arithmetic check possible — record the stored value as unverifiable, not as
            # evidence. (A metric the tool itself refused to publish has value None: nothing
            # to record at all.)
            _add_num(ev.unverifiable, stored)
        else:
            ev.derived.add(recomputed)
            if isinstance(stored, (int, float)) and not isinstance(stored, bool):
                if not _close(float(stored), recomputed):
                    ev.mismatches.append(
                        f"{path or 'metric'}: stored value {stored!r} != recomputed "
                        f"{recomputed!r} from '{obj['formula']}'"
                    )
            elif stored is None:
                # The tool refused to publish (e.g. roic_proxy's nonsense-tax-rate guard) while
                # the bare formula still computes. Keep the recomputation out of evidence: a
                # number the tool declined to stand behind must not verify a citation.
                ev.derived.discard(recomputed)
                _add_num(ev.unverifiable, recomputed)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _harvest(v, ev, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _harvest(v, ev, f"{path}[{i}]")
    else:
        _add_num(ev.reported, obj)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * _REL_TOL, 1e-9)


def _candidates(value: float, unit: str | None) -> list[float]:
    """The single absolute quantity a written number denotes, given its own unit.

    Exactly ONE reading per unit, deliberately. This data layer states its contract — ratios and
    returns are FRACTIONS (0.25 == 25%) — so "26.9%" can only mean 0.269, never 26.9. Admitting a
    second reading per number is what lets a percentage confirm itself against an unrelated dollar
    magnitude; in an anti-hallucination check a false confirmation is far worse than a false miss,
    and a false miss merely raises the unverified rate for a human to read.
    """
    av = abs(value)
    if unit == "%":
        return [av / 100.0]
    if unit in _MAGNITUDES:
        return [av * _MAGNITUDES[unit]]
    return [av]  # bare number, or an "x" multiple: taken at face value


def _matches(value: float, unit: str | None, pool: set[float]) -> bool:
    cands = _candidates(value, unit)
    return any(_close(c, abs(p)) for c in cands for p in pool)


# --------------------------------------------------------------------- text parsing
def _sentence_numbers(sent: str) -> list[tuple[float, str | None]]:
    """(value, unit) for every numeric claim in a sentence; bare years/FY labels excluded."""
    out: list[tuple[float, str | None]] = []
    for m in _NUM.finditer(sent):
        core = m.group(1).replace(",", "").replace("$", "").strip()
        if _YEAR.match(core):  # a bare four-digit year is not a data claim
            continue
        try:
            val = float(core)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower() or None
        out.append((val, unit))
    return out


def _normalize_trace(trace) -> dict[str, dict]:
    """Accept {id: entry} or [{id, tool, args, result}, ...] -> {id: entry}."""
    if isinstance(trace, dict):
        return {str(k): v for k, v in trace.items()}
    out: dict[str, dict] = {}
    for e in trace or []:
        tid = str(e.get("id"))
        if tid:
            out[tid] = e
    return out


def check_citations(text: str, trace) -> CitationResult:
    """Run the hard tag checks and the reported/derived classification over one thesis + trace."""
    tr = _normalize_trace(trace)
    result = CitationResult()
    counts = {REPORTED: 0, DERIVED: 0, UNVERIFIABLE: 0, UNVERIFIED: 0}

    # HARD: every tag must resolve to a recorded call.
    for tag_num in sorted(set(_TAG.findall(text)), key=int):
        tag = f"T{tag_num}"
        if tag not in tr:
            result.violations.append(f"tag [{tag}] resolves to no recorded tool call in this trace")

    # Harvest each cited call's evidence once.
    evidence: dict[str, _Evidence] = {}
    for tag, entry in tr.items():
        ev = _Evidence()
        _harvest((entry or {}).get("result"), ev)
        evidence[tag] = ev
        result.provenance_mismatches.extend(f"[{tag}] {m}" for m in ev.mismatches)

    for sent in _SENT_SPLIT.split(text.strip()):
        if not sent.strip():
            continue
        nums = _sentence_numbers(sent)
        tags = [f"T{n}" for n in _TAG.findall(sent)]
        if not nums:
            continue
        result.n_numeric_sentences += 1
        if not tags:
            result.violations.append(
                f"numeric sentence carries no [Tn] citation: {sent.strip()[:90]!r}"
            )

        pooled = _Evidence()
        for t in tags:
            ev = evidence.get(t)
            if ev is None:
                continue
            pooled.reported |= ev.reported
            pooled.derived |= ev.derived
            pooled.unverifiable |= ev.unverifiable

        bases: list[str] = []
        for val, unit in nums:
            result.n_checked_numbers += 1
            if _matches(val, unit, pooled.reported):
                basis = REPORTED
            elif _matches(val, unit, pooled.derived):
                basis = DERIVED
            elif _matches(val, unit, pooled.unverifiable):
                basis = UNVERIFIABLE
            else:
                basis = UNVERIFIED
            counts[basis] += 1
            bases.append(basis)

        ok = all(b in (REPORTED, DERIVED) for b in bases)
        for t in tags:
            result.citations.append(
                Citation(
                    tag=t,
                    tool=(tr.get(t) or {}).get("tool"),
                    sentence=sent.strip()[:200],
                    numbers=[v for v, _ in nums],
                    bases=bases,
                    verified=ok,
                )
            )

    result.hard_ok = not result.violations
    result.counts = counts
    n = result.n_checked_numbers
    result.unverified_rate = round(counts[UNVERIFIED] / n, 4) if n else 0.0
    return result
