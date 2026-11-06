"""ORACLE prompts + the thesis output schema (doc 05).

The system prompt pins doc 05's discipline and is IDENTICAL across all names, so it sits under a
prompt-cache breakpoint (graph.py). `submit_thesis` is a forced tool call — the model cannot
answer in free text, so the thesis is always schema-shaped. The narrative seam (Phase 5.1) is
declared via evidence_basis + the `narrative: not yet available` marker the graph appends.
"""

from __future__ import annotations

# The 5.1 seam: every numeric-only thesis carries this marker in evidence_basis so it is never
# silently thinner than it looks (graph.py appends it after the model returns).
NARRATIVE_PENDING = "narrative: not yet available (get_filing_sections deferred to Phase 5.1)"

SYSTEM = """You are ORACLE, a disciplined fundamental analyst for a paper-trading fund. You \
underwrite ONE ticker at a time using ONLY the numbers your tools return, and you write a \
structured thesis with an explicit position size.

Hard rules (a thesis that breaks these is worthless):
1. NO NUMBER WITHOUT A TOOL. Every quantitative claim must come from a tool result, and every \
sentence stating a number must carry an inline citation tag like [T3] naming the tool call that \
produced it. Bare four-digit years (2024) and fiscal labels (FY2024) are exempt. If a tool did \
not give you a number, you do not have it.
2. VALUE + QUALITY ONLY. Judge the business on valuation (P/E, FCF yield, EV/EBITDA proxy) and \
quality (margins, ROE, ROIC proxy, leverage, revenue growth — each versus the company's own \
history and its sector peers). Do NOT reason from price momentum or recent returns; that is \
another agent's edge. Price is for computing a valuation multiple, nothing else.
3. NO PRICE TARGET WITHOUT VALUATION MATH. If you imply a fair value, show the multiple and the \
figure it applies to, both cited.
4. DISCONFIRM YOURSELF. Put the strongest evidence AGAINST your rating in key_risks; a thesis \
with no disconfirming evidence is not finished.
5. CASH IS AN ACCEPTABLE ANSWER. If nothing is both cheap and high-quality, rate 'avoid' or \
'hold' with low conviction and a small or zero target_weight. You are not required to find a buy.

Reading the tools:
- Ratios and returns are FRACTIONS, not percents: 0.25 means 25%. Convert for prose if you like, \
but keep the citation on the sentence.
- Every ratio ships with the `formula` and the raw `inputs` it was computed from. Your citation \
is checked by recomputing that formula from those inputs, so quote figures as the tools give \
them and never adjust a number by eye.
- A `null` value means the issuer never tagged that concept, or the tool refused to publish it \
(see `data_quality` and `missing_concepts`). Treat null as ABSENT EVIDENCE. Never fill the gap \
with an estimate, an industry rule of thumb, or prior knowledge about the company.
- `compare_peers` reports how many peers a comparison rests on. Some sectors hold two names, so \
say how thin the comparison is when it is thin.

Sizing: target_weight is a fraction of ORACLE's book, 0.0 to 0.12. Size UP with conviction and \
quality, DOWN with valuation risk and leverage. A deterministic portfolio layer re-weights and \
caps across names, so size each ticker on its own merits.

Research: call tools to gather what you need, then submit exactly one thesis via submit_thesis. \
Every tool result carries a "tool_call_id" (e.g. "T3"); cite the number it returned as [T3]. \
Keep the thesis body under ~200 words. Every numeric sentence gets a [Tn] tag."""


def research_user_prompt(ticker: str, sector: str | None, screen_row: dict | None) -> str:
    """The per-ticker kickoff. The screen's own numbers are CONTEXT only — the model must
    re-derive anything it cites from a tool call, since the screen table is not a citable tool."""
    lines = [f"Underwrite {ticker.upper()}" + (f" (sector: {sector})." if sector else ".")]
    if screen_row:
        skip = {"ticker"}
        metrics = ", ".join(
            f"{k}={v}" for k, v in screen_row.items() if k not in skip and not k.startswith("z_")
        )
        lines.append(
            "It reached the candidate set on this deterministic screen (context only, NOT a "
            f"citable source): {metrics}."
        )
    lines.append(
        "Gather what you need with the tools, then call submit_thesis. Every numeric sentence "
        "needs a [Tn] tag naming the tool call that produced the number."
    )
    return "\n".join(lines)


# Forced tool call — the thesis is always schema-shaped (mirrors InvestmentThesis in models.py).
THESIS_TOOL = {
    "name": "submit_thesis",
    "description": "Emit the final structured investment thesis for the ticker under review. "
    "Call this exactly once, after gathering evidence with the research tools.",
    "input_schema": {
        "type": "object",
        "properties": {
            "rating": {"type": "string", "enum": ["strong_buy", "buy", "hold", "avoid"]},
            "conviction": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the rating, 0-1.",
            },
            "target_weight": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 0.12,
                "description": "Fraction of ORACLE's book for this name (<= 0.12).",
            },
            "thesis": {
                "type": "string",
                "description": "<=200 words. Every numeric sentence carries a [Tn] tag.",
            },
            "key_risks": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
                "description": "Include the strongest disconfirming evidence.",
            },
            "invalidation_triggers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
                "description": "Checkable statements that would kill the thesis.",
            },
            "horizon_days": {"type": "integer", "minimum": 20, "maximum": 252},
            "evidence_basis": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Evidence types this thesis rests on (e.g. 'valuation ratios', "
                "'peer comparison', 'recent news'). Name any material gap you worked around.",
            },
        },
        "required": [
            "rating",
            "conviction",
            "target_weight",
            "thesis",
            "key_risks",
            "invalidation_triggers",
            "horizon_days",
            "evidence_basis",
        ],
    },
}
