"""ORACLE's LangGraph pipeline (doc 05): screen -> research -> thesis.

- screen  : deterministic Python (screen.run_screen) -> candidates + the ranked table.
- research: per candidate, a bounded ReAct tool loop on oracle_model (temperature 0, max 12 tool
            calls), each call recorded to a per-ticker trace as {id:"T1", tool, args, result}.
            Bounded concurrency keeps the run inside the acceptance window without exhausting
            the connection pool.
- thesis  : each candidate's forced submit_thesis call is citation-checked (citations.py). A
            hard failure gets ONE repair round-trip quoting the exact violations; still failing
            -> dropped (no thesis -> no position, the fail-safe direction).

The graph writes no DB rows and no health events: it returns theses + drops + the screen table +
the cost meter, and orchestration/oracle.py persists and logs. Config (LLM, concurrency) is
injected via closures, so state carries only JSON-shaped data.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from trident.agents.analyst import prompts
from trident.agents.analyst.citations import CitationResult, check_citations
from trident.agents.analyst.llm import AnalystLLM
from trident.agents.analyst.screen import SCREEN_TOP_N, run_screen
from trident.agents.analyst.tools import TOOL_SCHEMAS, TOOLS, ToolContext
from trident.config import load_sectors, load_universe
from trident.models import InvestmentThesis, Rating

MAX_TOOL_CALLS = 12  # per name (doc 05)
MAX_TOKENS = 2000
DEFAULT_CONCURRENCY = 3  # semaphore — inside the 30-min acceptance, no pool exhaustion


@dataclass
class GraphResult:
    theses: list[InvestmentThesis] = field(default_factory=list)
    screen_table: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)  # {ticker, reason, violations}
    traces: dict[str, dict] = field(default_factory=dict)
    meter: Any = None  # the run's CostMeter (from the injected/own AnalystLLM)


class _State(TypedDict, total=False):
    asof: datetime
    held: list[str]
    universe: list[str]
    candidates: list[str]
    screen_table: list[dict]
    per_ticker: dict[str, dict]


# ------------------------------------------------------------------- ReAct loop
def _tool_result_block(tool_use_id: str, payload: dict) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(payload, default=str),
    }


async def _run_tools(
    ctx: ToolContext, tool_uses, trace: dict, counter: list[int]
) -> tuple[list, dict | None]:
    """Execute the model's tool calls, record to the trace, return (blocks, thesis_input)."""
    blocks, thesis_input = [], None
    for tu in tool_uses:
        name, args, tuid = tu.name, dict(tu.input), tu.id
        if name == "submit_thesis":
            thesis_input = args
            blocks.append(_tool_result_block(tuid, {"ok": True, "note": "thesis received"}))
            continue
        if name not in TOOLS:
            blocks.append(
                _tool_result_block(tuid, {"available": False, "reason": f"unknown tool {name}"})
            )
            continue
        if counter[0] >= MAX_TOOL_CALLS:
            blocks.append(
                _tool_result_block(
                    tuid,
                    {"available": False, "reason": "tool budget exhausted; call submit_thesis now"},
                )
            )
            continue
        counter[0] += 1
        tid = f"T{counter[0]}"
        try:
            result = await TOOLS[name](ctx, **args)
        except Exception as e:  # noqa: BLE001 — a tool error is data, never a crash (rule 8)
            result = {"available": False, "error": f"{type(e).__name__}: {e}"}
        trace[tid] = {"id": tid, "tool": name, "args": args, "result": result}
        blocks.append(_tool_result_block(tuid, {"tool_call_id": tid, **result}))
    return blocks, thesis_input


def _tools_payload() -> list[dict]:
    """Research tools + the forced thesis tool; the last schema carries the cache breakpoint
    (the tool list is byte-identical across all names)."""
    schemas = [dict(s) for s in TOOL_SCHEMAS] + [dict(prompts.THESIS_TOOL)]
    schemas[-1] = {**schemas[-1], "cache_control": {"type": "ephemeral"}}
    return schemas


def _system_payload() -> list[dict]:
    return [{"type": "text", "text": prompts.SYSTEM, "cache_control": {"type": "ephemeral"}}]


async def _force_thesis(llm, system, tools, messages) -> dict | None:
    messages = messages + [
        {"role": "user", "content": "You have gathered enough. Call submit_thesis now."}
    ]
    resp = await llm.create(
        max_tokens=MAX_TOKENS,
        system=system,
        tools=tools,
        messages=messages,
        tool_choice={"type": "tool", "name": "submit_thesis"},
    )
    for b in resp.content:
        if getattr(b, "type", None) == "tool_use" and b.name == "submit_thesis":
            return dict(b.input)
    return None


async def _repair(
    llm, system, tools, messages, trace, citation
) -> tuple[dict | None, CitationResult]:
    """One repair round-trip quoting the exact violations, forced back through submit_thesis."""
    quote = "; ".join(citation.violations[:5])
    messages = messages + [
        {
            "role": "user",
            "content": (
                "Your thesis failed citation checks and cannot be accepted as written:\n"
                f"{quote}\n"
                "Every sentence stating a number must carry a [Tn] tag naming a tool call you "
                "actually made (see the tool_call_id in each result). Resubmit via submit_thesis "
                "with the tags fixed. Do not invent numbers you did not fetch."
            ),
        }
    ]
    fixed = await _force_thesis(llm, system, tools, messages)
    if fixed is None:
        return None, citation
    return fixed, check_citations(fixed.get("thesis", ""), trace)


async def _underwrite(
    ticker: str, asof: datetime, sector: str | None, screen_row: dict | None, llm: AnalystLLM
) -> dict:
    """One candidate: ReAct loop -> forced submit_thesis -> citation check -> one repair."""
    ctx = ToolContext(asof=asof)
    system, tools = _system_payload(), _tools_payload()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": prompts.research_user_prompt(ticker, sector, screen_row)}
    ]
    trace: dict[str, dict] = {}
    counter = [0]
    thesis_input: dict | None = None

    for _ in range(MAX_TOOL_CALLS + 2):
        resp = await llm.create(
            max_tokens=MAX_TOKENS, system=system, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            break
        blocks, thesis_input = await _run_tools(ctx, tool_uses, trace, counter)
        messages.append({"role": "user", "content": blocks})
        if thesis_input is not None:
            break

    if thesis_input is None:  # never submitted -> force it once
        thesis_input = await _force_thesis(llm, system, tools, messages)
        if thesis_input is None:
            return {
                "thesis_input": None,
                "trace": trace,
                "citation": None,
                "dropped_reason": "no thesis produced",
            }

    citation = check_citations(thesis_input.get("thesis", ""), trace)
    if not citation.hard_ok:
        thesis_input, citation = await _repair(llm, system, tools, messages, trace, citation)
        if thesis_input is None or not citation.hard_ok:
            return {
                "thesis_input": thesis_input,
                "trace": trace,
                "citation": citation,
                "dropped_reason": f"citation hard-fail after repair: {citation.violations[:3]}",
            }
    return {
        "thesis_input": thesis_input,
        "trace": trace,
        "citation": citation,
        "dropped_reason": None,
    }


# --------------------------------------------------------------------- assembly
def _build_thesis(ticker: str, asof: datetime, data: dict) -> InvestmentThesis:
    ti = data["thesis_input"]
    citation: CitationResult = data["citation"]
    basis = list(ti.get("evidence_basis") or [])
    if prompts.NARRATIVE_PENDING not in basis:  # the 5.1 seam, always declared
        basis.append(prompts.NARRATIVE_PENDING)
    return InvestmentThesis(
        asof=asof,
        ticker=ticker,
        rating=Rating(ti["rating"]),
        conviction=float(ti["conviction"]),
        target_weight=float(ti["target_weight"]),
        thesis=ti["thesis"][:1600],
        key_risks=list(ti["key_risks"])[:5],
        invalidation_triggers=list(ti["invalidation_triggers"])[:5],
        horizon_days=int(ti["horizon_days"]),
        tool_trace=data.get("trace") or {},
        citations=citation.citations if citation else [],
        evidence_basis=basis,
        narrative_summary=None,
    )


def build_graph(llm: AnalystLLM, concurrency: int = DEFAULT_CONCURRENCY, top_n: int = SCREEN_TOP_N):
    """Compile the screen->research->thesis StateGraph over an injected LLM."""
    sectors = load_sectors()

    async def screen_node(state: _State) -> _State:
        res = run_screen(
            state.get("universe") or load_universe(),
            state["asof"],
            held=state.get("held", []),
            top_n=top_n,
        )
        return {
            "candidates": res.candidates,
            "screen_table": res.table.to_dict(orient="records"),
        }

    async def research_node(state: _State) -> _State:
        sem = asyncio.Semaphore(concurrency)
        table_by = {r["ticker"]: r for r in state.get("screen_table", [])}

        async def one(t: str) -> tuple[str, dict]:
            async with sem:
                return t, await _underwrite(t, state["asof"], sectors.get(t), table_by.get(t), llm)

        done = await asyncio.gather(*(one(t) for t in state["candidates"]))
        return {"per_ticker": dict(done)}

    async def thesis_node(state: _State) -> _State:
        return state  # assembly happens in run_graph (no LLM here)

    g = StateGraph(_State)
    g.add_node("screen", screen_node)
    g.add_node("research", research_node)
    g.add_node("thesis", thesis_node)
    g.add_edge(START, "screen")
    g.add_edge("screen", "research")
    g.add_edge("research", "thesis")
    g.add_edge("thesis", END)
    return g.compile()


async def run_graph(
    asof: datetime,
    held: list[str] | None = None,
    *,
    llm: AnalystLLM | None = None,
    universe: list[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    top_n: int = SCREEN_TOP_N,
) -> GraphResult:
    """Run the compiled graph and assemble the GraphResult."""
    llm = llm or AnalystLLM()
    graph = build_graph(llm, concurrency=concurrency, top_n=top_n)
    init: _State = {
        "asof": asof,
        "held": [h.upper() for h in (held or [])],
        "universe": [t.upper() for t in (universe or load_universe())],
    }
    final = await graph.ainvoke(init)

    out = GraphResult(screen_table=final.get("screen_table", []), meter=llm.meter)
    for ticker, data in (final.get("per_ticker") or {}).items():
        out.traces[ticker] = data.get("trace") or {}
        if data.get("dropped_reason"):
            cit = data.get("citation")
            out.dropped.append(
                {
                    "ticker": ticker,
                    "reason": data["dropped_reason"],
                    "violations": (cit.violations if cit else []),
                }
            )
            continue
        try:
            out.theses.append(_build_thesis(ticker, asof, data))
        except Exception as e:  # noqa: BLE001 — a malformed payload drops the name, never crashes
            out.dropped.append(
                {
                    "ticker": ticker,
                    "reason": f"thesis assembly failed: {type(e).__name__}: {e}",
                    "violations": [],
                }
            )
    return out
