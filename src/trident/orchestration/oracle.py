"""ORACLE orchestration entry points (doc 05 / config/schedule.yaml).

The callable targets the Phase-6 scheduler names (`agents.run_oracle`,
`oracle.check_invalidation_triggers`). All LLM / graph / network work lives HERE, never in the
pure agent: run_oracle validates the model strings, runs the LangGraph pipeline, persists the
theses, then drives the pure agent through the money path with theses injected via `extras` (so
the runner's inputs_digest covers them — golden rule 2).

Mirrors orchestration/sentinel.py: `extras` is inert JSON data (the runner's `_extras_are_data`
guard rejects anything else, so theses are passed as plain dicts).

This module defines callable entry points. The portfolio release does not include a job runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import text

from trident.agents.analyst import oracle_agent
from trident.agents.analyst.graph import run_graph
from trident.agents.analyst.llm import AnalystLLM, validate_models
from trident.agents.analyst.screen import SCREEN_TOP_N
from trident.broker.base import Broker
from trident.config import get_settings, load_risk_limits
from trident.data import store
from trident.models import AgentName, HealthEvent, HealthSeverity
from trident.orchestration import runner

COST_SOURCE = "oracle_cost"


def _session_asof(asof: datetime) -> datetime:
    """PERSISTENCE KEY for a thesis, NOT a data cutoff: midnight UTC of the run's calendar day.

    Theses upsert on (agent_id, ticker, asof), so this key must be constant across re-runs of the
    same session — a raw datetime.now() makes every re-run a fresh row and the unique index
    (ux_theses_agent_ticker_asof) decorative. It is deliberately NOT applied to
    research/tools/decisions/clock math, which need the real timestamp (news `until=asof`, the
    runner's minutes-to-close reject).

    The thesis is researched at the REAL timestamp; the true data cutoff lives in `tool_trace`
    (each tool result carries the asof it was bound to). This key is stamped to midnight — i.e.
    deliberately EARLIER than the research moment — so a future reader who mistakes it for a
    cutoff understates rather than overstates what the thesis actually knew. Erring toward
    "knew less" is the safe direction; erring toward "knew more" would imply look-ahead.
    """
    return asof.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass
class OracleRunResult:
    asof: datetime
    n_theses: int = 0
    n_dropped: int = 0
    screen_table: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    theses: list = field(default_factory=list)  # InvestmentThesis objects (for the artifact)
    cost: dict = field(default_factory=dict)
    degraded: bool = False  # over soft budget -> fewer deep names
    refused: bool = False  # over hard ceiling -> research phase skipped
    run: object = None  # the runner.RunResult, or None if research was refused


async def _month_to_date_spend() -> float:
    """Sum estimated_usd across this calendar month's oracle_cost events. Best-effort — a parse
    miss counts as $0 and never blocks a run."""
    async with store.get_engine().connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT detail FROM health_events WHERE source=:s "
                    "AND ts >= date_trunc('month', now())"
                ),
                {"s": COST_SOURCE},
            )
        ).all()
    total = 0.0
    for r in rows:
        try:
            total += float(json.loads(r[0]).get("estimated_usd", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return total


async def _write_cost_event(meter, extra: dict | None = None) -> dict:
    """One INFO oracle_cost row per run — written even on a zero-thesis run."""
    summary = (
        meter.summary() if meter is not None else {"calls": 0, "estimated_usd": 0.0, "by_model": {}}
    )
    if extra:
        summary.update(extra)
    await store.write_health_event(
        HealthEvent(
            ts=datetime.now(UTC),
            severity=HealthSeverity.INFO,
            source=COST_SOURCE,
            detail=json.dumps(summary),
        )
    )
    return summary


async def run_oracle(
    broker: Broker,
    dry_run: bool = True,
    asof: datetime | None = None,
    *,
    validate: bool = True,
    llm: AnalystLLM | None = None,
) -> OracleRunResult:
    """Weekly ORACLE cycle: validate models -> screen/research/thesis graph -> persist theses
    (upsert) -> run the pure agent through the money path with extras={theses} -> log cost.

    The $25/mo alarm is WARN, never CRITICAL: over budget the run continues with deep names cut
    to oracle_max_deep_names (doc 09); the hard ceiling refuses the research phase entirely
    (still logging a zero-cost event). Neither trips runner._global_freeze()."""
    from trident import persistence

    asof = asof or datetime.now(UTC)  # real timestamp: research, tools, clock, decisions
    session_asof = _session_asof(asof)  # stable upsert key for thesis identity (see _session_asof)
    settings = get_settings()
    limits = load_risk_limits()
    result = OracleRunResult(asof=asof)

    if validate:
        await validate_models(settings.oracle_model, settings.bulk_model)

    # ---- budget governor (WARN, never CRITICAL) -------------------------------
    spend = await _month_to_date_spend()
    soft = float(limits.get("oracle_monthly_budget_usd", 25.0))
    hard = float(limits.get("oracle_monthly_hard_ceiling_usd", 50.0))
    top_n = SCREEN_TOP_N
    if spend >= hard:
        result.refused = True
        await store.write_health_event(
            HealthEvent(
                ts=datetime.now(UTC),
                severity=HealthSeverity.WARN,
                source="oracle_budget",
                detail=f"month-to-date ${spend:.2f} >= hard ceiling ${hard:.2f}; research refused",
            )
        )
        result.cost = await _write_cost_event(
            None, {"refused": True, "mtd_usd_before": round(spend, 2)}
        )
        return result
    if spend >= soft:
        result.degraded = True
        top_n = int(limits.get("oracle_max_deep_names", 6))
        await store.write_health_event(
            HealthEvent(
                ts=datetime.now(UTC),
                severity=HealthSeverity.WARN,
                source="oracle_budget",
                detail=f"month-to-date ${spend:.2f} >= budget ${soft:.2f}; deep names -> {top_n}",
            )
        )

    # ---- research -------------------------------------------------------------
    llm = llm or AnalystLLM()
    book = await persistence.load_book(AgentName.ORACLE)
    held = [p.ticker for p in book.snapshot(asof).positions if p.qty > 0]
    graph = await run_graph(asof, held, llm=llm, top_n=top_n)  # research uses the REAL asof
    # Normalize ONLY the persistence key: the thesis was researched at `asof` (its tool_trace
    # carries the real cutoff), but its identity is the session so re-runs upsert (see
    # _session_asof). Everything downstream — decisions, clock, extras digest — keeps `asof`.
    for th in graph.theses:
        th.asof = session_asof
    result.screen_table = graph.screen_table
    result.dropped = graph.dropped
    result.theses = graph.theses
    result.n_theses = len(graph.theses)
    result.n_dropped = len(graph.dropped)

    for th in graph.theses:
        await persistence.upsert_thesis(th)
    for d in graph.dropped:  # no thesis -> no position; WARN so the spot-check sees it
        await store.write_health_event(
            HealthEvent(
                ts=datetime.now(UTC),
                severity=HealthSeverity.WARN,
                source="oracle.drop",
                detail=f"{d['ticker']} dropped: {d['reason']}",
            )
        )

    # ---- money path (theses injected as inert JSON data) ----------------------
    extras = {"theses": [th.model_dump(mode="json") for th in graph.theses]}
    result.run = await runner.run_agent(
        oracle_agent(), broker, dry_run=dry_run, asof=asof, extras=extras
    )

    result.cost = await _write_cost_event(graph.meter, {"mtd_usd_before": round(spend, 2)})
    return result


async def check_invalidation_triggers(broker: Broker | None = None, asof: datetime | None = None):
    """Phase-7 stub. `config/schedule.yaml` already names oracle.check_invalidation_triggers
    (Sunday), so the callable must exist; the evaluation itself lands in Phase 7."""
    return None
