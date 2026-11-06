"""ORACLE agent (doc 05) — the pure decision layer.

Reads pre-computed theses from ctx.extras['theses'] (built by orchestration/oracle.py via the
LangGraph pipeline, injected by the runner) and sizes a target-weight book through the
deterministic portfolio function. Nothing else: no LLM, no graph, no network, no DB — those all
live in orchestration. That keeps ORACLE a pure function of (context + extras) and backtestable
through BacktestBroker.

Caps come from config/risk_limits.yaml, never from the model (mirrors SentinelAgent):
  - max_position_weight_oracle (required, fail-loud — ORACLE trades on it)
  - max_positions_oracle (concentration control)
"""

from __future__ import annotations

from trident.agents.analyst.portfolio import build_decisions
from trident.agents.base import MarketContext, TradingAgent
from trident.config import load_risk_limits, required_risk_limit
from trident.models import AgentName, Decision


class OracleAgent(TradingAgent):
    name = AgentName.ORACLE

    def __init__(
        self, max_position_weight: float | None = None, max_positions: int | None = None
    ) -> None:
        limits = load_risk_limits()
        self.max_position_weight = (
            float(max_position_weight)
            if max_position_weight is not None
            else required_risk_limit("max_position_weight_oracle", limits)
        )
        # Concentration control — sourced from risk_limits.yaml, never a magic number in agent
        # code (doc 06 / risk-officer ruling, mirrors max_positions_sentinel).
        self.max_positions = int(
            max_positions if max_positions is not None else limits["max_positions_oracle"]
        )

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        theses = ctx.extras.get("theses", []) or []
        current = {p.ticker: ctx.book.weight(p.ticker) for p in ctx.book.positions if p.qty > 0}
        return build_decisions(
            self.name,
            theses,
            current,
            ctx.asof,
            max_positions=self.max_positions,
            max_position_weight=self.max_position_weight,
        )
