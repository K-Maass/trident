"""The agent contract. All three agents implement exactly this.

Purity rules (what makes agents testable and backtestable):
  - Agents READ the MarketContext, RETURN Decisions. No broker I/O. No DB writes
    (the runner persists). No wall-clock reads — use ctx.asof.
  - Same context in -> same decisions out (LLM agents: temperature 0).
  - Decisions are TARGET WEIGHTS for the agent's whole book, not deltas.
    "I want 6% NVDA, 6% MSFT, rest cash." The execution layer diffs vs the
    current book. Emitting an empty list means "keep current book unchanged";
    to go to cash, emit explicit 0-weight decisions for held names.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from trident.models import AgentName, BookSnapshot, Decision


class PriceHistory(Protocol):
    """Thin protocol over data.market so agents don't import pandas plumbing.

    get(tickers, lookback_days) -> tidy DataFrame indexed by (date, ticker) with
    columns [open, high, low, close, volume, dollar_volume]. Guaranteed to contain
    NOTHING after ctx.asof (enforced by data layer — the anti-look-ahead seam).
    """

    def get(self, tickers: list[str], lookback_days: int) -> Any: ...


class MarketContext(BaseModel):
    """Everything an agent may look at for one decision cycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    asof: datetime  # UTC; in backtests this is the simulated 'now'
    universe: list[str]
    book: BookSnapshot  # THIS agent's virtual portfolio (cash, positions, equity)
    prices: Any  # PriceHistory implementation
    extras: dict[str, Any] = {}  # agent-specific feeds injected by the runner:
    # SENTINEL: {"signals": today's sentiment aggregates}
    # ORACLE:   {"fundamentals": handle, "news": handle}


class TradingAgent(abc.ABC):
    name: AgentName

    @abc.abstractmethod
    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        """Return target-weight Decisions with non-empty rationale (golden rule 2)."""

    def describe(self) -> str:
        """One-line self description used in reports/dashboard."""
        return f"{self.name.value}"


class DummyAgent(TradingAgent):
    """Phase 2 pipe-cleaner: wants 2% AAPL, proving decision->risk->order->fill->ledger.

    2% (not 1%) so a $30k book buys at least one whole ~$300 share — otherwise sizing
    floors to 0 and no order is produced.
    """

    name = AgentName.QUANT  # borrows quant's book for the smoke test

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        return [
            Decision(
                agent=self.name,
                asof=ctx.asof,
                ticker="AAPL",
                target_weight=0.02,
                horizon_days=1,
                rationale={"reason": "dummy end-to-end pipe test"},
                inputs_digest=Decision.digest({"dummy": True}),
            )
        ]
