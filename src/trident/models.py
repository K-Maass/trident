"""Trident domain models — the shared vocabulary.

Every component speaks these types. Agents emit Decisions; the risk engine
emits RiskVerdicts; execution emits Orders/Fills; the ledger holds Positions.
Keep this file dependency-light (pydantic + stdlib only).
"""

from __future__ import annotations

import enum
import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- enums
class AgentName(enum.StrEnum):
    QUANT = "quant"
    SENTINEL = "sentinel"
    ORACLE = "oracle"
    BENCH = "bench"  # SPY buy-and-hold benchmark book


class AgentStatus(enum.StrEnum):
    ACTIVE = "active"
    HALTED = "halted"  # kill-switch fired; manual reactivation only
    DISABLED = "disabled"


class Side(enum.StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(enum.StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Rating(enum.StrEnum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    AVOID = "avoid"


class HealthSeverity(enum.StrEnum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"  # triggers global trade freeze


# ----------------------------------------------------------------------- decisions
class Decision(BaseModel):
    """What an agent WANTS. Target-weight based; never a raw order.

    Flow (golden rule 1): Decision -> RiskEngine -> Order -> Broker.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    agent: AgentName
    asof: datetime  # decision timestamp (UTC)
    ticker: str
    target_weight: float = Field(ge=0.0, le=0.25)  # fraction of the agent's book; long-only v1
    horizon_days: int = Field(default=5, ge=1, le=365)
    rationale: dict[str, Any]  # agent-specific explanation blob; NEVER empty
    inputs_digest: str = ""  # sha256 of the inputs the agent saw (reproducibility)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("rationale")
    @classmethod
    def _rationale_nonempty(cls, v: dict[str, Any]) -> dict[str, Any]:
        if not v:
            raise ValueError("rationale must not be empty — golden rule 2")
        return v

    @staticmethod
    def digest(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]


class RiskVerdict(BaseModel):
    """Risk engine's answer to one Decision."""

    decision_id: str
    approved: bool
    adjusted_target_weight: float | None = None  # set when clamped
    reasons: list[str] = Field(default_factory=list)  # machine-readable, e.g. "clamp:position_cap"


# -------------------------------------------------------------------------- orders
class Order(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    decision_id: str | None
    agent: AgentName
    client_order_id: str  # f"{agent}-{uuid4().hex[:12]}" — ledger attribution key
    ticker: str
    side: Side
    qty: float = Field(gt=0)
    limit_price: float | None = Field(default=None, gt=0)  # marketable limit; None=market
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: str | None = None
    risk_notes: list[str] = Field(default_factory=list)
    created_at: datetime

    @classmethod
    def make_client_order_id(cls, agent: AgentName) -> str:
        return f"{agent.value}-{uuid4().hex[:12]}"


class Fill(BaseModel):
    order_id: str
    agent: AgentName
    ticker: str
    side: Side
    qty: float = Field(gt=0)
    price: float = Field(gt=0)
    ts: datetime
    fees: float = 0.0


# ------------------------------------------------------------------------ holdings
class Position(BaseModel):
    agent: AgentName
    ticker: str
    qty: float = Field(ge=0)
    avg_cost: float = Field(ge=0)
    last_price: float | None = None

    @property
    def market_value(self) -> float:
        return self.qty * (self.last_price or self.avg_cost)


class BookSnapshot(BaseModel):
    """One agent's virtual portfolio at a point in time."""

    agent: AgentName
    ts: datetime
    cash: float
    positions: list[Position] = Field(default_factory=list)

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions)

    def weight(self, ticker: str) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        pos = next((p for p in self.positions if p.ticker == ticker), None)
        return (pos.market_value / eq) if pos else 0.0


# ------------------------------------------------------------------------- signals
class Signal(BaseModel):
    """Intermediate scored output (pre-decision), persisted for research."""

    agent: AgentName
    ts: datetime
    ticker: str
    score: float = Field(ge=-1.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NewsItem(BaseModel):
    source: str  # finnhub | rss:<feed> | reddit:<sub> | stocktwits
    ticker: str | None
    title: str
    body: str | None = None
    url: str
    published_at: datetime
    url_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.url_hash:
            self.url_hash = hashlib.sha256(self.url.encode()).hexdigest()[:20]


class Citation(BaseModel):
    """One inline `[Tn]` tag from a thesis, resolved against the ticker's tool trace.

    The anti-hallucination unit. `tool` is None when the tag resolves to no recorded call (a
    HARD citation failure). Each number in the tagged sentence is classified in `bases`:

      "reported"     — matched a raw figure the filing/market actually reported.
      "derived"      — matched a metric RECOMPUTED from its {formula, inputs} provenance, so the
                       cited number is checked against what the raw inputs imply, not against the
                       tool's own summary field.
      "unverifiable" — the metric's provenance is incomplete (a None input), so no arithmetic
                       check is possible; surfaced, never counted as a match.
      "unverified"   — found nowhere. This is the hallucination signal.
    """

    tag: str  # e.g. "T3"
    tool: str | None = None  # resolved tool name; None = unresolved (hard fail)
    sentence: str = ""  # the sentence carrying this tag
    numbers: list[float] = Field(default_factory=list)  # numeric claims in that sentence
    bases: list[str] = Field(default_factory=list)  # per-number basis, parallel to `numbers`
    verified: bool | None = None  # every number matched (reported or derived)


class InvestmentThesis(BaseModel):
    """ORACLE's structured output — schema-enforced discipline (doc 05)."""

    agent: AgentName = AgentName.ORACLE
    # Persistence key (session identity), not a data cutoff — the real cutoff is in tool_trace;
    # orchestration stamps this to the session's midnight UTC so re-runs upsert (see
    # orchestration.oracle._session_asof).
    asof: datetime
    ticker: str
    rating: Rating
    conviction: float = Field(ge=0.0, le=1.0)
    target_weight: float = Field(ge=0.0, le=0.12)
    thesis: str = Field(max_length=1600)  # ~200 words, hard cap
    key_risks: list[str] = Field(min_length=1, max_length=5)
    invalidation_triggers: list[str] = Field(min_length=1, max_length=5)  # checkable statements
    horizon_days: int = Field(ge=20, le=252)
    # A citation is only auditable if the cited tool result is stored STRUCTURALLY here — a
    # string pointer to a trace in a second store (the old `tool_trace_ref`) would make the
    # audit trail depend on a store that doesn't exist. Keyed by trace id: {"T1": {...}}.
    tool_trace: dict | None = None
    citations: list[Citation] = Field(default_factory=list)  # resolved [Tn] tags + verdicts
    # Declares its own evidence basis so a numeric-only thesis is never silently thinner than it
    # looks — carries the explicit "narrative: not yet available" marker until Phase 5.1.
    evidence_basis: list[str] = Field(default_factory=list)
    narrative_summary: str | None = None  # filled by Phase 5.1 (get_filing_sections)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class HealthEvent(BaseModel):
    ts: datetime
    severity: HealthSeverity
    source: str  # e.g. "data.market", "reconciliation", "oracle_cost"
    detail: str
