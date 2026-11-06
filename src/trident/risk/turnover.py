"""Exact B3 turnover authorization over a portfolio's frozen persisted evidence.

This calculation is pure. Its caller must read the evidence and persist the resulting
commitment under the same portfolio lock; it is not a replacement for that SQL boundary.
Actual executions are never rejected or removed because they exceed the authorization budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal

Side = Literal["buy", "sell"]
ZERO = Decimal("0")


def _amount(value: Decimal, name: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if value < ZERO or (positive and value == ZERO):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class CanonicalBaseline:
    portfolio_id: str
    session: date
    snapshot_id: int
    equity: Decimal
    eligible: bool

    def __post_init__(self) -> None:
        _amount(self.equity, "baseline equity", positive=True)
        if self.snapshot_id <= 0:
            raise ValueError("baseline must identify a persisted canonical snapshot")


@dataclass(frozen=True)
class ExactExecution:
    execution_id: str
    portfolio_id: str
    session: date
    ticker: str
    side: Side
    notional: Decimal

    def __post_init__(self) -> None:
        _amount(self.notional, "exact execution notional", positive=True)
        if not self.execution_id or self.side not in ("buy", "sell"):
            raise ValueError("execution identity and side are required")


@dataclass(frozen=True)
class Commitment:
    order_id: str
    portfolio_id: str
    ticker: str
    side: Side
    remaining_qty: Decimal
    limit_price: Decimal
    reference_price: Decimal

    def __post_init__(self) -> None:
        _amount(self.remaining_qty, "remaining quantity")
        _amount(self.limit_price, "limit price", positive=True)
        _amount(self.reference_price, "reference price", positive=True)
        if not self.order_id or self.side not in ("buy", "sell"):
            raise ValueError("commitment identity and side are required")

    @property
    def notional(self) -> Decimal:
        price = (
            self.limit_price if self.side == "buy" else max(self.reference_price, self.limit_price)
        )
        return self.remaining_qty * price


@dataclass(frozen=True)
class TurnoverExemption:
    """Evidence resolved from a durable BENCH intent or persisted kill-switch event."""

    kind: Literal["initial_bench", "kill_switch"]
    evidence_id: str
    portfolio_id: str
    order_id: str
    eligible_from_session: date


@dataclass(frozen=True)
class TurnoverDecision:
    approved: bool
    reason: str
    budget: Decimal | None
    exact_filled_notional: Decimal
    outstanding_notional: Decimal
    requested_notional: Decimal
    projected_notional: Decimal
    exemption_id: str | None


def authorize_turnover(
    *,
    portfolio_id: str,
    session: date,
    previous_session: date,
    max_daily_turnover: Decimal,
    baseline: CanonicalBaseline | None,
    executions: tuple[ExactExecution, ...],
    commitments: tuple[Commitment, ...],
    proposed: Commitment,
    exemption: TurnoverExemption | None = None,
) -> TurnoverDecision:
    """Authorize one new commitment; gross fills include exempt executions as well.

    Pass all unresolved commitments, including prior-session uncertain submissions.
    Recovery of an already reserved order does not reserve it a second time through this API.
    ``previous_session`` must come from the authoritative XNYS calendar, not date subtraction.
    """
    _amount(max_daily_turnover, "turnover fraction")
    if max_daily_turnover > Decimal("1"):
        raise ValueError("turnover fraction exceeds one")
    if not portfolio_id or proposed.portfolio_id != portfolio_id or previous_session >= session:
        raise ValueError("invalid portfolio or session boundary")
    if any(x.portfolio_id != portfolio_id for x in (*executions, *commitments)):
        raise ValueError("turnover evidence crosses portfolios")
    if len({x.execution_id for x in executions}) != len(executions):
        raise ValueError("duplicate execution evidence")
    if len({x.order_id for x in commitments}) != len(commitments):
        raise ValueError("duplicate outstanding commitment")
    if any(x.order_id == proposed.order_id for x in commitments):
        raise ValueError("existing commitment requires recovery, not a second authorization")
    if any(x.session > session for x in executions):
        raise ValueError("future execution evidence")

    today = tuple(x for x in executions if x.session == session)
    filled = sum((x.notional for x in today), ZERO)
    outstanding = sum((x.notional for x in commitments), ZERO)
    requested = proposed.notional
    projected = filled + outstanding + requested
    valid_baseline = (
        baseline is not None
        and baseline.portfolio_id == portfolio_id
        and baseline.session == previous_session
        and baseline.eligible
    )
    budget = baseline.equity * max_daily_turnover if valid_baseline else None

    def result(approved: bool, reason: str, exemption_id: str | None = None) -> TurnoverDecision:
        return TurnoverDecision(
            approved, reason, budget, filled, outstanding, requested, projected, exemption_id
        )

    if proposed.remaining_qty == ZERO:
        return result(False, "zero_quantity")
    if any(x.ticker == proposed.ticker and x.side != proposed.side for x in today):
        return result(False, "same_session_opposite_fill")
    if any(
        x.ticker == proposed.ticker and x.side != proposed.side and x.remaining_qty > ZERO
        for x in commitments
    ):
        return result(False, "unresolved_opposite_commitment")
    if exemption is not None:
        if (
            not exemption.evidence_id
            or exemption.portfolio_id != portfolio_id
            or exemption.order_id != proposed.order_id
            or exemption.eligible_from_session > session
            or (exemption.kind == "initial_bench" and proposed.side != "buy")
            or (exemption.kind == "kill_switch" and proposed.side != "sell")
            or exemption.kind not in ("initial_bench", "kill_switch")
        ):
            return result(False, "invalid_turnover_exemption")
        return result(True, "turnover_exempt", exemption.evidence_id)
    if budget is None:
        return result(False, "missing_eligible_previous_session_baseline")
    if projected > budget:
        return result(False, "daily_turnover_exceeded")
    return result(True, "within_daily_turnover")
