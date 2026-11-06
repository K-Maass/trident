"""Virtual per-agent portfolio — the source of truth for per-agent P&L (doc 01/06).

One Book per agent inside the single broker account; `client_order_id` attribution
(agent prefix) routes each fill to the right Book. This module is PURE accounting:
fills, dividends, and marks are inputs — it never touches the broker or market data, so
it is exhaustively unit-testable with hand-computed fixtures (golden rule 5: money math
gets tests). DB persistence (positions/equity_snapshots) is a thin layer wired in the
Phase 2 runner; the accounting logic lives here.

Money is Decimal quantized to 1e-6 (== NUMERIC(18,6)) so the in-memory book and Postgres
agree to the last digit and floats never drift the reconciliation invariant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal

from trident.models import AgentName, BookSnapshot, Fill, Position, Side

# Seeded in db/schema.sql: QUANT/SENTINEL/ORACLE $30k each, BENCH $10k.
STARTING_CASH: dict[AgentName, Decimal] = {
    AgentName.QUANT: Decimal("30000"),
    AgentName.SENTINEL: Decimal("30000"),
    AgentName.ORACLE: Decimal("30000"),
    AgentName.BENCH: Decimal("10000"),
}

_Q = Decimal("0.000001")  # NUMERIC(18,6)


def _d(x: float | int | str | Decimal) -> Decimal:
    """Coerce via str() so a float's binary noise never enters the ledger."""
    return Decimal(str(x)).quantize(_Q, rounding=ROUND_HALF_EVEN)


class LedgerError(RuntimeError):
    """Accounting invariant violated (e.g. an oversell — long-only v1 has no shorts)."""


@dataclass
class _Lot:
    qty: Decimal
    avg_cost: Decimal  # per-share, includes entry fees in the basis
    last_price: Decimal | None = None

    @property
    def mark(self) -> Decimal:
        return self.last_price if self.last_price is not None else self.avg_cost

    @property
    def market_value(self) -> Decimal:
        return _d(self.qty * self.mark)


@dataclass
class Book:
    """One agent's virtual portfolio. Mutated by apply_fill/apply_dividend/mark_to_market."""

    agent: AgentName
    cash: Decimal
    positions: dict[str, _Lot] = field(default_factory=dict)
    realized_pnl: Decimal = Decimal("0")

    @classmethod
    def new(cls, agent: AgentName, starting_cash: float | Decimal | None = None) -> Book:
        cash = _d(starting_cash) if starting_cash is not None else STARTING_CASH[agent]
        return cls(agent=agent, cash=_d(cash))

    # ------------------------------------------------------------------- fills
    def apply_fill(self, fill: Fill) -> Decimal:
        """Apply a fill; return the realized P&L booked by THIS fill (0 for buys).

        Buys: cash -= qty*price + fees; entry fees fold into the average-cost basis.
        Sells: cash += qty*price - fees; realized = qty*(price - avg_cost) - fees.
        Selling more than held raises LedgerError (no shorting — golden rule 7).
        """
        ticker = fill.ticker.upper()  # normalize: positions/marks/dividends must agree on case
        qty, price, fees = _d(fill.qty), _d(fill.price), _d(fill.fees)
        lot = self.positions.get(ticker)

        if fill.side == Side.BUY:
            # Cash may go negative here by design: the ledger mirrors real broker fills, and a
            # cash-only paper account won't fill an overspend. The no-leverage guarantee (golden
            # rule 7) lives upstream in the risk engine + broker, not in this accountant.
            cost = _d(qty * price + fees)
            self.cash = _d(self.cash - cost)
            if lot is None:
                self.positions[ticker] = _Lot(qty=qty, avg_cost=_d(cost / qty))
            else:
                new_qty = _d(lot.qty + qty)
                lot.avg_cost = _d((lot.qty * lot.avg_cost + cost) / new_qty)
                lot.qty = new_qty
            return Decimal("0")

        # SELL
        if lot is None or qty > lot.qty:
            held = lot.qty if lot else Decimal("0")
            raise LedgerError(
                f"{self.agent.value}: sell {qty} {ticker} but holds {held} (long-only, no short)"
            )
        proceeds = _d(qty * price - fees)
        realized = _d(qty * (price - lot.avg_cost) - fees)
        self.cash = _d(self.cash + proceeds)
        self.realized_pnl = _d(self.realized_pnl + realized)
        lot.qty = _d(lot.qty - qty)
        if lot.qty == 0:
            del self.positions[ticker]
        return realized

    # --------------------------------------------------------------- dividends
    def apply_dividend(self, ticker: str, amount: float | Decimal) -> None:
        """Credit a cash dividend to this book (doc 01: the ex-date holder). The caller
        (reconciliation/runner) has already attributed it to this agent; v1 uses a
        record-date approximation. Crediting a not-held ticker is an anomaly -> raise."""
        if ticker.upper() not in self.positions:
            raise LedgerError(
                f"{self.agent.value}: dividend for {ticker} but position not held "
                "(attribution bug?)"
            )
        self.cash = _d(self.cash + _d(amount))

    # ------------------------------------------------------------------- marks
    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update last_price for held names from a price map. Unpriced names keep their
        prior mark (or cost). Never invents a price (golden rule 8)."""
        upper = {t.upper(): p for t, p in prices.items()}
        for ticker, lot in self.positions.items():
            if ticker in upper:
                lot.last_price = _d(upper[ticker])

    def equity(self) -> Decimal:
        mv = sum((lot.market_value for lot in self.positions.values()), Decimal("0"))
        return _d(self.cash + mv)

    def weight(self, ticker: str) -> Decimal:
        eq = self.equity()
        if eq <= 0:
            return Decimal("0")
        lot = self.positions.get(ticker.upper())
        return _d(lot.market_value / eq) if lot else Decimal("0")

    # --------------------------------------------------------------- snapshots
    def snapshot(self, ts: datetime) -> BookSnapshot:
        """Read-model for persistence/eval (floats at the model boundary)."""
        return BookSnapshot(
            agent=self.agent,
            ts=ts,
            cash=float(self.cash),
            positions=[
                Position(
                    agent=self.agent,
                    ticker=ticker,
                    qty=float(lot.qty),
                    avg_cost=float(lot.avg_cost),
                    last_price=float(lot.last_price) if lot.last_price is not None else None,
                )
                for ticker, lot in self.positions.items()
            ],
        )
