"""Broker abstraction — the seam that makes paper -> live a config change.

Implementations:
  - AlpacaBroker (broker/alpaca.py): paper AND live, same class, flag flips endpoint.
  - BacktestBroker (evaluation/backtest.py): same interface over historical bars, so
    backtests run the IDENTICAL agent + risk + execution code (doc 01).

Rules:
  - Only ExecutionService calls submit/cancel. Agents never import this module.
  - All datetimes UTC. All quantities/prices floats here; DB stores NUMERIC.
"""

from __future__ import annotations

import abc
from datetime import datetime

from pydantic import BaseModel

from trident.models import Fill, Order


class AccountInfo(BaseModel):
    equity: float
    cash: float
    buying_power: float
    currency: str = "USD"


class BrokerPosition(BaseModel):
    ticker: str
    qty: float
    avg_entry_price: float
    market_value: float


class MarketClock(BaseModel):
    is_open: bool
    next_open: datetime
    next_close: datetime


class ActivityEvent(BaseModel):
    """Non-trade cash events we must ledger: dividends, fees."""

    kind: str  # "dividend" | "fee" | other passthrough
    ticker: str | None
    amount: float
    ts: datetime
    raw_id: str


class Broker(abc.ABC):
    """Minimal surface Trident needs. Add methods only when a phase requires them."""

    @abc.abstractmethod
    async def get_account(self) -> AccountInfo: ...

    @abc.abstractmethod
    async def get_positions(self) -> list[BrokerPosition]: ...

    @abc.abstractmethod
    async def get_clock(self) -> MarketClock: ...

    @abc.abstractmethod
    async def get_latest_price(self, ticker: str) -> float:
        """Last trade price. Raise BrokerDataError if unavailable — never guess (golden rule 8)."""

    @abc.abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit; return broker_order_id. Must pass order.client_order_id through unchanged
        (it is the ledger attribution key)."""

    @abc.abstractmethod
    async def cancel_order(self, broker_order_id: str) -> None: ...

    @abc.abstractmethod
    async def get_order_fills(self, broker_order_id: str) -> list[Fill]:
        """Fills so far for an order (may be partial)."""

    @abc.abstractmethod
    async def get_open_orders(self) -> list[Order]:
        """Open orders reconstructed with their client_order_id (idempotency checks)."""

    @abc.abstractmethod
    async def get_activities(self, since: datetime) -> list[ActivityEvent]:
        """Dividends/fees since timestamp — consumed nightly by the ledger."""


class BrokerError(RuntimeError): ...


class BrokerDataError(BrokerError): ...


class BrokerRejectError(BrokerError):
    """Order rejected by broker. Do NOT auto-retry rejects (doc 06)."""
