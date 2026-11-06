"""Execution service (doc 06) — turns approved target weights into filled orders.

Sizing: qty = floor(|target_Δweight| · equity / price), whole shares v1; sells capped at
shares held (long-only, no shorting). Orders are marketable limits (last ± buffer bps,
TIF=DAY) — a bad-fill cap, not the fill mechanism. Lifecycle: idempotency check → log →
submit → poll → apply fill to the ledger → log fill. A reject is never auto-retried (only
network errors retry, inside the broker). Nothing here bypasses the RiskEngine — it only
acts on approved verdicts the runner passes in.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from trident import persistence
from trident.broker.base import Broker, BrokerError, BrokerRejectError
from trident.config import load_risk_limits
from trident.data import store
from trident.ledger.book import Book
from trident.models import (
    AgentName,
    BookSnapshot,
    HealthEvent,
    HealthSeverity,
    Order,
    OrderStatus,
    Side,
)


def size_order(
    agent: AgentName,
    decision_id: str | None,
    ticker: str,
    target_weight: float,
    book: BookSnapshot,
    price: float,
    limit_bps: float,
    asof: datetime,
) -> Order | None:
    """Pure: diff target weight vs the current book into a whole-share marketable-limit
    Order, or None if the rounded trade is zero. Sells are capped at shares held."""
    if price <= 0:
        return None
    equity = book.equity
    delta_w = target_weight - book.weight(ticker)
    qty = int(abs(delta_w) * equity / price)  # floor toward zero (whole shares v1)
    if delta_w < 0:  # selling: never exceed what we hold (no shorts)
        held = next((p.qty for p in book.positions if p.ticker == ticker), 0.0)
        qty = min(qty, int(held))
    if qty <= 0:
        return None
    side = Side.BUY if delta_w > 0 else Side.SELL
    buf = limit_bps / 10_000.0
    limit = round(price * (1 + buf), 2) if side == Side.BUY else round(price * (1 - buf), 2)
    return Order(
        decision_id=decision_id,
        agent=agent,
        client_order_id=Order.make_client_order_id(agent),
        ticker=ticker,
        side=side,
        qty=qty,
        limit_price=limit,
        created_at=asof,
    )


class ExecutionService:
    def __init__(
        self,
        broker: Broker,
        limit_bps: float | None = None,
        poll_interval_s: float = 3.0,
        poll_timeout_s: float = 60.0,
        dry_run: bool = False,
    ) -> None:
        self.broker = broker
        self.limit_bps = (
            limit_bps
            if limit_bps is not None
            else float(load_risk_limits().get("execution_limit_bps", 50))
        )
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self.dry_run = dry_run

    async def _health(self, severity: HealthSeverity, detail: str) -> None:
        await store.write_health_event(
            HealthEvent(ts=datetime.now(UTC), severity=severity, source="execution", detail=detail)
        )

    async def execute(self, orders: list[Order], book: Book) -> list:
        """Submit each order, poll for fills, apply fills to the (in-memory) ledger `book`.
        Persists orders and fills. Returns the Fills that landed. dry_run skips the broker."""
        fills = []
        # Idempotency (doc 06, crash-restart safety): the BROKER's live open orders are the
        # authoritative "is this already working" — not our DB intent. A logged-but-unsent
        # order must not block a retry, and a truly-live order must block a double-submit.
        open_at_broker: set[tuple[str, str, str]] = set()
        if not self.dry_run:
            try:
                open_at_broker = {
                    (o.agent.value, o.ticker, o.side.value)
                    for o in await self.broker.get_open_orders()
                }
            except BrokerError as e:
                await self._health(HealthSeverity.WARN, f"open-order check failed: {e}")

        for order in orders:
            if (order.agent.value, order.ticker, order.side.value) in open_at_broker:
                await self._health(
                    HealthSeverity.WARN,
                    f"skip {order.agent.value} {order.side.value} {order.ticker}: "
                    "live order already at broker (idempotency)",
                )
                continue

            await persistence.log_order(order)  # logged before it hits the broker

            if self.dry_run:
                await persistence.update_order(order.client_order_id, OrderStatus.PENDING.value)
                continue

            try:
                broker_id = await self.broker.submit_order(order)
            except BrokerRejectError as e:
                await persistence.update_order(order.client_order_id, OrderStatus.REJECTED.value)
                await self._health(HealthSeverity.WARN, f"reject {order.client_order_id}: {e}")
                continue
            except BrokerError as e:  # network/other after retries — skip, never guess
                await persistence.update_order(order.client_order_id, OrderStatus.REJECTED.value)
                await self._health(
                    HealthSeverity.WARN, f"broker error {order.client_order_id}: {e}"
                )
                continue

            await persistence.update_order(
                order.client_order_id, OrderStatus.SUBMITTED.value, broker_id
            )
            fill = await self._poll_fill(broker_id)
            if fill is None:
                try:
                    await self.broker.cancel_order(broker_id)
                except BrokerError:
                    pass
                await persistence.update_order(order.client_order_id, OrderStatus.EXPIRED.value)
                await self._health(
                    HealthSeverity.WARN,
                    f"no fill in {self.poll_timeout_s:.0f}s: {order.client_order_id}",
                )
                continue

            realized = book.apply_fill(fill)
            await persistence.log_fill(fill, realized)
            await persistence.update_order(order.client_order_id, OrderStatus.FILLED.value)
            fills.append(fill)
        return fills

    async def _poll_fill(self, broker_id: str):
        elapsed = 0.0
        while elapsed < self.poll_timeout_s:
            await asyncio.sleep(self.poll_interval_s)
            elapsed += self.poll_interval_s
            fills = await self.broker.get_order_fills(broker_id)
            if fills:
                return fills[0]
        return None
