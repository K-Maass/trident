"""AlpacaBroker — paper and live behind one flag. Finish TODOs in Phase 0.

alpaca-py notes (verify against current docs — API surface occasionally shifts):
  TradingClient(api_key, secret_key, paper=True)
  StockHistoricalDataClient(api_key, secret_key)  # free tier serves IEX feed
  Orders: LimitOrderRequest(symbol, qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
          limit_price=..., client_order_id=...) -> trading.submit_order(order_data=req)
  alpaca-py is sync; we wrap calls in asyncio.to_thread to stay honest in async code.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest

from trident.broker.base import (
    AccountInfo,
    ActivityEvent,
    Broker,
    BrokerDataError,
    BrokerError,
    BrokerPosition,
    BrokerRejectError,
    MarketClock,
)
from trident.config import get_settings
from trident.models import AgentName, Fill, Order, OrderStatus, Side

_AGENT_VALUES = {a.value for a in AgentName}  # valid client_order_id prefixes


def _to_side(alpaca_side: OrderSide | str) -> Side:
    """Map an alpaca-py OrderSide to our Side by VALUE — NOT str(enum).

    str(OrderSide.BUY) == 'OrderSide.BUY', so the old `str(o.side).endswith("buy")`
    was ALWAYS False and mislabeled every buy fill as a sell (would misbook the ledger).
    OrderSide is a str-enum whose .value is "buy"/"sell"; compare that, case-folded.
    """
    val = str(getattr(alpaca_side, "value", alpaca_side)).lower()
    if val == "buy":
        return Side.BUY
    if val == "sell":
        return Side.SELL
    raise BrokerError(f"unknown order side from broker: {alpaca_side!r}")


class AlpacaBroker(Broker):
    def __init__(self) -> None:
        key, secret, paper = get_settings().broker_credentials()
        self._trading = TradingClient(key, secret, paper=paper)
        self._data = StockHistoricalDataClient(key, secret)
        self.paper = paper

    # ------------------------------------------------------------ account
    async def get_account(self) -> AccountInfo:
        acct = await asyncio.to_thread(self._trading.get_account)
        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
        )

    async def get_positions(self) -> list[BrokerPosition]:
        raw = await asyncio.to_thread(self._trading.get_all_positions)
        return [
            BrokerPosition(
                ticker=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
            )
            for p in raw
        ]

    async def get_clock(self) -> MarketClock:
        c = await asyncio.to_thread(self._trading.get_clock)
        return MarketClock(is_open=c.is_open, next_open=c.next_open, next_close=c.next_close)

    async def get_latest_price(self, ticker: str) -> float:
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trades = await asyncio.to_thread(self._data.get_stock_latest_trade, req)
            return float(trades[ticker].price)
        except Exception as e:  # noqa: BLE001 — map all data failures to our type
            raise BrokerDataError(f"no latest price for {ticker}: {e}") from e

    # ------------------------------------------------------------- orders
    async def submit_order(self, order: Order) -> str:
        side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL
        req: MarketOrderRequest | LimitOrderRequest
        if order.limit_price:
            req = LimitOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(order.limit_price, 2),
                client_order_id=order.client_order_id,
            )
        else:
            req = MarketOrderRequest(
                symbol=order.ticker,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                client_order_id=order.client_order_id,
            )
        try:
            resp = await asyncio.to_thread(self._trading.submit_order, order_data=req)
        except Exception as e:  # broker-side reject
            raise BrokerRejectError(str(e)) from e
        return str(resp.id)

    async def cancel_order(self, broker_order_id: str) -> None:
        await asyncio.to_thread(self._trading.cancel_order_by_id, broker_order_id)

    async def get_order_fills(self, broker_order_id: str) -> list[Fill]:
        o = await asyncio.to_thread(self._trading.get_order_by_id, broker_order_id)
        filled_qty = float(o.filled_qty or 0)
        if filled_qty <= 0:
            return []
        # Agent is the client_order_id prefix ("{agent}-{uuid}"); one aggregate Fill per
        # poll is sufficient for v1 (avg fill price). Per-execution granularity via
        # activities in P7. Order.make_client_order_id guarantees this prefix.
        agent = o.client_order_id.split("-", 1)[0]
        return [
            Fill(
                order_id=o.client_order_id,
                agent=agent,  # type: ignore[arg-type]
                ticker=o.symbol,
                side=_to_side(o.side),
                qty=filled_qty,
                price=float(o.filled_avg_price or 0),
                ts=(o.filled_at or datetime.now(UTC)),
            )
        ]

    async def get_open_orders(self) -> list[Order]:
        req = GetOrdersRequest(status="open")
        raw = await asyncio.to_thread(self._trading.get_orders, filter=req)
        out: list[Order] = []
        for o in raw:
            prefix = (o.client_order_id or "").split("-", 1)[0]
            if prefix not in _AGENT_VALUES:
                continue  # foreign/manual order in the shared account — not one of ours
            out.append(
                Order(
                    decision_id=None,
                    agent=prefix,  # type: ignore[arg-type]
                    client_order_id=o.client_order_id,
                    ticker=o.symbol,
                    side=_to_side(o.side),
                    qty=float(o.qty),
                    limit_price=float(o.limit_price) if o.limit_price else None,
                    status=OrderStatus.SUBMITTED,
                    broker_order_id=str(o.id),
                    created_at=o.created_at,
                )
            )
        return out

    async def get_activities(self, since: datetime) -> list[ActivityEvent]:
        # TODO(P2): use trading client account activities endpoint, filter DIV/FEE,
        # map to ActivityEvent. Needed before first ex-dividend date in the universe.
        raise NotImplementedError("Phase 2: dividends/fees via account activities API")
