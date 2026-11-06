"""Phase 0 acceptance: prove paper-broker connectivity end to end.

Run: uv run python scripts/smoke_broker.py
Expected: prints account equity (~$100k fresh paper), buys 1 AAPL via marketable
limit, polls the fill, sells it back, prints both fills. Finish TODOs in
broker/alpaca.py first if anything raises NotImplementedError.
"""

import asyncio
from datetime import UTC, datetime

from trident.broker.alpaca import AlpacaBroker
from trident.models import AgentName, Order, Side


async def main() -> None:
    broker = AlpacaBroker()
    assert broker.paper, "SAFETY: smoke test refuses to run against live."
    acct = await broker.get_account()
    print(f"account equity=${acct.equity:,.2f} cash=${acct.cash:,.2f}")
    clock = await broker.get_clock()
    print(f"market open: {clock.is_open} (next open {clock.next_open}, close {clock.next_close})")

    price = await broker.get_latest_price("AAPL")
    for side, limit in ((Side.BUY, price * 1.001), (Side.SELL, price * 0.999)):
        order = Order(
            decision_id=None,
            agent=AgentName.QUANT,
            client_order_id=Order.make_client_order_id(AgentName.QUANT),
            ticker="AAPL",
            side=side,
            qty=1,
            limit_price=limit,
            created_at=datetime.now(UTC),
        )
        broker_id = await broker.submit_order(order)
        print(f"submitted {side.value} 1 AAPL @ {limit:.2f} -> {broker_id}")
        for _ in range(20):
            await asyncio.sleep(3)
            fills = await broker.get_order_fills(broker_id)
            if fills:
                f = fills[0]
                print(f"filled: {f.side.value} {f.qty} {f.ticker} @ {f.price}")
                break
        else:
            print("no fill in 60s (market closed?) — canceling")
            await broker.cancel_order(broker_id)
            break


if __name__ == "__main__":
    asyncio.run(main())
