"""The runner (doc 01/06) — the one place the whole pipe is wired:

  load book -> build MarketContext -> agent.generate_decisions -> LOG decisions
  -> RiskEngine.review -> size approved verdicts -> ExecutionService.execute (fills ->
  ledger) -> mark -> save positions -> equity snapshot -> HWM ratchet -> kill-switch.

Agents never touch the broker or DB — this does. `dry_run=True` runs the full pipe but
ExecutionService skips the broker (orders logged PENDING, no fills). Golden rule 1's flow
lives here and nowhere else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from trident import persistence
from trident.agents.base import MarketContext, TradingAgent
from trident.broker.base import Broker, BrokerDataError
from trident.config import load_universe
from trident.data import market, store
from trident.execution.service import ExecutionService, size_order
from trident.models import (
    AgentName,
    AgentStatus,
    Decision,
    HealthEvent,
    HealthSeverity,
    Order,
    Side,
)
from trident.risk.engine import RiskEngine, RiskInputs


def _now_utc() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunResult:
    agent: AgentName
    asof: datetime
    status: str  # ran | skipped_halted | skipped_frozen | flattened
    n_decisions: int = 0
    n_orders: int = 0
    n_fills: int = 0
    equity: float = 0.0
    halted: bool = False
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------- shared inputs
async def _global_freeze() -> bool:
    """True if a SYSTEMIC CRITICAL health event fired in the last 24h (doc 06 rule 10:
    reconciliation break, repeated data failures). A single agent's kill-switch (rule 9)
    halts only THAT agent via its status — it must not freeze the others, so kill_switch
    events are excluded here."""
    from sqlalchemy import text

    async with store.get_engine().connect() as conn:
        n = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM health_events WHERE severity='critical' "
                    "AND cleared=false AND source <> 'kill_switch' "
                    "AND ts > now() - interval '24 hours'"
                )
            )
        ).scalar_one()
    return n > 0


async def _live_prices(broker: Broker, tickers: list[str]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for t in tickers:
        try:
            prices[t] = await broker.get_latest_price(t)
        except BrokerDataError:
            continue  # never guess a price (golden rule 8) — this ticker drops out
    return prices


def _prev_close_and_adv(tickers: list[str], asof: datetime) -> tuple[dict, dict]:
    """Previous-session close (bad-print band) and 21d avg dollar volume (liquidity) from
    the cached daily bars — strictly ≤ asof (no look-ahead)."""
    prev_close: dict[str, float] = {}
    adv: dict[str, float] = {}
    hist = market.get_history(tickers, asof.date() - timedelta(days=45), asof.date())
    for t in tickers:
        g = hist[hist["ticker"] == t].sort_values("date")
        # Exclude asof's OWN date: "previous close" must be a completed prior session, never
        # a partial same-day bar (which would make the bad-print band self-compare).
        g = g[g["date"].dt.date < asof.date()]
        if g.empty:
            continue
        prev_close[t] = float(g["close"].iloc[-1])
        dv = (g["close"] * g["volume"]).tail(21)
        adv[t] = float(dv.mean()) if len(dv) else 0.0
    return prev_close, adv


async def _risk_inputs(
    broker: Broker, tickers: list[str], asof: datetime, last_price: dict[str, float]
) -> RiskInputs:
    clock = await broker.get_clock()
    minutes_to_close = (clock.next_close - asof).total_seconds() / 60.0 if clock.is_open else -1.0
    prev_close, adv = _prev_close_and_adv(tickers, asof)
    return RiskInputs(
        last_price=last_price,
        prev_close=prev_close,
        adv_dollars_21d=adv,
        market_open=clock.is_open,
        minutes_to_close=minutes_to_close,
        global_freeze=await _global_freeze(),
    )


async def _health(severity: HealthSeverity, source: str, detail: str) -> None:
    await store.write_health_event(
        HealthEvent(ts=_now_utc(), severity=severity, source=source, detail=detail)
    )


def _extras_are_data(extras: dict) -> bool:
    """extras must be inert DATA — never a Broker, DB session, or callable. Strict dumps (no
    `default=`) so anything non-JSON is rejected rather than coerced to a string, and
    `allow_nan=False` so a pandas NaN/Inf leaking out of aggregate() is a refused run rather
    than a non-standard token silently baked into the inputs digest."""
    try:
        json.dumps(extras, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


# ------------------------------------------------------------------- run_agent
async def run_agent(
    agent: TradingAgent,
    broker: Broker,
    dry_run: bool = False,
    asof: datetime | None = None,
    extras: dict | None = None,
) -> RunResult:
    asof = asof or _now_utc()
    name = agent.name
    result = RunResult(agent=name, asof=asof, status="ran")

    status = await persistence.get_status(name)
    if status == AgentStatus.HALTED:
        # Halted: flatten once (doc 06 #9), refuse new decisions until manual reactivation.
        book = await persistence.load_book(name)
        execution = ExecutionService(broker, dry_run=dry_run)
        await _flatten(name, book, broker, execution, asof)
        result.status = "flattened"
        result.halted = True
        return result
    if status != AgentStatus.ACTIVE:
        result.status = "skipped_disabled"
        return result

    book = await persistence.load_book(name)
    universe = load_universe()

    # Mark held positions with live prices so the book the agent sees has a real equity.
    held = [p.ticker for p in book.snapshot(asof).positions]
    book.mark_to_market(await _live_prices(broker, held))

    extras = extras or {}  # SENTINEL: {"signals", "holding_days"}; ORACLE later. QUANT: {}.
    if not _extras_are_data(extras):
        # extras is a channel from the runner straight into agent code. It carries DATA only —
        # a live Broker/session/callable in here would let an agent reach the broker without
        # RiskEngine (golden rule 1). Refuse the run rather than trust it (golden rule 8).
        # WARN, not CRITICAL: this run is already refused (no orders can escape), and CRITICAL
        # would trip _global_freeze() and halt the other two agents over one agent's wiring bug.
        await _health(
            HealthSeverity.WARN,
            f"runner.{name.value}",
            "extras contains non-JSON-serializable values; refusing to run",
        )
        result.status = "skipped_bad_extras"
        return result

    snapshot = book.snapshot(asof)
    ctx = MarketContext(
        asof=asof, universe=universe, book=snapshot, prices=_Prices(asof), extras=extras
    )
    # golden rule 2: the digest pins EXACTLY what the agent saw. SENTINEL's inputs arrive only
    # via extras and their source table is rewritten on every aggregate() run (delete-then-
    # insert), so without this a re-score silently rewrites the inputs behind an old decision
    # and the drift is undetectable.
    digest = Decision.digest(
        {
            "asof": asof,
            "universe": universe,
            "book": snapshot.model_dump(),
            "extras": extras,
        }
    )
    decisions = await agent.generate_decisions(ctx)
    for d in decisions:  # golden rule 2: log EVERY decision BEFORE any order
        if not d.inputs_digest:
            d.inputs_digest = digest
        await persistence.log_decision(d)
    result.n_decisions = len(decisions)
    # Target weights are per-ticker for the whole book; a well-formed agent emits at most one
    # per ticker. Guard a malformed agent: keep the last decision per ticker (else two would
    # each size the full delta -> a double order).
    decisions = list({d.ticker: d for d in decisions}.values())
    if not decisions:
        await _finalize(name, book, broker, asof, result)
        return result

    tickers = sorted({d.ticker for d in decisions})
    last_price = await _live_prices(broker, tickers)
    inputs = await _risk_inputs(broker, tickers, asof, last_price)
    verdicts = RiskEngine().review(decisions, book.snapshot(asof), inputs)

    by_decision = {d.id: d for d in decisions}
    orders: list[Order] = []
    snap = book.snapshot(asof)
    for v in verdicts:
        d = by_decision[v.decision_id]
        if not v.approved:
            await _health(
                HealthSeverity.INFO, "runner", f"{name.value} reject {d.ticker}: {v.reasons}"
            )
            continue
        price = last_price.get(d.ticker)
        if price is None:
            continue
        order = size_order(
            name,
            d.id,
            d.ticker,
            v.adjusted_target_weight or 0.0,
            snap,
            price,
            ExecutionService(broker).limit_bps,
            asof,
        )
        if order is not None:
            order.risk_notes = v.reasons
            orders.append(order)
    result.n_orders = len(orders)

    execution = ExecutionService(broker, dry_run=dry_run)
    fills = await execution.execute(orders, book)
    result.n_fills = len(fills)

    await _finalize(name, book, broker, asof, result)
    return result


async def _finalize(
    name: AgentName,
    book,
    broker: Broker,
    asof: datetime,
    result: RunResult,
    prices: dict[str, float] | None = None,
) -> None:
    """Mark, persist positions + equity snapshot, ratchet HWM, evaluate the kill-switch.

    `prices` lets a caller supply already-fetched marks (the daily snapshot job fetches once and
    verifies full coverage before calling, so it can refuse on a missing price rather than
    stale-mark). When None, fetches live prices as the trading path always has.
    """
    held = [p.ticker for p in book.snapshot(asof).positions]
    book.mark_to_market(await _live_prices(broker, held) if prices is None else prices)
    equity = float(book.equity())
    result.equity = equity
    await persistence.save_positions(book, asof)
    # Ordering is deliberately snapshot-BEFORE-killswitch: a crash between the two leaves the
    # equity recorded but the HWM un-ratcheted, which only makes the NEXT session's kill-switch
    # strictly more likely to fire on a persistent drawdown — it errs safe.
    await persistence.write_equity_snapshot(name, asof, equity, float(book.cash))
    if await evaluate_kill_switch(name, equity, asof):
        result.halted = True
        result.status = "halted"


async def evaluate_kill_switch(agent: AgentName, equity: float, asof: datetime) -> bool:
    """Ratchet the HWM to `equity`, then fire the drawdown kill-switch (doc 06 #9) if the
    agent has fallen below (1 − kill_dd)·HWM: set status=halted + CRITICAL health event.
    Reactivation is manual only. Returns whether it halted this cycle."""
    hwm = await persistence.update_high_water_mark(agent, equity)
    if not RiskEngine().kill_switch_should_fire(equity, float(hwm)):
        return False
    await persistence.set_status(agent, AgentStatus.HALTED)
    await _health(
        HealthSeverity.CRITICAL,
        "kill_switch",
        f"{agent.value} halted: equity {equity:.2f} < 85% of HWM {float(hwm):.2f}; "
        "flatten next session, manual reactivation only",
    )
    return True


async def _flatten(
    name: AgentName, book, broker: Broker, execution: ExecutionService, asof: datetime
) -> None:
    """Sell all held positions (kill-switch aftermath). Best-effort; leaves status halted."""
    snap = book.snapshot(asof)
    prices = await _live_prices(broker, [p.ticker for p in snap.positions])
    orders = []
    for p in snap.positions:
        if p.qty <= 0 or p.ticker not in prices:
            continue
        orders.append(
            size_order(name, None, p.ticker, 0.0, snap, prices[p.ticker], execution.limit_bps, asof)
        )
    await execution.execute([o for o in orders if o is not None], book)
    await persistence.save_positions(book, asof)


# ---------------------------------------------------------------- BENCH (SPY)
async def run_bench_spy_buy(broker: Broker, dry_run: bool = False, asof: datetime | None = None):
    """One-time: put BENCH's cash into SPY, then never touch it (doc 00). Idempotent —
    a no-op once SPY is held. BENCH is the benchmark, so it bypasses the RiskEngine."""
    asof = asof or _now_utc()
    book = await persistence.load_book(AgentName.BENCH)
    if any(p.ticker == "SPY" for p in book.snapshot(asof).positions):
        return RunResult(agent=AgentName.BENCH, asof=asof, status="already_invested")
    price = (await _live_prices(broker, ["SPY"])).get("SPY")
    if price is None:
        await _health(HealthSeverity.WARN, "bench", "no SPY price; BENCH buy deferred")
        return RunResult(agent=AgentName.BENCH, asof=asof, status="no_price")
    execution = ExecutionService(broker, dry_run=dry_run)
    buf = 1 + execution.limit_bps / 10_000.0
    qty = int(float(book.cash) / (price * buf))  # spend ~all cash on whole shares
    result = RunResult(agent=AgentName.BENCH, asof=asof, status="ran", n_orders=1 if qty else 0)
    if qty <= 0:
        return result
    order = Order(
        decision_id=None,
        agent=AgentName.BENCH,
        client_order_id=Order.make_client_order_id(AgentName.BENCH),
        ticker="SPY",
        side=Side.BUY,
        qty=qty,
        limit_price=round(price * buf, 2),
        created_at=asof,
    )
    fills = await execution.execute([order], book)
    result.n_fills = len(fills)
    await _finalize(AgentName.BENCH, book, broker, asof, result)
    return result


# ------------------------------------------------------------ reconciliation
@dataclass
class ReconResult:
    ok: bool
    mismatches: list[str] = field(default_factory=list)


async def reconcile(broker: Broker, share_tol: float = 1e-6, cash_tol: float = 1.0) -> ReconResult:
    """Reconcile the ledger against broker truth (doc 06:30, doc 01:55). TWO checks:

      1. shares:  broker positions == Σ ledger positions per ticker
      2. cash:    broker account cash == Σ ledger cash (± $1)

    Given (1), cash equality is the mark-independent form of "Σ agent equity == broker
    equity" — and it catches any unledgered cash event (e.g. a dividend before broker
    activities are wired) that share-only reconciliation would miss silently.

    This is also the exactly-once backstop: a crash between submit and log_fill can leave a
    filled order unledgered; the next reconcile flags the resulting drift. Any breach ->
    CRITICAL health event -> global freeze until manually cleared.
    """
    mismatches: list[str] = []
    broker_pos = {p.ticker: p.qty for p in await broker.get_positions()}
    ledger_pos = await persistence.sum_ledger_positions()
    for t in set(broker_pos) | set(ledger_pos):
        if abs(broker_pos.get(t, 0.0) - ledger_pos.get(t, 0.0)) > share_tol:
            mismatches.append(
                f"shares {t}: broker={broker_pos.get(t, 0.0)} ledger={ledger_pos.get(t, 0.0)}"
            )

    broker_cash = (await broker.get_account()).cash
    ledger_cash = float(await persistence.sum_ledger_cash())
    if abs(broker_cash - ledger_cash) > cash_tol:
        mismatches.append(f"cash: broker={broker_cash:.2f} ledger={ledger_cash:.2f}")

    ok = not mismatches
    if not ok:
        await _health(
            HealthSeverity.CRITICAL,
            "reconciliation",
            "ledger/broker mismatch (halting all trading): " + "; ".join(mismatches[:10]),
        )
    return ReconResult(ok=ok, mismatches=mismatches)


# PriceHistory adapter for MarketContext.prices — anti-look-ahead: bars strictly ≤ asof.
class _Prices:
    def __init__(self, asof: datetime) -> None:
        self.asof = asof

    def get(self, tickers: list[str], lookback_days: int):
        return market.get_history(
            tickers, self.asof.date() - timedelta(days=lookback_days), self.asof.date()
        )
