"""Event-driven backtester (doc 01/06, backtest-protocol).

Parity (pre-registered, doc 01): the backtest reuses the IDENTICAL `RiskEngine`,
`size_order`, and `Book` (ledger) math as paper. `ExecutionService`'s I/O (DB logging,
idempotency, polling) is excluded — the driver runs a thin submit→fill loop against a
`BacktestBroker`. Fill model = session-T open ± slippage bps.

Historical price access uses `_LaggedPrices`, which returns bars strictly before session T.
The book snapshot uses T-open marks for holdings. Fills are at T's open and closing equity
is marked at T's close. `test_backtest.py` checks that price history excludes session T.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from trident.agents.base import MarketContext, TradingAgent
from trident.broker.base import (
    AccountInfo,
    ActivityEvent,
    Broker,
    BrokerDataError,
    BrokerPosition,
    BrokerRejectError,
    MarketClock,
)
from trident.config import load_risk_limits
from trident.data import calendar as trading_calendar
from trident.data import market
from trident.execution.service import size_order
from trident.ledger.book import Book
from trident.models import Fill, Order, Side
from trident.risk.engine import RiskEngine, RiskInputs

WARMUP_DAYS = 500  # calendar days of history loaded before `start` (~340 sessions) so the
# longest baseline/feature lookback (12-1 momentum needs 252 sessions; SMA200 needs 200) is
# warm from session 1.


# --------------------------------------------------------------- price access
class _Bars:
    """Universe bars as one tidy frame with a sorted DatetimeIndex on `date`, plus O(1)
    per-(ticker, date) open/close lookups."""

    def __init__(self, frame: pd.DataFrame) -> None:
        f = frame.copy()
        f["date"] = pd.to_datetime(f["date"])
        self.frame = f.set_index("date").sort_index()
        self._by = {
            t: g.set_index(pd.to_datetime(g["date"]))[["open", "close", "volume"]]
            for t, g in frame.groupby("ticker")
        }

    def open(self, ticker: str, d: date) -> float | None:
        df = self._by.get(ticker)
        ts = pd.Timestamp(d)
        return float(df.loc[ts, "open"]) if df is not None and ts in df.index else None

    def close(self, ticker: str, d: date) -> float | None:
        df = self._by.get(ticker)
        ts = pd.Timestamp(d)
        return float(df.loc[ts, "close"]) if df is not None and ts in df.index else None


class _LaggedPrices:
    """`PriceHistory` for a backtest session: returns bars dated STRICTLY < asof (≤ the
    previous session), so features can never see session T. Anti-look-ahead seam."""

    def __init__(self, bars: _Bars, asof: date) -> None:
        self._bars = bars
        self.asof = asof
        self._hi = pd.Timestamp(trading_calendar.previous_session(asof))  # last date < asof

    def get(self, tickers: list[str], lookback_days: int) -> pd.DataFrame:
        lo = pd.Timestamp(self.asof - timedelta(days=lookback_days))
        sub = self._bars.frame.loc[lo : self._hi]  # inclusive of _hi, which is < asof
        sub = sub[sub["ticker"].isin([t.upper() for t in tickers])]
        return sub.reset_index()


# ------------------------------------------------------------- backtest broker
class BacktestBroker(Broker):
    """Broker over historical bars. Fills synchronously at session-T open ± slippage;
    tracks its own account so the interface is complete. `set_asof` is called each step."""

    def __init__(self, bars: _Bars, start_cash: float, slippage_bps: float = 5.0) -> None:
        self._bars = bars
        self.slippage_bps = slippage_bps
        self.asof: date | None = None
        self._cash = start_cash
        self._positions: dict[str, list[float]] = {}  # ticker -> [qty, avg_entry]
        self._fills: dict[str, list[Fill]] = {}
        self._seq = 0

    def set_asof(self, d: date) -> None:
        self.asof = d

    async def get_latest_price(self, ticker: str) -> float:
        px = self._bars.open(ticker, self.asof)
        if px is None:
            raise BrokerDataError(f"no {ticker} open on {self.asof}")
        return px

    async def submit_order(self, order: Order) -> str:
        px = self._bars.open(order.ticker, self.asof)
        if px is None:
            raise BrokerRejectError(f"no {order.ticker} bar on {self.asof}")
        slip = self.slippage_bps / 10_000.0
        fill_px = round(px * (1 + slip), 4) if order.side == Side.BUY else round(px * (1 - slip), 4)
        signed = order.qty if order.side == Side.BUY else -order.qty
        pos = self._positions.setdefault(order.ticker, [0.0, 0.0])
        if order.side == Side.BUY:
            pos[1] = (pos[0] * pos[1] + order.qty * fill_px) / (pos[0] + order.qty)
        pos[0] += signed
        self._cash -= signed * fill_px
        if abs(pos[0]) < 1e-9:
            self._positions.pop(order.ticker, None)
        bid = str(self._seq)
        self._seq += 1
        self._fills[bid] = [
            Fill(
                order_id=order.client_order_id,
                agent=order.agent,
                ticker=order.ticker,
                side=order.side,
                qty=order.qty,
                price=fill_px,
                ts=datetime.combine(self.asof, datetime.min.time(), tzinfo=UTC),
            )
        ]
        return bid

    async def get_order_fills(self, broker_order_id: str) -> list[Fill]:
        return self._fills.get(broker_order_id, [])

    async def get_clock(self) -> MarketClock:
        now = datetime.combine(self.asof, datetime.min.time(), tzinfo=UTC)
        return MarketClock(
            is_open=True, next_open=now, next_close=now + timedelta(hours=6, minutes=30)
        )

    async def get_account(self) -> AccountInfo:
        eq = self._cash + sum(
            q * (self._bars.close(t, self.asof) or e) for t, (q, e) in self._positions.items()
        )
        return AccountInfo(equity=eq, cash=self._cash, buying_power=self._cash)

    async def get_positions(self) -> list[BrokerPosition]:
        out = []
        for t, (q, e) in self._positions.items():
            mark = self._bars.close(t, self.asof) or e
            out.append(BrokerPosition(ticker=t, qty=q, avg_entry_price=e, market_value=q * mark))
        return out

    async def get_open_orders(self) -> list[Order]:
        return []  # fills are synchronous — nothing ever stays open

    async def cancel_order(self, broker_order_id: str) -> None:
        return None

    async def get_activities(self, since: datetime) -> list[ActivityEvent]:
        return []  # dividends deferred (consistent with live; doc-clocked before Phase 8)


# --------------------------------------------------------------------- driver
@dataclass
class BacktestResult:
    name: str
    equity: pd.Series  # indexed by session date, marked at close
    traded_value: pd.Series  # |fill notional| per session (for turnover)
    trades: int = 0
    sessions: list[date] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)


def _risk_inputs(
    bars: _Bars, asof: date, tickers: list[str], last_price: dict[str, float]
) -> RiskInputs:
    """Built from REAL bars (never stubbed) so the liquidity/price-sanity checks fire exactly
    as in paper: prev_close = T-1 close; ADV = 21d mean close×volume over bars ≤ T-1."""
    prev = trading_calendar.previous_session(asof)
    lo = pd.Timestamp(prev - timedelta(days=45))
    hist = bars.frame.loc[lo : pd.Timestamp(prev)]
    prev_close: dict[str, float] = {}
    adv: dict[str, float] = {}
    for t in tickers:
        g = hist[hist["ticker"] == t]
        if g.empty:
            continue
        prev_close[t] = float(g["close"].iloc[-1])
        dv = (g["close"] * g["volume"]).tail(21)
        adv[t] = float(dv.mean()) if len(dv) else 0.0
    return RiskInputs(
        last_price=last_price,
        prev_close=prev_close,
        adv_dollars_21d=adv,
        market_open=True,
        minutes_to_close=390.0,
    )


async def run_backtest(
    agent: TradingAgent,
    tickers: list[str],
    start: str | date,
    end: str | date,
    start_cash: float = 30_000.0,
    slippage_bps: float = 5.0,
    name: str | None = None,
    bars: pd.DataFrame | None = None,
) -> BacktestResult:
    """Drive `agent` over sessions [start, end], reusing RiskEngine + size_order + Book.
    Supply bars with date, ticker, OHLC and volume columns for an offline run.
    Otherwise load WARMUP_DAYS of history so lookbacks are warm from session 1."""
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()
    tickers = [t.upper() for t in tickers]
    raw = (
        market.get_history(tickers, start_d - timedelta(days=WARMUP_DAYS), end_d)
        if bars is None
        else bars
    )
    if raw.empty:
        raise ValueError("no bars for backtest window")
    bars = _Bars(raw)
    sessions = trading_calendar.sessions(start_d, end_d)

    broker = BacktestBroker(bars, start_cash, slippage_bps)
    book = Book.new(agent.name, start_cash)
    engine = RiskEngine()
    limit_bps = float(load_risk_limits().get("execution_limit_bps", 50))

    equity, traded = {}, {}
    fills: list[Fill] = []
    for d in sessions:
        broker.set_asof(d)
        asof_dt = datetime.combine(d, datetime.min.time(), tzinfo=UTC)
        # mark held positions at T-open (the tradeable price), then snapshot for the agent
        held = list(book.positions.keys())
        book.mark_to_market({t: p for t in held if (p := bars.open(t, d)) is not None})
        ctx = MarketContext(
            asof=asof_dt,
            universe=tickers,
            book=book.snapshot(asof_dt),
            prices=_LaggedPrices(bars, d),
        )
        decisions = await agent.generate_decisions(ctx)
        decisions = list({dec.ticker: dec for dec in decisions}.values())  # dedupe per ticker

        session_traded = 0.0
        if decisions:
            dtk = sorted({dec.ticker for dec in decisions})
            last_price = {t: p for t in dtk if (p := bars.open(t, d)) is not None}
            inputs = _risk_inputs(bars, d, dtk, last_price)
            snap = book.snapshot(asof_dt)
            verdicts = engine.review(decisions, snap, inputs)
            by_id = {dec.id: dec for dec in decisions}
            for v in verdicts:
                dec = by_id[v.decision_id]
                if not v.approved or dec.ticker not in last_price:
                    continue
                order = size_order(
                    agent.name,
                    dec.id,
                    dec.ticker,
                    v.adjusted_target_weight or 0.0,
                    snap,
                    last_price[dec.ticker],
                    limit_bps,
                    asof_dt,
                )
                if order is None:
                    continue
                bid = await broker.submit_order(order)
                for fill in await broker.get_order_fills(bid):
                    book.apply_fill(fill)
                    session_traded += fill.qty * fill.price
                    fills.append(fill)

        # mark at T-close for the daily equity point
        held = list(book.positions.keys())
        book.mark_to_market({t: c for t in held if (c := bars.close(t, d)) is not None})
        equity[pd.Timestamp(d)] = float(book.equity())
        traded[pd.Timestamp(d)] = session_traded

    return BacktestResult(
        name=name or agent.describe(),
        equity=pd.Series(equity).sort_index(),
        traded_value=pd.Series(traded).sort_index(),
        trades=len(fills),
        sessions=sessions,
        fills=fills,
    )


def buy_and_hold_equity(
    ticker: str, start: str | date, end: str | date, start_cash: float
) -> pd.Series:
    """SPY (or any ticker) buy-and-hold benchmark: whole shares bought at the first session's
    open, marked at each session's close. The BENCH comparison for every backtest."""
    start_d, end_d = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    bars = _Bars(market.get_history([ticker], start_d - timedelta(days=10), end_d))
    sessions = trading_calendar.sessions(start_d, end_d)
    first_open = bars.open(ticker, sessions[0])
    qty = int(start_cash / first_open) if first_open else 0
    cash = start_cash - qty * (first_open or 0.0)
    eq = {}
    last_close = first_open or 0.0
    for d in sessions:
        c = bars.close(ticker, d)
        if c is not None:  # carry the mark forward on a missing bar (calendar-vs-bar edge)
            last_close = c
        eq[pd.Timestamp(d)] = cash + qty * last_close
    return pd.Series(eq).sort_index()
