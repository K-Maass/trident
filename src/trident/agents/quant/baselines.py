"""The four rule baselines (doc 03) — the bar the ML must beat AFTER costs.

Each is a pure `TradingAgent` emitting target-weight Decisions from `ctx.prices`. Sizing is
**pre-registered** here so nobody "fixes" a baseline after seeing the ML lose to it:

  - Longs are equal-weighted at `min(1/N, max_position_weight)` per name (max_position_weight
    from risk_limits.yaml = 0.10), so a full basket is ~100% invested and the RiskEngine never
    has to clamp them.
  - A name currently held but no longer a long gets an explicit 0-weight Decision (exit to
    cash), per the agent contract. Emitting no decisions means "hold the book unchanged".

Baselines are fixed a priori and never tuned (doc 03).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trident.agents.base import MarketContext, TradingAgent
from trident.config import required_risk_limit
from trident.data import calendar as trading_calendar
from trident.models import AgentName, Decision

MAX_POS = required_risk_limit("max_position_weight")


# ------------------------------------------------------------------ helpers
def _close_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Tidy long bars -> a (date × ticker) close matrix, sorted by date."""
    if prices_df.empty:
        return pd.DataFrame()
    return prices_df.pivot_table(index="date", columns="ticker", values="close").sort_index()


def _held(ctx: MarketContext) -> set[str]:
    return {p.ticker for p in ctx.book.positions}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    # loss==0 (all up over the window) -> RSI 100 (overbought), not NaN
    return (100 - 100 / (1 + rs)).fillna(100.0)


def _decisions(
    name: AgentName, longs: list[str], held: set[str], ctx: MarketContext, reason: str
) -> list[Decision]:
    n = len(longs)
    w = min(1.0 / n, MAX_POS) if n else 0.0
    out = [
        Decision(agent=name, asof=ctx.asof, ticker=t, target_weight=w, rationale={"rule": reason})
        for t in longs
    ]
    for t in held - set(longs):  # exit names that dropped out of the basket
        out.append(
            Decision(
                agent=name,
                asof=ctx.asof,
                ticker=t,
                target_weight=0.0,
                rationale={"rule": reason, "action": "exit"},
            )
        )
    return out


def _first_session_of_month(asof) -> bool:
    prev = trading_calendar.previous_session(asof.date() if hasattr(asof, "date") else asof)
    return prev.month != (asof.date() if hasattr(asof, "date") else asof).month


# --------------------------------------------------------------- 1. MA-cross
class MACrossBaseline(TradingAgent):
    """Long a name while its 20d SMA > 100d SMA; equal-weight all such longs; else cash.
    Daily rebalance."""

    name = AgentName.QUANT
    LOOKBACK = 220

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        close = _close_matrix(ctx.prices.get(ctx.universe, self.LOOKBACK))
        if len(close) < 100:
            return []
        sma20, sma100 = close.rolling(20).mean().iloc[-1], close.rolling(100).mean().iloc[-1]
        longs = [t for t in close.columns if sma20[t] > sma100[t]]
        return _decisions(self.name, longs, _held(ctx), ctx, "ma_cross_20_100")


# ------------------------------------------------------- 2. 12-1 momentum
class MomentumBaseline(TradingAgent):
    """Monthly: long the top-8 by trailing 12-month return SKIPPING the last month
    (~252d ago vs ~21d ago). Equal-weight. Holds unchanged between month starts.

    `force_rebalance` (default False = the audited monthly behavior) rebalances regardless of
    the calendar — used ONLY for the one-time initial entry at deploy, so QUANT establishes its
    basket immediately instead of idling until the next month start. It does not change the
    standing strategy or the backtest (which always uses the default)."""

    name = AgentName.QUANT
    LOOKBACK = 420
    TOP_N = 8

    def __init__(self, force_rebalance: bool = False) -> None:
        self.force_rebalance = force_rebalance

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        if not self.force_rebalance and not _first_session_of_month(ctx.asof):
            return []  # hold between monthly rebalances
        close = _close_matrix(ctx.prices.get(ctx.universe, self.LOOKBACK))
        if len(close) < 260:
            return []
        mom = close.iloc[-21] / close.iloc[-252] - 1.0  # 12m return skipping the last month
        longs = list(mom.dropna().sort_values(ascending=False).head(self.TOP_N).index)
        return _decisions(self.name, longs, _held(ctx), ctx, "mom_12_1_top8")


# -------------------------------------------------------- 3. RSI-2 reversion
class RSI2Baseline(TradingAgent):
    """Connors RSI-2 in an uptrend: ENTER when RSI(2) < 10 and close > 200d SMA; HOLD until
    RSI(2) > 70, then EXIT. Equal-weight the held basket. Daily."""

    name = AgentName.QUANT
    LOOKBACK = 340  # ~235 sessions -> SMA200 is defined

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        close = _close_matrix(ctx.prices.get(ctx.universe, self.LOOKBACK))
        if len(close) < 200:
            return []
        sma200 = close.rolling(200).mean().iloc[-1]
        held = _held(ctx)
        longs = []
        for t in close.columns:
            rsi2 = _rsi(close[t], 2).iloc[-1]
            if pd.isna(rsi2):
                continue
            if t in held:
                if rsi2 <= 70:  # hold until the exit signal
                    longs.append(t)
            elif rsi2 < 10 and close[t].iloc[-1] > sma200[t]:  # entry
                longs.append(t)
        return _decisions(self.name, longs, held, ctx, "rsi2_reversion")


# ------------------------------------------------------- 4. Markov regime
class MarkovRegimeBaseline(TradingAgent):
    """Label each name Bull/Bear/Sideways by 20d rolling return (±2% threshold, fixed a
    priori). Estimate its 3×3 transition matrix on trailing labels only; go long when
    P(Bull | current state) − P(Bear | current state) > 0. Equal-weight. Daily.

    Caveat (doc 03): consecutive 20d-return labels share 19/20 window days, so the diagonal
    is partly an overlapping-window artifact — this is ~20d momentum in a Markov costume."""

    name = AgentName.QUANT
    LOOKBACK = 400
    _BINS = [-np.inf, -0.02, 0.02, np.inf]
    _LABELS = ["Bear", "Sideways", "Bull"]

    def _is_long(self, series: pd.Series) -> bool:
        ret20 = series.pct_change(20).dropna()
        if len(ret20) < 40:
            return False
        labels = pd.cut(ret20, self._BINS, labels=self._LABELS)
        trans = pd.crosstab(labels.shift(1), labels)  # from (row) -> to (col)
        cur = labels.iloc[-1]
        if cur not in trans.index:
            return False
        row = trans.loc[cur]
        total = row.sum()
        if total == 0:
            return False
        return (row.get("Bull", 0) - row.get("Bear", 0)) / total > 0

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        close = _close_matrix(ctx.prices.get(ctx.universe, self.LOOKBACK))
        if len(close) < 60:
            return []
        longs = [t for t in close.columns if self._is_long(close[t].dropna())]
        return _decisions(self.name, longs, _held(ctx), ctx, "markov_regime_20d")


BASELINES = {
    "ma_cross": MACrossBaseline,
    "momentum_12_1": MomentumBaseline,
    "rsi2": RSI2Baseline,
    "markov": MarkovRegimeBaseline,
}
