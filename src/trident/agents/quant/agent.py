"""QUANT agent (doc 03) — cross-sectional LightGBM ranker, zero LLM.

Daily: score the universe -> P(top tercile), long the top-8 clearing the base-rate floor
(equal-weight, capped by risk_limits), else more cash. Turnover damping: skip any name whose
target vs current weight moves < 1% of the book. Rationale per pick = top-5 TreeSHAP features.

`QuantAgent` scores live from the active model. `OOSReplayAgent` replays the walk-forward
OUT-OF-SAMPLE predictions (each from a model trained only on prior data) — the only valid way
to backtest the ML without in-sample look-ahead. Both share `build_decisions`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text

from trident.agents.base import MarketContext, TradingAgent
from trident.agents.quant import model as qmodel
from trident.agents.quant.features import FEATURE_NAMES, compute_features
from trident.config import required_risk_limit
from trident.data import calendar as trading_calendar
from trident.db import get_engine
from trident.models import AgentName, Decision

MAX_POS = required_risk_limit("max_position_weight")
TOP_N = 8
# doc 03 says "score > 0.55", but that assumes a 0.5-base-rate up/down score. The label is
# TOP-TERCILE, so P(top tercile) centers on the 1/3 base rate (observed 0.30-0.38) and 0.55 is
# unreachable. The principled "above chance" floor for a tercile model is the base rate; we
# long the top-8 by score that clear it. Chosen a priori, NOT tuned to results.
SCORE_THRESHOLD = 1.0 / 3.0
TURNOVER_MIN = 0.01  # trade only if |target - current| > 1% of book (doc 03)


async def active_model_path() -> str | None:
    """artifact_path of the active QUANT model in model_registry (newest active wins)."""
    async with get_engine().connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT artifact_path FROM model_registry WHERE agent_id='quant' AND active "
                    "ORDER BY trained_at DESC LIMIT 1"
                )
            )
        ).first()
    return row[0] if row else None


def build_decisions(name, scores: pd.Series, current: dict[str, float], asof, rationale_for):
    """Shared: top-8 scores above the floor -> equal-weight targets; exit held-but-unpicked;
    turnover-damp; attach a rationale via `rationale_for(ticker, target)`."""
    picks = scores[scores > SCORE_THRESHOLD].sort_values(ascending=False).head(TOP_N)
    n = len(picks)
    weight = min(1.0 / n, MAX_POS) if n else 0.0
    targets = {t: weight for t in picks.index}
    decisions: list[Decision] = []
    for ticker in set(targets) | set(current):
        target = targets.get(ticker, 0.0)  # held-but-not-picked -> exit to cash
        if abs(target - current.get(ticker, 0.0)) <= TURNOVER_MIN:
            continue  # turnover damping: leave unchanged
        decisions.append(
            Decision(
                agent=name,
                asof=asof,
                ticker=ticker,
                target_weight=target,
                horizon_days=5,
                rationale=rationale_for(ticker, target),
            )
        )
    return decisions


class QuantAgent(TradingAgent):
    name = AgentName.QUANT
    LOOKBACK = 400  # calendar days -> enough for 126d returns, SMA100, SPY regime warmup

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._booster = None

    async def _ensure_model(self) -> bool:
        if self._booster is not None:
            return True
        path = self._model_path or await active_model_path()
        if path is None:
            return False
        self._booster = qmodel.load(path)
        return True

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        if not await self._ensure_model():
            return []  # no model registered -> do nothing (fail safe)
        prices = ctx.prices.get(ctx.universe, self.LOOKBACK)
        if prices.empty:
            return []
        feats = compute_features(prices, ctx.asof).dropna(how="all")
        if feats.empty:
            return []
        X = feats[FEATURE_NAMES]
        scores = pd.Series(qmodel.predict(self._booster, X), index=feats.index)
        current = {p.ticker: ctx.book.weight(p.ticker) for p in ctx.book.positions}
        contribs = self._rationales(X, scores)

        def rationale_for(ticker, target):
            if target > 0:
                return {
                    "model": "quant_lgbm",
                    "score": round(float(scores[ticker]), 4),
                    "top_features": contribs.get(ticker, []),
                }
            return {
                "model": "quant_lgbm",
                "action": "exit",
                "score": round(float(scores.get(ticker, float("nan"))), 4),
            }

        return build_decisions(self.name, scores, current, ctx.asof, rationale_for)

    def _rationales(self, X: pd.DataFrame, scores: pd.Series) -> dict[str, list]:
        """Top-5 |TreeSHAP| feature contributions per name (doc 03 rationale)."""
        contrib = qmodel.feature_contributions(self._booster, X)  # (n, n_features+1)
        out: dict[str, list] = {}
        for i, t in enumerate(X.index):
            c = contrib[i, : len(FEATURE_NAMES)]
            top = np.argsort(np.abs(c))[::-1][:5]
            out[t] = [
                {"feature": FEATURE_NAMES[j], "contribution": round(float(c[j]), 4)} for j in top
            ]
        return out


class OOSReplayAgent(TradingAgent):
    """Backtest-only: replays walk-forward OUT-OF-SAMPLE predictions (date, ticker, score) so
    the ML's costed Sharpe is honest — every score came from a model trained only on data
    strictly before that date. Reuses the exact QuantAgent selection logic.

    EXECUTION ALIGNMENT: a score keyed at date d is computed from features that INCLUDE d's
    close (e.g. ret_1d(d)=log(close_d/close_(d-1))), so it is only known after d's close and
    can first be TRADED at the next session's open. At session s we therefore use the score
    keyed at previous_session(s) — matching the live QuantAgent, which computes features from
    bars ≤ s-1. (Using score(s) at s's open would be a 1-day look-ahead.)"""

    name = AgentName.QUANT

    def __init__(self, oos_predictions: pd.DataFrame) -> None:
        o = oos_predictions.copy()
        o["date"] = pd.to_datetime(o["date"]).dt.normalize()
        self._by_date = {d: g.set_index("ticker")["score"] for d, g in o.groupby("date")}

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        s = pd.Timestamp(ctx.asof).tz_localize(None).normalize()
        prev = pd.Timestamp(
            trading_calendar.previous_session(s.date())
        )  # score executable at s-open
        scores = self._by_date.get(prev)
        if scores is None or scores.empty:
            return []
        current = {p.ticker: ctx.book.weight(p.ticker) for p in ctx.book.positions}

        def rationale_for(ticker, target):
            return {
                "model": "quant_lgbm_oos",
                "score": round(float(scores.get(ticker, 0.0)), 4),
                **({"action": "exit"} if target == 0 else {}),
            }

        return build_decisions(self.name, scores, current, ctx.asof, rationale_for)
