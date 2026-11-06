"""ORACLE's portfolio construction (doc 05) — a PURE function; no I/O, no API, no DB.

The LLM proposes a thesis; position sizes are computed HERE from conviction x rating with caps
read from config, never suggested by the model. That is governing rule 1: sizing is arithmetic,
so it is code.

Rules (doc 05):
  - weight per name = conviction x rating-multiplier x max_position_weight, clamped to
    max_position_weight (<= 12%, `max_position_weight_oracle`).
  - at most `max_positions_oracle` (8) names — a concentration control, so its VALUE lives in
    risk_limits.yaml (mirroring max_positions_sentinel) and is enforced here.
  - cash is the default: only strong_buy/buy with positive conviction get weight; hold/avoid and
    any DROPPED (no-thesis) held name get an explicit 0-weight exit.
  - trade only if |target - current| > TRADE_MIN — a turnover control, hence a module constant
    mirroring QUANT's TURNOVER_MIN, not a concentration control.

Theses arrive as plain dicts (they travel through the runner's `extras`, which must be JSON).
"""

from __future__ import annotations

from datetime import datetime

from trident.models import AgentName, Decision

TRADE_MIN = 0.02  # trade only if |target - current| > 2% of the book (turnover damping)
_RATING_MULT = {"strong_buy": 1.0, "buy": 0.75, "hold": 0.0, "avoid": 0.0}


def target_weights(
    theses: list[dict], current: dict[str, float], *, max_positions: int, max_position_weight: float
) -> dict[str, float]:
    """Full target book: sized buys (top `max_positions` by conviction x rating) plus explicit
    0-weight exits for every held name not among them. Pure; operates on thesis DICTS."""
    scored: list[tuple[str, float, float]] = []
    for th in theses:
        ticker = str(th.get("ticker", "")).upper()
        mult = _RATING_MULT.get(str(th.get("rating")), 0.0)
        conv = float(th.get("conviction") or 0.0)
        w = min(conv * mult * max_position_weight, max_position_weight)
        if ticker and w > 0.0:
            scored.append((ticker, w, conv * mult))
    scored.sort(key=lambda r: r[2], reverse=True)
    targets = {t: w for t, w, _ in scored[:max_positions]}
    for t in current:  # held names not selected (avoid/hold/dropped) -> exit to cash
        targets.setdefault(t.upper(), 0.0)
    return targets


def _rationale(th: dict | None, target: float) -> dict:
    if target > 0 and th is not None:
        return {
            "model": "oracle_llm_v1",
            "rating": th.get("rating"),
            "conviction": round(float(th.get("conviction") or 0.0), 3),
            "thesis": (th.get("thesis") or "")[:280],
        }
    if th is None:
        reason = "no_thesis"  # dropped / never underwritten held name
    elif th.get("rating") in ("avoid", "hold"):
        reason = str(th.get("rating"))
    else:
        reason = "not_selected"  # ranked out by the position cap
    return {"model": "oracle_llm_v1", "action": "exit", "reason": reason}


def build_decisions(
    name: AgentName,
    theses: list[dict],
    current: dict[str, float],
    asof: datetime,
    *,
    max_positions: int,
    max_position_weight: float,
    trade_min: float = TRADE_MIN,
) -> list[Decision]:
    """Turn theses + the current book into target-weight Decisions, turnover-damped. This IS the
    agent's decision logic; agent.py only wires context in."""
    targets = target_weights(
        theses, current, max_positions=max_positions, max_position_weight=max_position_weight
    )
    by_ticker = {str(th.get("ticker", "")).upper(): th for th in theses}
    decisions: list[Decision] = []
    for ticker in sorted(set(targets) | set(current)):
        target = targets.get(ticker, 0.0)
        if abs(target - current.get(ticker, 0.0)) <= trade_min:
            continue  # turnover damping: leave unchanged
        decisions.append(
            Decision(
                agent=name,
                asof=asof,
                ticker=ticker,
                target_weight=round(target, 6),
                horizon_days=60,
                rationale=_rationale(by_ticker.get(ticker), target),
            )
        )
    return decisions
