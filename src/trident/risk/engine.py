"""RiskEngine — agents propose, this disposes (doc 06). Parameters from config/risk_limits.yaml.

Implemented now: pure-logic checks (caps, exposure scaling, kill-switch math).
Phase 2 wires the data-dependent inputs (ADV, last/prev prices, market state) via RiskInputs.
Every clamp/reject carries a machine-readable reason -> orders.risk_notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trident.config import load_risk_limits, load_universe, required_risk_limit
from trident.models import BookSnapshot, Decision, RiskVerdict


@dataclass
class RiskInputs:
    """Data the engine needs; assembled by the runner each cycle (Phase 2)."""

    last_price: dict[str, float]
    prev_close: dict[str, float]
    adv_dollars_21d: dict[str, float]
    market_open: bool
    minutes_to_close: float
    halted_tickers: set[str] = field(default_factory=set)
    global_freeze: bool = False  # CRITICAL health event in last 24h (doc 06 rule 10)


class RiskEngine:
    def __init__(self) -> None:
        lim = load_risk_limits()
        # Required, fail-loud: QUANT trades live on this limit; a silent 0.10 fallback would hide
        # a config gap in the money-path backstop.
        self.max_pos = required_risk_limit("max_position_weight", lim)
        # Required, fail-loud: ORACLE trades live on this cap (Phase 5), so a silent 0.12
        # fallback would hide a config gap in the money-path backstop — the same class of gap
        # already closed for max_position_weight.
        self.max_pos_oracle = required_risk_limit("max_position_weight_oracle", lim)
        self.max_gross = float(lim.get("max_gross_exposure", 1.0))
        self.min_adv = float(lim.get("min_adv_dollars", 50_000_000))
        self.max_pct_adv = float(lim.get("max_order_pct_adv", 0.01))
        self.min_price = float(lim.get("min_price", 5.0))
        self.price_jump_reject = float(lim.get("price_jump_reject", 0.10))
        self.max_daily_turnover = float(lim.get("max_daily_turnover", 0.50))
        self.kill_dd = float(lim.get("kill_switch_drawdown", 0.15))
        self.close_buffer_min = float(lim.get("close_buffer_minutes", 2))
        self.universe = set(load_universe())

    # ------------------------------------------------------------ per-decision
    def review(
        self, decisions: list[Decision], book: BookSnapshot, inputs: RiskInputs
    ) -> list[RiskVerdict]:
        verdicts: list[RiskVerdict] = []
        if inputs.global_freeze:
            return [
                RiskVerdict(decision_id=d.id, approved=False, reasons=["reject:global_freeze"])
                for d in decisions
            ]
        for d in decisions:
            reasons: list[str] = []
            w = d.target_weight

            if d.ticker not in self.universe:
                verdicts.append(self._reject(d, "reject:not_in_universe"))
                continue
            if d.ticker in inputs.halted_tickers:
                verdicts.append(self._reject(d, "reject:halted"))
                continue
            if not inputs.market_open or inputs.minutes_to_close < self.close_buffer_min:
                verdicts.append(self._reject(d, "reject:market_state"))
                continue

            price = inputs.last_price.get(d.ticker)
            prev = inputs.prev_close.get(d.ticker)
            if price is None or prev is None:
                verdicts.append(self._reject(d, "reject:no_price"))
                continue
            if price < self.min_price:
                verdicts.append(self._reject(d, "reject:penny_stock"))
                continue
            if abs(price / prev - 1) > self.price_jump_reject:
                verdicts.append(self._reject(d, "reject:price_jump_sanity"))
                continue
            if inputs.adv_dollars_21d.get(d.ticker, 0.0) < self.min_adv:
                verdicts.append(self._reject(d, "reject:illiquid"))
                continue

            cap = self.max_pos_oracle if d.agent.value == "oracle" else self.max_pos
            if w > cap:
                w, reasons = cap, reasons + [f"clamp:position_cap:{cap}"]

            # order-size vs ADV (only the delta trades)
            delta_dollars = abs(w - book.weight(d.ticker)) * book.equity
            max_dollars = inputs.adv_dollars_21d[d.ticker] * self.max_pct_adv
            if delta_dollars > max_dollars:
                w_delta_capped = max_dollars / book.equity
                base = book.weight(d.ticker)
                w = base + (w_delta_capped if w > base else -w_delta_capped)
                reasons.append("clamp:adv_participation")

            verdicts.append(
                RiskVerdict(
                    decision_id=d.id, approved=True, adjusted_target_weight=w, reasons=reasons
                )
            )

        return self._scale_gross(decisions, verdicts, book)

    # ------------------------------------------------------------- book-level
    def _scale_gross(
        self, decisions: list[Decision], verdicts: list[RiskVerdict], book: BookSnapshot
    ) -> list[RiskVerdict]:
        """If post-trade gross > max_gross, scale new buys so gross lands AT max_gross (doc 06 #4).

        Only newly-decided buys are scalable; positions held with no decision this cycle are
        fixed (the engine can't sell them here). So the buy budget is (max_gross - fixed), and
        scale = (max_gross - fixed) / Σ(buy targets). Scaling buys by max_gross/gross instead
        would leave gross ABOVE the cap whenever fixed holdings exist (implicit leverage).
        """
        by_id = {d.id: d for d in decisions}
        target_by_ticker = {p.ticker: book.weight(p.ticker) for p in book.positions}
        scalable_verdicts: list[RiskVerdict] = []
        for v in verdicts:
            if v.approved:
                w = v.adjusted_target_weight or 0.0
                target_by_ticker[by_id[v.decision_id].ticker] = w
                if w > 0:
                    scalable_verdicts.append(v)
        gross = sum(target_by_ticker.values())
        scalable = sum(v.adjusted_target_weight or 0.0 for v in scalable_verdicts)
        if gross <= self.max_gross or scalable <= 0:
            return verdicts
        fixed = gross - scalable  # held names with no decision this cycle
        available = max(0.0, self.max_gross - fixed)
        scale = available / scalable
        for v in scalable_verdicts:
            v.adjusted_target_weight = (v.adjusted_target_weight or 0.0) * scale
            v.reasons.append(f"clamp:gross_exposure_scale:{scale:.3f}")
        return verdicts

    def kill_switch_should_fire(self, equity: float, high_water_mark: float) -> bool:
        """Agent equity below (1 - kill_dd) x HWM -> halt + flatten (doc 06 #9).
        Reactivation is MANUAL ONLY. Unit-test with synthetic 16% drawdown (Phase 2)."""
        return high_water_mark > 0 and equity < high_water_mark * (1 - self.kill_dd)

    @staticmethod
    def _reject(d: Decision, reason: str) -> RiskVerdict:
        return RiskVerdict(decision_id=d.id, approved=False, reasons=[reason])
