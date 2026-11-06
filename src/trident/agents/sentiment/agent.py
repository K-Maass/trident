"""SENTINEL agent (doc 04, pipeline stage 5) — the decision layer, pure & testable.

Reads the day's per-ticker sentiment aggregates from `ctx.extras['signals']` (built by
aggregate.py, injected by the runner) plus `ctx.extras['holding_days']` (how long each name has
been held, computed by the runner from the ledger — the pure agent can't query the DB) and emits
target-weight Decisions:

  ENTER 6%  when sent_z > +1.5 AND attention_z > +1 AND evidence floor met (>=5 relevant items,
            or >=3 + a top-20 ApeWisdom rank) AND not a knife (sent_z >= -2) AND < 6 held.
  EXIT      (explicit 0.0) when sent_z < 0 OR 10 trading days elapsed.
  KNIFE     sent_z < -2 & not held -> do NOT buy the dip (no decision; the surrounding
            orchestration logs the flag).

Max 6 concurrent positions; entries ranked by sent_z. Never trades thin evidence. The rationale
carries the top-5 contributing items (title/source/score/url) — the url is the item link back to
the news that drove the call (doc 04).
"""

from __future__ import annotations

from trident.agents.base import MarketContext, TradingAgent
from trident.config import load_risk_limits, required_risk_limit
from trident.models import AgentName, Decision

ENTER_WEIGHT = 0.06
SENT_Z_ENTER = 1.5
ATTN_Z_ENTER = 1.0
KNIFE_Z = -2.0
EXIT_SENT_Z = 0.0
MAX_HOLD_DAYS = 10
MIN_EVIDENCE = 5
MIN_EVIDENCE_WITH_APE = 3
APE_TOP_RANK = 20
MODEL = "sentinel_rules_v1"


class SentinelAgent(TradingAgent):
    name = AgentName.SENTINEL

    def __init__(
        self, max_position_weight: float | None = None, max_positions: int | None = None
    ) -> None:
        limits = load_risk_limits()
        self.max_pos = (
            float(max_position_weight)
            if max_position_weight is not None
            else required_risk_limit("max_position_weight", limits)
        )
        # Concentration control — sourced from risk_limits.yaml, never a magic number in agent
        # code (doc 06 / risk-officer binding condition, 2026-07-13).
        self.max_positions = int(
            max_positions if max_positions is not None else limits["max_positions_sentinel"]
        )

    async def generate_decisions(self, ctx: MarketContext) -> list[Decision]:
        signals: dict[str, dict] = ctx.extras.get("signals", {}) or {}
        holding_days: dict[str, int] = ctx.extras.get("holding_days", {}) or {}
        current = {p.ticker: ctx.book.weight(p.ticker) for p in ctx.book.positions if p.qty > 0}
        held = set(current)
        decisions: list[Decision] = []

        # 1) EXITS first — freed slots then count toward the max-6 cap for new entries.
        exited: set[str] = set()
        for tk in sorted(held):
            sig = signals.get(tk, {})
            sz = sig.get("sent_z")
            hd = int(holding_days.get(tk, 0))
            reason = None
            if hd >= MAX_HOLD_DAYS:
                reason = "max_hold_10d"
            elif sz is not None and sz < EXIT_SENT_Z:
                reason = "sent_z<0"
            if reason:
                decisions.append(
                    Decision(
                        agent=self.name,
                        asof=ctx.asof,
                        ticker=tk,
                        target_weight=0.0,
                        horizon_days=MAX_HOLD_DAYS,
                        rationale={
                            "model": MODEL,
                            "action": "exit",
                            "reason": reason,
                            "sent_z": sz,
                            "holding_days": hd,
                        },
                    )
                )
                exited.add(tk)

        slots = self.max_positions - len(held - exited)

        # 2) ENTRIES — rank qualifying non-held candidates by sent_z, fill remaining slots.
        # NOTE: the maturity gate below lives ONLY here, in the entry path. Exits and the
        # knife-guard (section 1) are deliberately never gated on maturity — a risk-reducing
        # action must stay allowed even if a held name's coverage regresses back to immature.
        candidates: list[tuple[float, str, dict]] = []
        for tk, sig in signals.items():
            if tk in held:
                continue
            if sig.get("immature", True):
                # A z-score off a <21-session baseline is not trustworthy enough to OPEN a
                # position on; entering here would let SENTINEL's first live trades fire on
                # ~10-observation baselines. Reachability/maturity are re-checked 2026-07-22.
                # Default True = fail CLOSED: a signal missing the flag is a data anomaly, and on
                # the open-a-position path we refuse rather than assume maturity — matching every
                # other gate in this loop (sent_z/attention_z None, n_evidence default 0).
                continue
            sz, az = sig.get("sent_z"), sig.get("attention_z")
            if sz is None or az is None:
                continue
            if sz < KNIFE_Z:  # never-catch-knives: do not buy the dip
                continue
            if not (sz > SENT_Z_ENTER and az > ATTN_Z_ENTER):
                continue
            n_ev = int(sig.get("n_evidence", 0))
            rank = sig.get("apewisdom_rank")
            floor = n_ev >= MIN_EVIDENCE or (
                n_ev >= MIN_EVIDENCE_WITH_APE and rank is not None and rank <= APE_TOP_RANK
            )
            if not floor:
                continue
            candidates.append((float(sz), tk, sig))

        candidates.sort(key=lambda c: c[0], reverse=True)
        for _, tk, sig in candidates[: max(0, slots)]:
            decisions.append(
                Decision(
                    agent=self.name,
                    asof=ctx.asof,
                    ticker=tk,
                    target_weight=min(ENTER_WEIGHT, self.max_pos),
                    horizon_days=MAX_HOLD_DAYS,
                    rationale={
                        "model": MODEL,
                        "sent_z": sig.get("sent_z"),
                        "attention_z": sig.get("attention_z"),
                        "disagreement": sig.get("disagreement"),
                        "sent_raw": sig.get("sent_raw"),
                        "n_evidence": sig.get("n_evidence"),
                        "apewisdom_rank": sig.get("apewisdom_rank"),
                        "sources_used": sig.get("sources_used", []),
                        "immature": sig.get("immature"),
                        "top_items": sig.get("top_items", []),
                    },
                )
            )
        return decisions
