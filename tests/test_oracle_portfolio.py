"""ORACLE portfolio sizing — property tests that actively try to VIOLATE each safety rule
(Phase-4 lesson: a passing dry-run proves outcome, not mechanism). Pure functions, no I/O."""

from __future__ import annotations

from datetime import UTC, datetime

from trident.agents.analyst.portfolio import build_decisions, target_weights
from trident.models import AgentName

ASOF = datetime(2026, 7, 21, tzinfo=UTC)
CAPS = {"max_positions": 8, "max_position_weight": 0.12}


def _th(ticker, rating="buy", conviction=0.8, tw=0.1):
    return {
        "ticker": ticker,
        "rating": rating,
        "conviction": conviction,
        "target_weight": tw,
        "thesis": "x",
    }


def test_a_thesis_asking_20pct_is_clamped_to_12pct():
    tw = target_weights([_th("AAPL", "strong_buy", 1.0, 0.20)], {}, **CAPS)
    assert tw["AAPL"] == 0.12


def test_12_theses_produce_at_most_8_positions():
    theses = [_th(f"T{i}", "buy", 0.9 - 0.01 * i) for i in range(12)]
    tw = target_weights(theses, {}, **CAPS)
    assert sum(1 for w in tw.values() if w > 0) == 8


def test_a_1_5pct_weight_delta_produces_no_decision():
    th = [_th("AAPL", "buy", 0.6)]
    tw = target_weights(th, {}, **CAPS)["AAPL"]
    current = {"AAPL": tw - 0.015}  # inside the 2% turnover band
    assert build_decisions(AgentName.ORACLE, th, current, ASOF, **CAPS) == []


def test_avoid_rating_on_held_name_produces_a_zero_weight_exit():
    d = build_decisions(AgentName.ORACLE, [_th("MSFT", "avoid", 0.9)], {"MSFT": 0.10}, ASOF, **CAPS)
    assert len(d) == 1 and d[0].ticker == "MSFT" and d[0].target_weight == 0.0
    assert d[0].rationale["action"] == "exit" and d[0].rationale["reason"] == "avoid"


def test_dropped_held_name_with_no_thesis_exits_to_cash():
    d = build_decisions(AgentName.ORACLE, [], {"NVDA": 0.08}, ASOF, **CAPS)
    assert len(d) == 1 and d[0].target_weight == 0.0 and d[0].rationale["reason"] == "no_thesis"


def test_hold_rating_is_not_bought():
    assert target_weights([_th("AAPL", "hold", 1.0)], {}, **CAPS).get("AAPL", 0.0) == 0.0


def test_weight_scales_with_conviction_and_rating():
    strong = target_weights([_th("A", "strong_buy", 1.0)], {}, **CAPS)["A"]
    weak = target_weights([_th("B", "buy", 0.5)], {}, **CAPS)["B"]
    assert strong == 0.12
    assert weak == 0.5 * 0.75 * 0.12 < strong
