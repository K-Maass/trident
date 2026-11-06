from datetime import date
from decimal import Decimal as D

import pytest

from trident.risk.turnover import (
    CanonicalBaseline,
    Commitment,
    ExactExecution,
    TurnoverExemption,
    authorize_turnover,
)

SESSION = date(2026, 9, 8)
PREVIOUS = date(2026, 9, 4)  # Labor Day weekend: use the previous XNYS session.


def order(side="buy", qty="1", price="100", reference="100", ticker="AAPL", id="new"):
    return Commitment(id, "p", ticker, side, D(qty), D(price), D(reference))


def fill(id="f", side="buy", amount="100", ticker="MSFT", session=SESSION):
    return ExactExecution(id, "p", session, ticker, side, D(amount))


def authorize(**kwargs):
    args = dict(
        portfolio_id="p",
        session=SESSION,
        previous_session=PREVIOUS,
        max_daily_turnover=D("0.5"),
        baseline=CanonicalBaseline("p", PREVIOUS, 1, D("1000"), True),
        executions=(),
        commitments=(),
        proposed=order(),
    )
    args.update(kwargs)
    return authorize_turnover(**args)


def test_gross_buys_and_sells_do_not_net_and_limit_is_inclusive():
    executions = (fill("b", "buy", "200"), fill("s", "sell", "200"))
    assert authorize(executions=executions).approved
    refused = authorize(executions=executions, proposed=order(price="100.01"))
    assert not refused.approved
    assert refused.exact_filled_notional == D("400")
    assert refused.projected_notional == D("500.01")


def test_outstanding_buy_and_sell_use_conservative_prices():
    commitments = (
        order(qty="2", price="105", reference="100", ticker="MSFT", id="b"),
        order("sell", qty="2", price="95", reference="110", ticker="NVDA", id="s"),
    )
    result = authorize(commitments=commitments)
    assert result.outstanding_notional == D("430")
    assert not result.approved
    assert order("sell", price="115", reference="110").notional == D("115")


@pytest.mark.parametrize(
    "baseline",
    [
        None,
        CanonicalBaseline("p", SESSION, 1, D("1000"), True),
        CanonicalBaseline("p", PREVIOUS, 1, D("1000"), False),
        CanonicalBaseline("other", PREVIOUS, 1, D("1000"), True),
    ],
)
def test_only_eligible_previous_portfolio_session_baseline_is_accepted(baseline):
    assert authorize(baseline=baseline).reason == "missing_eligible_previous_session_baseline"


@pytest.mark.parametrize("side,opposite", [("buy", "sell"), ("sell", "buy")])
def test_opposite_execution_blocks_both_directions_and_exemptions(side, opposite):
    exemption = TurnoverExemption(
        "initial_bench" if side == "buy" else "kill_switch", "event", "p", "new", SESSION
    )
    result = authorize(
        proposed=order(side), executions=(fill(side=opposite, ticker="AAPL"),), exemption=exemption
    )
    assert result.reason == "same_session_opposite_fill"


def test_uncertain_opposite_commitment_remains_blocking():
    assert (
        authorize(commitments=(order("sell", id="uncertain"),)).reason
        == "unresolved_opposite_commitment"
    )


def test_favorable_fill_overrun_is_counted_without_rejecting_fill_truth():
    result = authorize(executions=(fill(side="sell", amount="550"),))
    assert result.reason == "daily_turnover_exceeded"
    assert result.exact_filled_notional == D("550")
    assert result.budget == D("500")


def test_exemption_bypasses_only_budget_and_activity_is_still_counted():
    exemption = TurnoverExemption("kill_switch", "halt-event", "p", "new", SESSION)
    result = authorize(
        baseline=None, executions=(fill(amount="600"),), proposed=order("sell"), exemption=exemption
    )
    assert result.approved and result.exemption_id == "halt-event"
    assert result.projected_notional == D("700")
    assert not authorize(exemption=exemption).approved  # SELL exemption cannot authorize a BUY.


def test_previous_session_fills_do_not_consume_today_budget():
    assert authorize(executions=(fill(amount="10000", session=PREVIOUS),)).approved


def test_duplicate_evidence_and_reauthorization_are_refused():
    with pytest.raises(ValueError, match="duplicate execution"):
        authorize(executions=(fill(), fill()))
    with pytest.raises(ValueError, match="duplicate outstanding"):
        authorize(commitments=(order(id="old"), order(id="old")))
    with pytest.raises(ValueError, match="recovery"):
        authorize(commitments=(order(),))


@pytest.mark.parametrize("amount", [D("NaN"), D("Infinity"), D("-1"), 1.0])
def test_nonfinite_negative_and_float_money_is_refused(amount):
    with pytest.raises(ValueError):
        ExactExecution("f", "p", SESSION, "AAPL", "buy", amount)


def test_immutable_policy_fraction_controls_budget():
    result = authorize(max_daily_turnover=D("0.05"))
    assert result.budget == D("50")
    assert not result.approved
