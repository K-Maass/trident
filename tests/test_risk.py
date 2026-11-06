"""RiskEngine checks (doc 06). The engine logic already exists; these feed it synthetic
RiskInputs (no live data needed) to lock every reject/clamp path — money-path code that
shipped without tests. Asserts against the real config/risk_limits.yaml values."""

from __future__ import annotations

from datetime import UTC, datetime

from trident.models import AgentName, BookSnapshot, Decision, Position
from trident.risk.engine import RiskEngine, RiskInputs

TS = datetime(2026, 7, 3, 15, 0, tzinfo=UTC)


def _decision(ticker="AAPL", w=0.05, agent=AgentName.QUANT) -> Decision:
    return Decision(agent=agent, asof=TS, ticker=ticker, target_weight=w, rationale={"t": "test"})


def _inputs(tickers=("AAPL",), price=250.0, prev=248.0, adv=1e9, market_open=True, mtc=120.0):
    return RiskInputs(
        last_price={t: price for t in tickers},
        prev_close={t: prev for t in tickers},
        adv_dollars_21d={t: adv for t in tickers},
        market_open=market_open,
        minutes_to_close=mtc,
    )


def _empty_book(agent=AgentName.QUANT, cash=30000.0) -> BookSnapshot:
    return BookSnapshot(agent=agent, ts=TS, cash=cash, positions=[])


def _only(verdicts):
    assert len(verdicts) == 1
    return verdicts[0]


# --------------------------------------------------------------- happy path
def test_happy_path_approves_unchanged():
    v = _only(RiskEngine().review([_decision(w=0.05)], _empty_book(), _inputs()))
    assert v.approved
    assert v.adjusted_target_weight == 0.05
    assert v.reasons == []


# ------------------------------------------------------------------ rejects
def test_reject_not_in_universe():
    v = _only(RiskEngine().review([_decision(ticker="FAKE")], _empty_book(), _inputs(("FAKE",))))
    assert not v.approved and v.reasons == ["reject:not_in_universe"]


def test_reject_halted_ticker():
    inp = _inputs()
    inp.halted_tickers = {"AAPL"}
    v = _only(RiskEngine().review([_decision()], _empty_book(), inp))
    assert not v.approved and v.reasons == ["reject:halted"]


def test_reject_market_closed():
    v = _only(RiskEngine().review([_decision()], _empty_book(), _inputs(market_open=False)))
    assert not v.approved and v.reasons == ["reject:market_state"]


def test_reject_within_close_buffer():
    v = _only(RiskEngine().review([_decision()], _empty_book(), _inputs(mtc=1.0)))  # < 2 min
    assert not v.approved and v.reasons == ["reject:market_state"]


def test_reject_no_price():
    inp = RiskInputs(
        last_price={},
        prev_close={},
        adv_dollars_21d={"AAPL": 1e9},
        market_open=True,
        minutes_to_close=120,
    )
    v = _only(RiskEngine().review([_decision()], _empty_book(), inp))
    assert not v.approved and v.reasons == ["reject:no_price"]


def test_reject_penny_stock():
    v = _only(RiskEngine().review([_decision()], _empty_book(), _inputs(price=3.0, prev=3.0)))
    assert not v.approved and v.reasons == ["reject:penny_stock"]


def test_reject_price_jump_sanity():
    # 300 vs prev 250 = +20% > 10% band
    v = _only(RiskEngine().review([_decision()], _empty_book(), _inputs(price=300.0, prev=250.0)))
    assert not v.approved and v.reasons == ["reject:price_jump_sanity"]


def test_reject_illiquid():
    v = _only(RiskEngine().review([_decision()], _empty_book(), _inputs(adv=1e6)))  # < $50M
    assert not v.approved and v.reasons == ["reject:illiquid"]


def test_global_freeze_rejects_all():
    inp = _inputs(("AAPL", "MSFT"))
    inp.global_freeze = True
    verdicts = RiskEngine().review([_decision("AAPL"), _decision("MSFT")], _empty_book(), inp)
    assert len(verdicts) == 2
    assert all(not v.approved and v.reasons == ["reject:global_freeze"] for v in verdicts)


# ------------------------------------------------------------------- clamps
def test_clamp_position_cap_quant():
    v = _only(RiskEngine().review([_decision(w=0.20)], _empty_book(), _inputs()))
    assert v.approved
    assert v.adjusted_target_weight == 0.10  # max_position_weight
    assert "clamp:position_cap:0.1" in v.reasons


def test_oracle_gets_higher_cap():
    # ORACLE cap is 0.12: 0.12 passes, QUANT's 0.12 clamps to 0.10
    vo = _only(
        RiskEngine().review(
            [_decision(w=0.12, agent=AgentName.ORACLE)], _empty_book(AgentName.ORACLE), _inputs()
        )
    )
    assert vo.approved and vo.adjusted_target_weight == 0.12 and vo.reasons == []
    vq = _only(RiskEngine().review([_decision(w=0.12)], _empty_book(), _inputs()))
    assert vq.adjusted_target_weight == 0.10


def test_clamp_adv_participation_on_large_book():
    # Paper size never trips this (doc 06); use a big book so the delta exceeds 1% ADV.
    book = _empty_book(cash=100_000_000.0)  # equity 100M
    inp = _inputs(adv=50_000_000.0)  # liquid enough to pass, 1% = $500k cap
    v = _only(RiskEngine().review([_decision(w=0.10)], book, inp))  # delta $10M > $500k
    assert v.approved
    assert "clamp:adv_participation" in v.reasons
    assert abs(v.adjusted_target_weight - 0.005) < 1e-9  # 500k / 100M


def test_gross_exposure_scales_new_buys_down():
    # Book already 95% gross; a new 10% buy pushes post-trade to 105% -> scale to 100%.
    book = BookSnapshot(
        agent=AgentName.QUANT,
        ts=TS,
        cash=1500.0,
        positions=[
            Position(
                agent=AgentName.QUANT, ticker="AAPL", qty=50, avg_cost=300, last_price=300
            ),  # 15000
            Position(
                agent=AgentName.QUANT, ticker="MSFT", qty=135, avg_cost=100, last_price=100
            ),  # 13500
        ],
    )  # equity 30000; AAPL 0.50, MSFT 0.45
    inp = _inputs(("NVDA",), price=500.0, prev=498.0)
    v = _only(RiskEngine().review([_decision("NVDA", w=0.10)], book, inp))
    assert v.approved
    assert any(r.startswith("clamp:gross_exposure_scale") for r in v.reasons)
    # budget for buys = max_gross - fixed(0.95) = 0.05, so NVDA -> 0.05 (not 0.10/1.05)
    assert abs(v.adjusted_target_weight - 0.05) < 1e-9
    # the invariant that matters (doc 06 rule 4/11): post-trade gross must not exceed the cap
    post_trade_gross = 0.50 + 0.45 + v.adjusted_target_weight
    assert post_trade_gross <= 1.00 + 1e-9


# -------------------------------------------------------------- kill switch
def test_kill_switch_fires_below_85pct_hwm():
    eng = RiskEngine()  # kill_switch_drawdown 0.15 -> threshold 85% HWM
    assert eng.kill_switch_should_fire(equity=84_000, high_water_mark=100_000)  # -16%
    assert not eng.kill_switch_should_fire(equity=86_000, high_water_mark=100_000)  # -14%
    assert not eng.kill_switch_should_fire(equity=85_000, high_water_mark=100_000)  # exactly -15%
    assert not eng.kill_switch_should_fire(equity=50_000, high_water_mark=0)  # no HWM yet
