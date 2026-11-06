"""Runner seam tests — `run_agent` is the money path (golden rule 1's flow lives there and
nowhere else), so the `extras` channel it opens to agent code gets its own coverage.

These stub persistence/broker/finalize: the point is the seam (what reaches the agent, what is
logged, and in what order), not the DB.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trident.agents.base import TradingAgent
from trident.ledger.book import Book
from trident.models import AgentName, AgentStatus, Decision
from trident.orchestration import runner as R

ASOF = datetime(2026, 7, 10, 14, 0, tzinfo=UTC)


class SpyAgent(TradingAgent):
    """Captures the MarketContext it was handed, and emits whatever decisions we hand it."""

    name = AgentName.SENTINEL

    def __init__(self, decisions: list[Decision] | None = None) -> None:
        self.seen_ctx = None
        self._decisions = decisions or []

    async def generate_decisions(self, ctx):
        self.seen_ctx = ctx
        return self._decisions


def _decision(ticker: str = "AAPL") -> Decision:
    return Decision(
        agent=AgentName.SENTINEL,
        asof=ASOF,
        ticker=ticker,
        target_weight=0.06,
        rationale={"model": "test"},
    )


@pytest.fixture
def seam(monkeypatch):
    """Stub everything around run_agent and record the call order + logged decisions."""
    calls: list[str] = []
    logged: list[Decision] = []
    health: list[tuple] = []

    async def get_status(_name):
        return AgentStatus.ACTIVE

    async def load_book(name):
        return Book.new(name, 30_000)

    async def log_decision(d):
        calls.append("log_decision")
        logged.append(d.model_copy(deep=True))  # freeze it: prove the digest was set BEFORE

    async def live_prices(_broker, _tickers):
        return {}

    async def finalize(_name, _book, _broker, _asof, _result):
        calls.append("finalize")

    async def health_event(sev, src, detail):
        health.append((sev, src, detail))

    async def risk_inputs(*_a, **_k):
        return None

    class FakeEngine:
        def review(self, *_a, **_k):
            calls.append("risk_review")
            return []  # reject-all: keeps the test on the seam, not on risk math

    class FakeExecution:
        limit_bps = 50

        def __init__(self, *_a, **_k): ...

        async def execute(self, orders, _book):
            calls.append("execute")
            return []

    monkeypatch.setattr(R.persistence, "get_status", get_status)
    monkeypatch.setattr(R.persistence, "load_book", load_book)
    monkeypatch.setattr(R.persistence, "log_decision", log_decision)
    monkeypatch.setattr(R, "_live_prices", live_prices)
    monkeypatch.setattr(R, "_finalize", finalize)
    monkeypatch.setattr(R, "_health", health_event)
    monkeypatch.setattr(R, "_risk_inputs", risk_inputs)
    monkeypatch.setattr(R, "RiskEngine", FakeEngine)
    monkeypatch.setattr(R, "ExecutionService", FakeExecution)
    return {"calls": calls, "logged": logged, "health": health}


async def test_extras_reach_the_agent_and_default_to_empty(seam):
    agent = SpyAgent()
    await R.run_agent(agent, broker=object(), dry_run=True, asof=ASOF)
    assert agent.seen_ctx.extras == {}  # QUANT passes nothing → {} , not None

    agent = SpyAgent()
    payload = {"signals": {"AAPL": {"sent_z": 2.0}}, "holding_days": {"AAPL": 3}}
    await R.run_agent(agent, broker=object(), dry_run=True, asof=ASOF, extras=payload)
    assert agent.seen_ctx.extras == payload


async def test_every_decision_is_logged_with_a_digest_before_any_order(seam):
    agent = SpyAgent([_decision("AAPL"), _decision("MSFT")])
    await R.run_agent(
        agent, broker=object(), dry_run=True, asof=ASOF, extras={"signals": {"AAPL": 1.0}}
    )
    # golden rule 2: logged, with a digest, BEFORE risk review or any order is executed.
    assert seam["calls"].index("log_decision") < seam["calls"].index("risk_review")
    assert seam["calls"].index("log_decision") < seam["calls"].index("execute")
    assert len(seam["logged"]) == 2
    assert all(d.inputs_digest for d in seam["logged"])


async def test_digest_pins_extras(seam):
    """The whole point: SENTINEL's inputs arrive ONLY via extras, and aggregate() rewrites its
    source rows on every run (delete-then-insert). If the digest ignored extras, a silent
    re-score behind an old decision would be undetectable."""

    async def digest_for(extras):
        seam["logged"].clear()
        await R.run_agent(
            SpyAgent([_decision()]), broker=object(), dry_run=True, asof=ASOF, extras=extras
        )
        return seam["logged"][0].inputs_digest

    d1 = await digest_for({"signals": {"AAPL": {"sent_z": 2.0}}})
    d2 = await digest_for({"signals": {"AAPL": {"sent_z": 2.0}}})
    d3 = await digest_for({"signals": {"AAPL": {"sent_z": 2.9}}})  # a re-score
    assert d1 == d2  # same inputs → same digest (reproducible)
    assert d1 != d3  # different inputs → different digest (drift is detectable)


async def test_non_data_extras_are_refused_and_nothing_is_logged(seam):
    """A live Broker/session/callable in extras would let an agent reach the broker without
    RiskEngine (golden rule 1). Refuse the run loudly (golden rule 8) — never trust it."""

    class FakeBroker:
        async def submit(self, *_a, **_k): ...

    agent = SpyAgent([_decision()])
    result = await R.run_agent(
        agent, broker=object(), dry_run=True, asof=ASOF, extras={"broker": FakeBroker()}
    )
    assert result.status == "skipped_bad_extras"
    assert result.n_decisions == 0 and result.n_orders == 0
    assert agent.seen_ctx is None  # the agent never even ran
    assert seam["logged"] == []
    # alerted, but WARN not CRITICAL: a CRITICAL here would trip the global freeze and halt the
    # other two agents over one agent's wiring bug.
    assert seam["health"] and seam["health"][0][0] is R.HealthSeverity.WARN


def test_extras_are_data_rejects_live_objects_and_nan():
    assert R._extras_are_data({"signals": {"AAPL": {"sent_z": 1.9}}, "holding_days": {"A": 2}})
    assert not R._extras_are_data({"broker": object()})
    assert not R._extras_are_data({"fn": lambda: None})
    # a NaN leaking out of aggregate() must refuse the run, not bake a non-standard token into
    # the digest (aggregate() is supposed to _nan_to_none them — this is the backstop).
    assert not R._extras_are_data({"signals": {"AAPL": {"sent_z": float("nan")}}})
