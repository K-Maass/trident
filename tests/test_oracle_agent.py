"""ORACLE agent + graph — the pure agent reads theses from extras, and the research graph drops
a name whose citations never resolve (no thesis -> no position, the fail-safe direction).
Offline: a fake Anthropic client (the duck-typed pattern from test_sentiment_relevance)."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from trident.agents.analyst import graph as G
from trident.agents.analyst.agent import OracleAgent
from trident.agents.analyst.llm import AnalystLLM
from trident.agents.base import MarketContext
from trident.data import fundamentals as f
from trident.models import AgentName, BookSnapshot, Position

ASOF = datetime(2026, 7, 21, tzinfo=UTC)


# ------------------------------------------------------------------- pure agent
async def test_agent_sizes_from_extras_theses():
    theses = [
        {
            "ticker": "AAPL",
            "rating": "strong_buy",
            "conviction": 1.0,
            "target_weight": 0.10,
            "thesis": "cheap and high quality [T1]",
        }
    ]
    book = BookSnapshot(agent=AgentName.ORACLE, ts=ASOF, cash=30000.0, positions=[])
    ctx = MarketContext(
        asof=ASOF, universe=["AAPL"], book=book, prices=object(), extras={"theses": theses}
    )
    decisions = await OracleAgent().generate_decisions(ctx)
    assert len(decisions) == 1
    assert decisions[0].ticker == "AAPL"
    assert decisions[0].target_weight == 0.12  # conviction 1.0 x strong_buy, clamped at the cap


async def test_agent_exits_held_name_absent_from_theses():
    book = BookSnapshot(
        agent=AgentName.ORACLE,
        ts=ASOF,
        cash=1000.0,
        positions=[
            Position(
                agent=AgentName.ORACLE, ticker="NVDA", qty=10, avg_cost=100.0, last_price=100.0
            )
        ],
    )
    ctx = MarketContext(
        asof=ASOF, universe=["NVDA"], book=book, prices=object(), extras={"theses": []}
    )
    decisions = await OracleAgent().generate_decisions(ctx)
    assert len(decisions) == 1
    assert decisions[0].ticker == "NVDA" and decisions[0].target_weight == 0.0


async def test_agent_with_no_theses_and_no_holdings_does_nothing():
    """Cash is an acceptable answer: an empty research run must not manufacture trades."""
    book = BookSnapshot(agent=AgentName.ORACLE, ts=ASOF, cash=30000.0, positions=[])
    ctx = MarketContext(
        asof=ASOF, universe=["AAPL"], book=book, prices=object(), extras={"theses": []}
    )
    assert await OracleAgent().generate_decisions(ctx) == []


# --------------------------------------------------------------- fake LLM client
class _Usage:
    input_tokens = output_tokens = 10
    cache_creation_input_tokens = cache_read_input_tokens = 0


class _TU:
    type = "tool_use"

    def __init__(self, name, inp, id):
        self.name, self.input, self.id = name, inp, id


class _Resp:
    def __init__(self, blocks):
        self.content, self.usage = blocks, _Usage()


class _Msgs:
    """Replays a script of responses; the last entry repeats (so a forced/repair call still
    gets an answer)."""

    def __init__(self, script):
        self._script = list(script)

    async def create(self, **kw):
        return _Resp(self._script.pop(0) if len(self._script) > 1 else self._script[-1])


class _Client:
    def __init__(self, script):
        self.messages = _Msgs(script)


def _thesis_input(thesis_text):
    return {
        "rating": "buy",
        "conviction": 0.6,
        "target_weight": 0.08,
        "thesis": thesis_text,
        "key_risks": ["valuation"],
        "invalidation_triggers": ["margin decline"],
        "horizon_days": 90,
        "evidence_basis": ["valuation ratios"],
    }


@pytest.fixture
def _stub_data(monkeypatch):
    def _row(fy, filed):
        base = {name: None for name, _, _ in f._LINE_ITEMS}
        return base | {
            "fy": fy,
            "end": f"{fy}-12-31",
            "filed": filed,
            "form": "10-K",
            "revenue": 1000.0,
            "net_income": 250.0,
            "operating_cash_flow": 300.0,
            "capital_expenditures": 100.0,
            "diluted_shares": 10_000_000.0,
        }

    monkeypatch.setattr(
        f,
        "financials",
        lambda ticker, years=5, **k: {
            "ticker": ticker,
            "fetched_at": "2026-07-21T00:00:00+00:00",
            "fiscal_years": [_row(2025, "2026-02-15"), _row(2024, "2025-02-15")],
            "concepts_used": {},
            "missing_concepts": [],
        },
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-20"]),
            "ticker": ["AAPL"],
            "open": [99.0],
            "high": [101.0],
            "low": [98.0],
            "close": [100.0],
            "volume": [1e7],
        }
    )
    monkeypatch.setattr(G.run_screen.__globals__["market"], "get_history", lambda *a, **k: bars)

    async def fake_ratios(ctx, ticker, **k):
        return {"available": True, "ticker": ticker, "valuation": {"pe": 18.5}}

    monkeypatch.setitem(G.TOOLS, "get_ratios", fake_ratios)


async def test_graph_builds_thesis_on_valid_citation(_stub_data):
    script = [
        [_TU("get_ratios", {"ticker": "AAPL"}, "u1")],
        [_TU("submit_thesis", _thesis_input("P/E is 18.5 [T1]. Quality is high."), "u2")],
    ]
    lm = AnalystLLM(model="claude-sonnet-4-6", client=_Client(script))
    res = await G.run_graph(ASOF, held=[], llm=lm, universe=["AAPL"], top_n=1, concurrency=1)
    assert len(res.theses) == 1 and not res.dropped
    th = res.theses[0]
    assert th.tool_trace and th.citations
    assert any("narrative: not yet available" in b for b in th.evidence_basis)  # the 5.1 seam


async def test_graph_drops_name_when_citation_never_resolves(_stub_data):
    """Cites [T9], which was never recorded; the repair round-trip repeats the mistake, so the
    ticker is dropped — no thesis means no position."""
    bad = _thesis_input("P/E is 18.5 [T9].")
    script = [
        [_TU("get_ratios", {"ticker": "AAPL"}, "u1")],
        [_TU("submit_thesis", bad, "u2")],
        [_TU("submit_thesis", bad, "u3")],
    ]
    lm = AnalystLLM(model="claude-sonnet-4-6", client=_Client(script))
    res = await G.run_graph(ASOF, held=[], llm=lm, universe=["AAPL"], top_n=1, concurrency=1)
    assert not res.theses
    assert res.dropped and res.dropped[0]["ticker"] == "AAPL"
    assert "citation" in res.dropped[0]["reason"]


async def test_graph_repair_round_trip_can_rescue_a_thesis(_stub_data):
    """First submission omits the tag (hard fail); the repair adds it and the thesis survives."""
    script = [
        [_TU("get_ratios", {"ticker": "AAPL"}, "u1")],
        [_TU("submit_thesis", _thesis_input("P/E is 18.5."), "u2")],  # no tag -> hard fail
        [_TU("submit_thesis", _thesis_input("P/E is 18.5 [T1]."), "u3")],  # repaired
    ]
    lm = AnalystLLM(model="claude-sonnet-4-6", client=_Client(script))
    res = await G.run_graph(ASOF, held=[], llm=lm, universe=["AAPL"], top_n=1, concurrency=1)
    assert len(res.theses) == 1 and not res.dropped
