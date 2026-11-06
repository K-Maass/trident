"""ORACLE orchestration — the thesis persistence key.

Two runs a moment apart must not leave two thesis rows. The unit test proves the key generator
collapses same-day timestamps; the integration test proves the FULL path (helper + the live-shape
unique index) yields exactly one row on a re-run — a test that actively constructs the failure
mode (two different microsecond-level timestamps) rather than inferring idempotency from the
index looking right.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from trident import persistence
from trident.data import store
from trident.models import AgentName, InvestmentThesis, Rating
from trident.orchestration.oracle import _session_asof


# ------------------------------------------------------------------ unit (no DB)
def test_session_asof_collapses_same_day_and_separates_days():
    t1 = datetime(2026, 7, 24, 13, 30, 5, 111111, tzinfo=UTC)
    t2 = datetime(2026, 7, 24, 13, 30, 5, 999999, tzinfo=UTC)  # same UTC day, µs apart
    t3 = datetime(2026, 7, 25, 0, 0, 0, 1, tzinfo=UTC)  # next UTC day
    key = _session_asof(t1)
    assert key == _session_asof(t2)  # a re-run within the session shares the key
    assert key != _session_asof(t3)  # a different session gets a different key
    assert (key.hour, key.minute, key.second, key.microsecond) == (0, 0, 0, 0)
    assert key.tzinfo == UTC


# --------------------------------------------------------------- integration (DB)
def _thesis(asof, *, conviction: float, rating: str, ticker: str = "AAPL") -> InvestmentThesis:
    return InvestmentThesis(
        asof=asof,
        ticker=ticker,
        rating=Rating(rating),
        conviction=conviction,
        target_weight=0.05,
        thesis=f"probe [{rating}]",
        key_risks=["r"],
        invalidation_triggers=["i"],
        horizon_days=60,
        tool_trace={"T1": {"id": "T1", "tool": "get_ratios", "result": {}}},
        evidence_basis=["valuation ratios"],
    )


async def _clear(ticker: str = "AAPL") -> None:
    eng = store.get_engine()
    async with eng.begin() as c:
        name = (await c.execute(text("SELECT current_database()"))).scalar_one()
        assert str(name).endswith("_test"), f"refusing DELETE on non-test DB '{name}'"
        await c.execute(
            text("DELETE FROM theses WHERE agent_id='oracle' AND ticker=:t"), {"t": ticker}
        )


@pytest.mark.integration
async def test_microsecond_apart_runs_upsert_to_one_row():
    """THE prove-it test. Two runs whose real timestamps differ by microseconds normalize to the
    same session key, so the second upsert must UPDATE the first, not insert a duplicate. If
    _session_asof were the identity (raw now()), the stamps would differ and this asserts 2 rows —
    which is exactly the failure the migration exists to prevent."""
    await _clear()
    try:
        t1 = datetime(2026, 7, 24, 13, 30, 5, 111111, tzinfo=UTC)
        t2 = datetime(2026, 7, 24, 13, 30, 5, 999999, tzinfo=UTC)
        assert _session_asof(t1) == _session_asof(t2)

        await persistence.upsert_thesis(_thesis(_session_asof(t1), conviction=0.60, rating="hold"))
        await persistence.upsert_thesis(_thesis(_session_asof(t2), conviction=0.70, rating="buy"))

        rows = [r for r in await persistence.get_theses(AgentName.ORACLE) if r["ticker"] == "AAPL"]
        assert len(rows) == 1  # the re-run upserted, it did not duplicate
        assert float(rows[0]["conviction"]) == 0.70  # ...and the second run's fields won
        assert rows[0]["rating"] == "buy"

        # cross-session control: a DIFFERENT day IS a new row (the key discriminates, too)
        await persistence.upsert_thesis(
            _thesis(_session_asof(t1.replace(day=25)), conviction=0.55, rating="hold")
        )
        rows = [r for r in await persistence.get_theses(AgentName.ORACLE) if r["ticker"] == "AAPL"]
        assert len(rows) == 2
    finally:
        await _clear()
        await store.dispose_engine()
