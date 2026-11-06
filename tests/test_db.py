"""Phase 0 acceptance: the decisions table can be written to and read back.

This is an *integration* test — it needs Postgres up (`make up && make db-init`).
CI skips it (`pytest -m "not integration"`); local `make test` runs it.

It round-trips a real `Decision` model through the schema to catch model↔DDL
drift early (golden rule 2: every decision is logged before any order).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from trident.db import get_engine
from trident.models import AgentName, Decision

pytestmark = pytest.mark.integration

_INSERT = text(
    """
    INSERT INTO decisions
        (id, agent_id, asof, ticker, target_weight, horizon_days, rationale, inputs_digest)
    VALUES
        (:id, :agent_id, :asof, :ticker, :target_weight, :horizon_days,
         CAST(:rationale AS JSONB), :inputs_digest)
    """
)
_SELECT = text(
    "SELECT agent_id, ticker, target_weight, horizon_days, rationale FROM decisions WHERE id = :id"
)
_DELETE = text("DELETE FROM decisions WHERE id = :id")


async def test_decisions_insert_and_read_back():
    engine = get_engine()

    decision = Decision(
        agent=AgentName.QUANT,
        asof=datetime.now(UTC),
        ticker="AAPL",
        target_weight=0.05,
        horizon_days=5,
        rationale={"test": "phase0_db_roundtrip", "top_features": ["mom_12_1", "rsi_2"]},
        inputs_digest=Decision.digest({"phase": 0}),
    )
    params = {
        "id": decision.id,
        "agent_id": decision.agent.value,
        "asof": decision.asof,
        "ticker": decision.ticker,
        "target_weight": decision.target_weight,
        "horizon_days": decision.horizon_days,
        "rationale": json.dumps(decision.rationale),
        "inputs_digest": decision.inputs_digest,
    }

    try:
        # Write and commit.
        async with engine.begin() as conn:
            await conn.execute(_INSERT, params)

        # Read back on a *fresh* transaction — proves durability, not read-your-write.
        async with engine.connect() as conn:
            row = (await conn.execute(_SELECT, {"id": decision.id})).mappings().one()

        assert row["agent_id"] == "quant"
        assert row["ticker"] == "AAPL"
        assert float(row["target_weight"]) == pytest.approx(0.05)
        assert row["horizon_days"] == 5
        # JSONB survives the round-trip intact (rationale is never empty — golden rule 2).
        assert row["rationale"]["test"] == "phase0_db_roundtrip"
        assert row["rationale"]["top_features"] == ["mom_12_1", "rsi_2"]
    finally:
        async with engine.begin() as conn:
            await conn.execute(_DELETE, {"id": decision.id})
