"""Route integration tests to a disposable database ending in `_test`.

Integration fixtures clear tables. They require TRIDENT_TEST_DATABASE_URL and never use
the application DATABASE_URL.
"""

from __future__ import annotations

import os
from pathlib import Path

import asyncpg
import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from trident import db

SCHEMA = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
_prepared = False

# Transactional tables cleared for a fresh slate each run (agents are kept + re-seeded).
_DATA_TABLES = (
    "fills",
    "orders",
    "decisions",
    "positions",
    "equity_snapshots",
    "signals",
    "health_events",
    "model_registry",
    "news_items",
    "relevance_labels",
    "news_url_collisions",
    "theses",
)


def require_test_db_url() -> str:
    """The dedicated test DB URL, or a HARD failure. No fallback to DATABASE_URL by design."""
    url = os.environ.get("TRIDENT_TEST_DATABASE_URL")
    if not url:
        pytest.fail(
            "Integration tests require TRIDENT_TEST_DATABASE_URL — a DEDICATED throwaway database "
            "whose name ends in '_test'. These tests DELETE rows, so they must NEVER run against "
            "the live DATABASE_URL; there is no fallback by design. Example:\n"
            "  TRIDENT_TEST_DATABASE_URL=postgresql+asyncpg://trident:<password>@localhost:5432/trident_test",
            pytrace=False,
        )
    name = url.rsplit("/", 1)[-1].split("?")[0]
    if not name.endswith("_test"):
        pytest.fail(
            f"TRIDENT_TEST_DATABASE_URL points at database '{name}', which does not end in "
            "'_test'. Refusing — this guard stops a destructive test from hitting a live DB.",
            pytrace=False,
        )
    return url


async def _prepare(url: str) -> None:
    """Apply schema fresh + a clean-slate truncate + a dedicated 'test' agent, once per run.

    Uses asyncpg directly so the whole schema.sql runs as one multi-statement script (comment-
    and DDL-aware, like psql) rather than a fragile split-on-';' that trips over semicolons in
    comments."""
    conn = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        await conn.execute(SCHEMA.read_text())  # CREATE ... IF NOT EXISTS + seed the real agents
        await conn.execute(f"TRUNCATE {', '.join(_DATA_TABLES)} RESTART IDENTITY CASCADE")
        await conn.execute(
            "INSERT INTO agents (id, kind, start_cash) VALUES ('test','test_harness',30000) "
            "ON CONFLICT (id) DO NOTHING"  # fixtures MAY use this; NOT the protection layer
        )
        await conn.execute("UPDATE agents SET status='active', high_water_mark=0")
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
async def _isolate_integration_db(request):
    """Route integration tests onto the ephemeral test DB by overriding the engine singleton.
    Non-integration tests fall straight through untouched."""
    if request.node.get_closest_marker("integration") is None:
        yield
        return
    global _prepared
    url = require_test_db_url()
    if not _prepared:
        await _prepare(url)
        _prepared = True
    await db.dispose_engine()
    db._engine = create_async_engine(url, pool_pre_ping=True)
    try:
        yield
    finally:
        await db.dispose_engine()


@pytest.fixture(autouse=True)
def _block_health_writes_in_unit_tests(request, monkeypatch):
    """Stub the health writer in unit tests so failure paths never reach Postgres."""
    if request.node.get_closest_marker("integration") is not None:
        return  # integration tests are routed to the test DB by _isolate_integration_db above
    from trident.data import store

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(store, "write_health_event", _noop)
