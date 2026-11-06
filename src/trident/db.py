"""Shared async SQLAlchemy engine — one connection pool for the whole process.

Both the data layer (data/store.py) and the money-path persistence (persistence.py)
use this so there is a single engine bound to the running event loop.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from trident.config import get_settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Lazy singleton engine bound to the running event loop."""
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.require("database_url")
        _engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return _engine


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
