"""ORACLE's Anthropic client seam (doc 05) — the single metered door to the LLM.

Follows agents/sentiment/relevance.py where it applies: function-local `import anthropic`,
settings.require("anthropic_api_key"), AsyncAnthropic(max_retries=6, timeout=60), an injectable
`client=None` seam so unit tests never touch the network, and inter-call throttling only for the
real client.

Two invariants:
  1. EVERY call goes through AnalystLLM.create — no path to messages.create skips cost
     accounting or the temperature-0 determinism pin ("from the very first API call").
  2. validate_models fails loud on an unknown model id BEFORE any research runs, listing the
     valid ids from the Models API. It NEVER renames the configured strings.
"""

from __future__ import annotations

import asyncio

from trident.agents.analyst.cost import CostMeter
from trident.config import get_settings

MAX_RETRIES = 6  # SDK-level exponential backoff on 429 / connection errors
CLIENT_TIMEOUT = 60.0
THROTTLE_SECONDS = 0.25  # gentle inter-call pacing for the real client (pool-friendly)


class ModelValidationError(RuntimeError):
    """A configured model id is not a valid id today (startup drift guard)."""


def _build_client():
    settings = get_settings()
    settings.require("anthropic_api_key")
    import anthropic

    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, max_retries=MAX_RETRIES, timeout=CLIENT_TIMEOUT
    )


async def _valid_ids(client) -> list[str]:
    """Best-effort list of valid model ids from the Models API (for the error message only)."""
    ids: list[str] = []
    try:
        page = client.models.list()
        if hasattr(page, "__aiter__"):
            async for m in page:
                ids.append(getattr(m, "id", str(m)))
        else:
            if asyncio.iscoroutine(page):
                page = await page
            for m in page:
                ids.append(getattr(m, "id", str(m)))
    except Exception:  # noqa: BLE001 — the list is a courtesy in the error, never load-bearing
        pass
    return ids


async def _aclose(client) -> None:
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if close is None:
        return
    try:
        res = close()
        if asyncio.iscoroutine(res):
            await res
    except Exception:  # noqa: BLE001
        pass


async def validate_models(*model_ids: str, client=None) -> None:
    """Raise ModelValidationError if any configured id is unknown. Injectable client for tests."""
    own = client is None
    client = client or _build_client()
    try:
        for mid in model_ids:
            try:
                await client.models.retrieve(mid)
            except Exception as e:  # noqa: BLE001 — any failure to resolve = treat as invalid
                valid = await _valid_ids(client)
                raise ModelValidationError(
                    f"configured model id '{mid}' is not valid today. "
                    f"Valid ids: {valid or '(could not list)'}. "
                    "Fix the config string (oracle_model / bulk_model); do NOT rename it here."
                ) from e
    finally:
        if own:
            await _aclose(client)


class AnalystLLM:
    """Metered wrapper over one AsyncAnthropic model. All research traffic flows through
    `create`; the CostMeter and the temperature-0 pin live here so no call site bypasses them."""

    def __init__(self, model: str | None = None, *, client=None, meter: CostMeter | None = None):
        settings = get_settings()
        self.model = model or settings.oracle_model
        self._client = client
        self._own_client = client is None  # a test's fake client is never throttled or closed
        self.meter = meter if meter is not None else CostMeter()

    def _ensure(self):
        if self._client is None:
            self._client = _build_client()
        return self._client

    async def create(self, **kwargs):
        """One metered messages.create. temperature=0 is pinned here (same context -> same
        decisions); callers pass system/tools/messages only."""
        client = self._ensure()
        kwargs.setdefault("temperature", 0)
        resp = await client.messages.create(model=self.model, **kwargs)
        self.meter.record(self.model, getattr(resp, "usage", None) or _ZeroUsage())
        if self._own_client:
            await asyncio.sleep(THROTTLE_SECONDS)
        return resp

    async def aclose(self) -> None:
        if self._own_client and self._client is not None:
            await _aclose(self._client)


class _ZeroUsage:
    input_tokens = output_tokens = cache_creation_input_tokens = cache_read_input_tokens = 0
