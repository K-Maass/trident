"""ORACLE cost metering, model-string validation, and the budget governor's fail-safes.
No network: a fake client plus a monkeypatched health-event writer."""

from __future__ import annotations

import json

import pytest

from trident.agents.analyst.cost import CostMeter
from trident.agents.analyst.llm import ModelValidationError, validate_models


class _Usage:
    def __init__(self, i=0, o=0, cw=0, cr=0):
        self.input_tokens, self.output_tokens = i, o
        self.cache_creation_input_tokens, self.cache_read_input_tokens = cw, cr


class _Model:
    def __init__(self, mid):
        self.id = mid


class _Models:
    def __init__(self, known):
        self._known = known

    async def retrieve(self, mid):
        if mid not in self._known:
            raise KeyError(mid)
        return _Model(mid)

    def list(self):
        return [_Model(k) for k in self._known]


class _Client:
    def __init__(self, known):
        self.models = _Models(known)


# ------------------------------------------------------------------ cost meter
def test_cost_meter_prices_per_tier():
    m = CostMeter()
    m.record("claude-sonnet-4-6", _Usage(i=1_000_000, o=1_000_000))
    assert m.dollars() == pytest.approx(18.0)  # sonnet: $3 in + $15 out per MTok
    assert m.calls == 1


def test_cost_meter_summary_is_json_clean_and_zero_by_default():
    s = CostMeter().summary()
    json.dumps(s)  # must not raise
    assert s == {"calls": 0, "estimated_usd": 0.0, "by_model": {}}


def test_unknown_model_tier_is_priced_at_the_most_expensive_rate():
    """Erring expensive makes the budget governor trip EARLY on an unrecognised model rather
    than silently under-counting real spend."""
    m = CostMeter()
    m.record("some-future-model", _Usage(i=1_000_000))
    assert m.dollars() == pytest.approx(5.0)


# ------------------------------------------------------------- model validation
async def test_valid_model_ids_pass():
    known = ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    await validate_models(*known, client=_Client(known))


async def test_unknown_model_id_aborts_startup_and_lists_valid_ids():
    with pytest.raises(ModelValidationError) as ei:
        await validate_models("claude-made-up", client=_Client(["claude-sonnet-4-6"]))
    assert "claude-sonnet-4-6" in str(ei.value)  # the error lists the real ids


# --------------------------------------------------- cost event on a zero run
async def test_cost_event_written_even_when_run_produces_zero_theses(monkeypatch):
    from trident.data import store
    from trident.orchestration import oracle

    events = []

    async def fake_write(ev):
        events.append(ev)

    async def fake_spend():
        return 999.0  # over the hard ceiling -> research refused, zero theses

    monkeypatch.setattr(store, "write_health_event", fake_write)
    monkeypatch.setattr(oracle, "_month_to_date_spend", fake_spend)

    res = await oracle.run_oracle(broker=None, dry_run=True, validate=False)
    assert res.refused and res.n_theses == 0
    cost_events = [e for e in events if e.source == "oracle_cost"]
    assert len(cost_events) == 1
    assert json.loads(cost_events[0].detail)["estimated_usd"] == 0.0
    # WARN, never CRITICAL: a CRITICAL would trip _global_freeze() and halt QUANT + SENTINEL.
    assert all(e.severity.value != "critical" for e in events)
