"""Near-miss instrumentation (pre-registered 2026-07-21 deploy) — classification is correct and
it never touches decisions. Pure logic; the health-event write is mocked."""

from __future__ import annotations

from trident.orchestration import sentinel as S


async def test_near_miss_classification(monkeypatch):
    events = []

    async def fake_health(ev):
        events.append(ev)

    monkeypatch.setattr(S.store, "write_health_event", fake_health)

    signals = {
        "JOINT": {
            "sent_z": 2.0,
            "attention_z": 1.5,
            "sources_used": ["finnhub_news", "rss"],
            "apewisdom_rank": 10,
        },
        "SENTONLY": {
            "sent_z": 2.2,
            "attention_z": 0.3,
            "sources_used": ["finnhub_news"],
            "apewisdom_rank": None,
        },
        "ATTNONLY": {
            "sent_z": 0.1,
            "attention_z": 3.0,
            "sources_used": ["finnhub_news", "rss"],
            "apewisdom_rank": 5,
        },
        "NEITHER": {"sent_z": 0.0, "attention_z": 0.0, "sources_used": ["finnhub_news"]},
        "CLIP": {"sent_z": None, "attention_z": 5.0, "sources_used": ["finnhub_news"]},  # clip pin
    }
    summ = await S.log_run_diagnostics(signals)

    assert summ["joint_fires"] == 1
    assert summ["sent_only"] == ["SENTONLY(sz=2.20,az=0.30)"]
    assert summ["attn_only"] == ["ATTNONLY(sz=0.10,az=3.00)"]
    assert (
        summ["src_rss"] == 2 and summ["src_ape"] == 2
    )  # rss on JOINT+ATTNONLY; ape on JOINT+ATTNONLY
    assert summ["clip_pins"] == 1  # CLIP's attention_z == 5.0
    assert len(events) == 1 and events[0].source == "sentinel_nearmiss"


async def test_diagnostics_never_raises_on_empty(monkeypatch):
    async def fake_health(ev):
        pass

    monkeypatch.setattr(S.store, "write_health_event", fake_health)
    summ = await S.log_run_diagnostics({})
    assert summ["joint_fires"] == 0 and summ["n"] == 0
