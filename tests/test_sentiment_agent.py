"""SENTINEL decision-rule tests — entries, exits, knife guard, max-6 cap, rationale/item-links.

Offline: synthetic signal dicts + a hand-built book; no DB/broker. Exercises exactly the doc-04
rules the agent must obey.
"""

from __future__ import annotations

from datetime import UTC, datetime

from trident.agents.base import MarketContext
from trident.agents.sentiment import sentinel_agent
from trident.models import AgentName, BookSnapshot, Position


def _ctx(signals, positions=None, holding_days=None, cash=30_000.0):
    book = BookSnapshot(
        agent=AgentName.SENTINEL, ts=datetime.now(UTC), cash=cash, positions=positions or []
    )
    return MarketContext(
        asof=datetime.now(UTC),
        universe=list(signals),
        book=book,
        prices=None,
        extras={"signals": signals, "holding_days": holding_days or {}},
    )


def _pos(ticker):
    return Position(agent=AgentName.SENTINEL, ticker=ticker, qty=10, avg_cost=100, last_price=100)


async def test_enters_only_when_all_conditions_met():
    # all mature (immature=False) so this isolates the z/attention/evidence gates
    sigs = {
        "AAA": {  # passes: strong z + attention + 6 items
            "sent_z": 2.0,
            "attention_z": 1.5,
            "n_evidence": 6,
            "apewisdom_rank": None,
            "immature": False,
            "top_items": [{"title": "t", "source": "finnhub", "score": 0.5, "url": "u"}],
        },
        "LOWZ": {"sent_z": 1.0, "attention_z": 2.0, "n_evidence": 9, "immature": False},  # z<1.5
        "LOWA": {"sent_z": 2.0, "attention_z": 0.5, "n_evidence": 9, "immature": False},  # attn<1
        "THIN": {"sent_z": 2.5, "attention_z": 2.0, "n_evidence": 2, "immature": False},  # thin
        "APE": {
            "sent_z": 3.0,
            "attention_z": 2.0,
            "n_evidence": 3,
            "apewisdom_rank": 5,
            "immature": False,
        },  # 3 items + top-20 ape
    }
    ds = await sentinel_agent().generate_decisions(_ctx(sigs))
    entered = {d.ticker: d.target_weight for d in ds}
    assert set(entered) == {"AAA", "APE"}
    assert all(w == 0.06 for w in entered.values())


async def test_never_catches_knives():
    # mature, so the REFUSAL is attributable to the knife guard, not the maturity gate
    sigs = {"FALL": {"sent_z": -2.5, "attention_z": 3.0, "n_evidence": 20, "immature": False}}
    ds = await sentinel_agent().generate_decisions(_ctx(sigs))
    assert ds == []  # knife → no buy, no decision


async def test_immature_baseline_blocks_entry_even_when_all_gates_clear():
    """The violation test the dry-run could not be: a signal that clears BOTH z-gates AND the
    evidence floor but sits on a <21-session baseline must NOT open a position. A z off a
    10-observation baseline is not trustworthy enough to enter on."""
    sigs = {
        "RIPE": {"sent_z": 3.0, "attention_z": 2.0, "n_evidence": 9, "immature": False},
        "GREEN": {"sent_z": 3.0, "attention_z": 2.0, "n_evidence": 9, "immature": True},
    }
    ds = await sentinel_agent().generate_decisions(_ctx(sigs))
    entered = {d.ticker for d in ds}
    assert entered == {"RIPE"}  # GREEN is identical but immature → refused


async def test_missing_immature_flag_fails_closed():
    """Fail-safe direction (golden rule 8): a candidate that clears both z-gates AND the evidence
    floor but is MISSING the `immature` key entirely must be REFUSED, not entered. Locks in the
    `sig.get("immature", True)` default so a future refactor can't silently reopen the gap — a
    dropped flag is a data anomaly, and on the open-a-position path ambiguity means no."""
    sigs = {"NOFLAG": {"sent_z": 3.0, "attention_z": 2.0, "n_evidence": 9}}  # no `immature` key
    ds = await sentinel_agent().generate_decisions(_ctx(sigs))
    assert ds == []  # absent flag → treated as immature → no entry


async def test_immature_never_blocks_a_risk_reducing_exit():
    """The maturity gate is entry-only. A held name whose coverage has regressed to immature and
    whose sentiment has flipped negative must STILL be exitable — never trap a position because
    its baseline got thin."""
    held = [_pos("STUCK")]
    sigs = {"STUCK": {"sent_z": -0.5, "attention_z": 0.0, "n_evidence": 1, "immature": True}}
    ds = await sentinel_agent().generate_decisions(
        _ctx(sigs, positions=held, holding_days={"STUCK": 3})
    )
    assert len(ds) == 1
    assert ds[0].ticker == "STUCK" and ds[0].target_weight == 0.0
    assert ds[0].rationale["reason"] == "sent_z<0"  # exit fired despite immaturity


async def test_exit_on_negative_z_and_max_hold():
    held = [_pos("FLIP"), _pos("OLD"), _pos("KEEP")]
    sigs = {
        "FLIP": {"sent_z": -0.3, "attention_z": 0.0, "n_evidence": 1},
        "OLD": {"sent_z": 2.0, "attention_z": 2.0, "n_evidence": 9},  # still positive but 10d
        "KEEP": {"sent_z": 1.9, "attention_z": 2.0, "n_evidence": 9},
    }
    ds = await sentinel_agent().generate_decisions(
        _ctx(sigs, positions=held, holding_days={"OLD": 10, "FLIP": 3, "KEEP": 3})
    )
    exits = {d.ticker: d.rationale["reason"] for d in ds if d.target_weight == 0.0}
    assert exits == {"FLIP": "sent_z<0", "OLD": "max_hold_10d"}


async def test_max_six_positions_ranked_by_sent_z():
    sigs = {
        f"T{i}": {"sent_z": 2.0 + i * 0.1, "attention_z": 2.0, "n_evidence": 9, "immature": False}
        for i in range(8)
    }
    ds = await sentinel_agent().generate_decisions(_ctx(sigs))
    picks = sorted(d.ticker for d in ds)
    assert len(ds) == 6
    assert picks == ["T2", "T3", "T4", "T5", "T6", "T7"]  # the 6 highest sent_z


async def test_entry_rationale_has_item_links():
    sigs = {
        "AAA": {
            "sent_z": 2.0,
            "attention_z": 1.5,
            "n_evidence": 6,
            "immature": False,
            "sources_used": ["finnhub_news", "rss"],
            "top_items": [
                {"title": "Beat", "source": "finnhub", "score": 0.7, "url": "http://n/1"}
            ],
        }
    }
    ds = await sentinel_agent().generate_decisions(_ctx(sigs))
    r = ds[0].rationale
    assert r["top_items"][0]["url"] == "http://n/1"
    assert r["sources_used"] == ["finnhub_news", "rss"]
    assert r  # non-empty (golden rule 2)
