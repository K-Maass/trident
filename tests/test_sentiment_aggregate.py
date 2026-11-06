"""SENTINEL aggregation tests — the signal math + the mandatory look-ahead tripwire.

Offline: pure helpers and hand-built DataFrames, no DB/network. The load-bearing test is
`test_session_metrics_excludes_future_items` — a session's aggregate must be byte-for-byte
identical when items published AFTER the session cutoff are added, or the signal peeks ahead.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from trident.agents.sentiment import aggregate as A


def _mk(ticker, source, pub, sent, hid):
    return {
        "url_hash": hid,
        "source": source,
        "ticker": ticker,
        "title": f"title {hid}",
        "body": None,
        "url": f"http://x/{hid}",
        "published_at": pub,
        "relevant": True,
        "sentiment": sent,
    }


def test_decay_half_life_24h():
    assert A.decay_weight(0) == 1.0
    assert A.decay_weight(24) == 0.5
    assert A.decay_weight(48) == 0.25


def test_source_key_mapping():
    assert A.source_key("finnhub") == "finnhub_news"
    assert A.source_key("rss:cnbc") == "rss"
    assert A.source_key("tradestie_wsb") == "tradestie_wsb"


def test_sentiment_source_weights_excludes_disabled_and_attention():
    reg = {
        "finnhub_news": {"enabled": True, "kind": "raw_text", "weight": 1.0},
        "rss": {"enabled": True, "kind": "raw_text", "weight": 0.8},
        "apewisdom": {"enabled": True, "kind": "attention_only", "weight": 1.0},
        "tradestie_wsb": {"enabled": False, "kind": "pre_scored", "weight": 0.5},
    }
    assert A.sentiment_source_weights(reg) == {"finnhub_news": 1.0, "rss": 0.8}


def test_combine_sources_renormalizes_over_present():
    w = {"finnhub_news": 1.0, "rss": 0.8, "tradestie_wsb": 0.5}
    # only finnhub + rss present → weights renormalize over just those two
    val, used = A.combine_sources({"finnhub_news": 0.5, "rss": 0.0}, w)
    assert math.isclose(val, (1.0 * 0.5 + 0.8 * 0.0) / 1.8)
    assert used == ["finnhub_news", "rss"]


def test_combine_sources_empty_when_nothing_present():
    val, used = A.combine_sources({"tradestie_wsb": float("nan")}, {"finnhub_news": 1.0})
    assert math.isnan(val) and used == []


def test_trailing_z_detects_spike_and_flags_immaturity():
    base = [0.0, 0.1, -0.1, 0.05, -0.05, 0.0, 0.1, -0.1, 0.05, -0.05]
    s = pd.Series(base + [0.8])
    z, count = A.trailing_z(s)  # window=21, min_periods=10
    assert z.iloc[-1] > 2.0  # 0.8 is a big spike vs the ~0 baseline
    assert int(count.iloc[-1]) == 10  # 10 prior sessions → still immature (<21)


def test_trailing_z_zero_variance_baseline_is_nan():
    s = pd.Series([0.1] * 10 + [0.9])  # flat baseline → std 0 → undefined z
    z, _ = A.trailing_z(s)
    assert math.isnan(z.iloc[-1])


def test_std_floor_damps_near_zero_baseline_and_clips():
    """The attention_z=20.8 pathology: a near-flat count baseline dividing a jump by a ~0.3 std.
    Unfloored, a 4-item day on a mostly-dark ticker scores ~13 sigma; floored it is a sane ~3.9.
    Z_CLIP bounds whatever still survives (a big enough jump saturates the clip either way)."""
    s = pd.Series([0.0] * 9 + [1.0, 4.0])  # baseline mean 0.1, std ≈ 0.3 → naive z ≈ 13
    unfloored, _ = A.trailing_z(s)
    floored, _ = A.trailing_z(s, std_floor=A.COUNT_STD_FLOOR, poisson_floor=True)
    assert unfloored.iloc[-1] == A.Z_CLIP  # naive z blows past the clip and is only bounded by it
    assert floored.iloc[-1] == pytest.approx((4.0 - 0.1) / A.COUNT_STD_FLOOR)  # ≈3.9, sub-clip
    assert floored.iloc[-1] < unfloored.iloc[-1]  # the floor does real work below the clip
    assert abs(floored.iloc[-1]) <= A.Z_CLIP


def test_sent_std_floor_blocks_fabricated_entry():
    """The money-relevant case: a well-covered ticker with flat, neutral sentiment. A
    noise-sized wobble must NOT clear SENT_Z_ENTER (1.5) just because the baseline std is ~1e-3."""
    baseline = [0.020, 0.021, 0.019, 0.020, 0.022, 0.018, 0.021, 0.019, 0.020, 0.021]
    s = pd.Series(baseline + [0.045])  # a +0.025 wobble — noise, not news
    unfloored, _ = A.trailing_z(s)
    floored, _ = A.trailing_z(s, std_floor=A.SENT_STD_FLOOR)
    assert unfloored.iloc[-1] > 1.5  # unfloored, this noise fabricates an entry
    assert floored.iloc[-1] < 1.5  # floored, it does not


def test_sent_std_floor_still_admits_a_genuine_spike():
    """The floor must not silently kill real signal: a true sentiment surge still clears 1.5."""
    baseline = [0.020, 0.021, 0.019, 0.020, 0.022, 0.018, 0.021, 0.019, 0.020, 0.021]
    s = pd.Series(baseline + [0.60])  # genuinely strong positive news
    floored, _ = A.trailing_z(s, std_floor=A.SENT_STD_FLOOR)
    assert floored.iloc[-1] > 1.5


def test_poisson_floor_scales_with_magnitude():
    """ApeWisdom mentions run from single digits to hundreds, so a single ABSOLUTE floor is wrong
    at both ends. The sqrt-mean (Poisson) floor damps the low-mention ticker — where a 2.5→6 jump
    is just count noise — while leaving a high-mention ticker's own std to govern."""
    small = pd.Series([2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 6.0])  # mean 2.5, std .5
    big = pd.Series([200.0, 300.0] * 5 + [600.0])  # mean 250, std 50
    z_small_abs, _ = A.trailing_z(small, std_floor=A.MENTIONS_STD_FLOOR)  # fixed floor only
    z_small, _ = A.trailing_z(small, std_floor=A.MENTIONS_STD_FLOOR, poisson_floor=True)
    z_big, _ = A.trailing_z(big, std_floor=A.MENTIONS_STD_FLOOR, poisson_floor=True)
    # small: own std .5 → fixed floor 1.0 → z 3.5; Poisson floor √2.5≈1.58 damps it further.
    assert z_small.iloc[-1] < z_small_abs.iloc[-1]
    # big: own std 50 already dwarfs its Poisson floor (√250≈15.8) → unfloored, std governs.
    assert z_big.iloc[-1] == pytest.approx(min((600.0 - 250.0) / 50.0, A.Z_CLIP))
    assert abs(z_small.iloc[-1]) <= A.Z_CLIP and abs(z_big.iloc[-1]) <= A.Z_CLIP


def test_session_metrics_excludes_future_items():
    tref = datetime(2026, 7, 10, 23, 59, 59, tzinfo=UTC)
    weights = {"finnhub_news": 1.0, "rss": 0.8}
    ape = A._ape_df([])
    base = [
        _mk("AAA", "finnhub", tref - timedelta(hours=2), 0.5, "a"),
        _mk("AAA", "finnhub", tref - timedelta(hours=10), -0.2, "b"),
    ]
    future = base + [_mk("AAA", "finnhub", tref + timedelta(hours=5), 0.9, "c")]
    m_base = A._session_metrics(A._news_df(base), ape, tref, weights)
    m_future = A._session_metrics(A._news_df(future), ape, tref, weights)
    assert m_base["sent_raw"] == m_future["sent_raw"]
    assert m_base["n_evidence"] == m_future["n_evidence"] == 2


def test_session_metrics_decay_weights_recent_more():
    tref = datetime(2026, 7, 10, 23, 59, 59, tzinfo=UTC)
    weights = {"finnhub_news": 1.0}
    ape = A._ape_df([])
    items = [
        _mk("AAA", "finnhub", tref - timedelta(hours=1), 1.0, "recent"),  # near-full weight
        _mk("AAA", "finnhub", tref - timedelta(hours=48), -1.0, "old"),  # decayed to 0.25
    ]
    m = A._session_metrics(A._news_df(items), ape, tref, weights)
    # decayed mean = (~1.0*1.0 + ~0.25*-1.0)/(1.0+0.25) > 0 → recent positive dominates
    assert m["sent_raw"] > 0.4
