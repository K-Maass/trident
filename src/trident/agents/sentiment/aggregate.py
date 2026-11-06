"""SENTINEL aggregation (doc 04, pipeline stage 4) — per ticker, per session.

Turns scored news_items + ApeWisdom attention snapshots into the four signals SENTINEL trades:

  sent_raw     weight-normalized mean of item FinBERT scores over enabled+present sentiment
               sources, each source's items combined with exponential time decay (half-life 24h).
               Source weights (sentinel_sources.yaml) are RENORMALIZED over whatever sources are
               actually present that session — a missing/stale/disabled source contributes
               nothing, never a silent stale value.
  sent_z       (sent_raw - trailing-21d mean) / trailing-21d std. The baseline is the PRIOR
               sessions only (shift(1)), so it is a look-ahead-safe "spike vs its own history".
               Marked `immature` until 21 prior sessions exist (min_periods=10 to compute at all).
  attention_z  z of (raw relevant-item count) and z of (ApeWisdom mentions), each vs its own
               trailing-21d baseline, averaged over whichever is available.
  disagreement std of that session's item scores (high = contested story).

Look-ahead seam: a session d's aggregate uses only items published <= end-of-day-d (UTC); the
agent consumes the PREVIOUS session's aggregate, so the signal is always fully formed before it
is acted on (mirrors QUANT's previous_session execution).
"""

from __future__ import annotations

import math
import warnings
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from trident.config import load_sentinel_sources
from trident.data import calendar as trading_calendar
from trident.data import store

HALF_LIFE_HOURS = 24.0
DECAY_WINDOW_HOURS = 72.0  # ~3 half-lives; older items contribute < 12.5%
ATTENTION_WINDOW_HOURS = 24.0
APEWISDOM_WINDOW_HOURS = 48.0
Z_WINDOW = 21
Z_MIN_PERIODS = 10
TOP_ITEMS = 5
AGG_KIND = "sentiment_aggregate"
PANEL_LOOKBACK_DAYS = 160  # how far back aggregate() computes: >= 21 sessions + decay buffer

# Baseline-std floors. A z-score divides by the trailing std, so a ticker whose baseline barely
# moves manufactures enormous z's out of noise-sized changes — the dangerous case is a
# well-covered ticker with flat, neutral sent_raw (std ~1e-3), which clears the evidence floor
# and could fabricate a sent_z >> SENT_Z_ENTER from a meaningless wobble. Flooring the
# DENOMINATOR (not winsorizing the input, which would kill the real spikes we exist to catch) is
# the fix. The input itself can still move freely.
SENT_STD_FLOOR = 0.05  # sentiment units on [-1,1]; a decayed-mean move under this is noise
COUNT_STD_FLOOR = 1.0  # items/day; counts are integers — a sub-one-item baseline is not a sigma
MENTIONS_STD_FLOOR = 1.0  # absolute floor beneath the Poisson (sqrt-mean) floor
Z_CLIP = 5.0  # backstop on any residual pathology; keeps rationale numbers readable


# --------------------------------------------------------------------- pure helpers
def decay_weight(age_hours: float, half_life: float = HALF_LIFE_HOURS) -> float:
    """Exponential time decay, half-life 24h (doc 04)."""
    return 0.5 ** (max(0.0, age_hours) / half_life)


def source_key(item_source: str) -> str:
    """Map a news_items.source ('finnhub', 'rss:cnbc', …) to a registry source name."""
    s = (item_source or "").lower()
    if s.startswith("rss"):
        return "rss"
    if s.startswith("finnhub"):
        return "finnhub_news"
    if s.startswith("tradestie"):
        return "tradestie_wsb"
    if s.startswith("stocktwits"):
        return "stocktwits"
    if s.startswith("reddit"):
        return "reddit_praw"
    return s


def sentiment_source_weights(registry: dict | None = None) -> dict[str, float]:
    """Enabled sentiment-contributing sources (raw_text + pre_scored) → their yaml weights."""
    reg = registry if registry is not None else load_sentinel_sources()
    return {
        name: float(c.get("weight", 0.0))
        for name, c in reg.items()
        if c.get("enabled") and c.get("kind") in ("raw_text", "pre_scored")
    }


def combine_sources(
    source_means: dict[str, float], weights: dict[str, float]
) -> tuple[float, list[str]]:
    """Weight-normalized combine over the sources PRESENT this session (renormalized to those).
    Returns (sent_raw, sources_used). NaN + [] when no enabled source is present."""
    present = {
        k: v
        for k, v in source_means.items()
        if k in weights and v is not None and not math.isnan(v)
    }
    if not present:
        return float("nan"), []
    den = sum(weights[k] for k in present)
    if den <= 0:
        return float("nan"), []
    num = sum(weights[k] * v for k, v in present.items())
    return num / den, sorted(present)


def trailing_z(
    s: pd.Series,
    window: int = Z_WINDOW,
    min_periods: int = Z_MIN_PERIODS,
    std_floor: float = 0.0,
    poisson_floor: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """z of s vs its OWN trailing baseline (prior sessions only). Returns (z, baseline_count).

    The denominator is floored at `std_floor` so a near-flat baseline cannot manufacture huge
    z's from noise-sized moves. `poisson_floor` additionally floors it at sqrt(baseline mean) —
    the natural noise scale of a count process, needed for series whose magnitude varies by
    orders of magnitude across tickers (ApeWisdom mentions run from single digits to hundreds,
    so a single ABSOLUTE floor would be wrong for both ends).

    A baseline with EXACTLY zero variance stays NaN (refused, not floored): it carries no
    information to z-score against, and such a ticker is dark enough that its sent_raw baseline
    is all-NaN anyway, so it cannot enter regardless. The final z is clipped to +/-Z_CLIP.
    """
    base = s.shift(1)
    mean = base.rolling(window, min_periods=min_periods).mean()
    std = base.rolling(window, min_periods=min_periods).std(ddof=0)
    count = base.rolling(window, min_periods=1).count()
    floor = pd.Series(float(std_floor), index=s.index, dtype=float)
    if poisson_floor:
        floor = np.maximum(floor, np.sqrt(mean.clip(lower=0.0)))
    denom = std.replace(0.0, np.nan).clip(lower=floor)
    z = ((s - mean) / denom).clip(-Z_CLIP, Z_CLIP)
    return z, count


def _nan_to_none(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _tref(d: date) -> datetime:
    """End-of-day UTC for session d — the look-ahead cutoff for that session's aggregate."""
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=UTC)


# ------------------------------------------------------------------ DB-driven compute
def _news_df(news: list[dict]) -> pd.DataFrame:
    cols = ["ticker", "srckey", "published_at", "sentiment", "title", "url", "source"]
    scored = [n for n in news if n.get("sentiment") is not None]
    if not scored:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(scored)
    df["srckey"] = df["source"].map(source_key)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["sentiment"] = df["sentiment"].astype(float)
    return df[cols]


def _ape_df(signals: list[dict]) -> pd.DataFrame:
    rows = []
    for s in signals:
        md = s.get("metadata") or {}
        if md.get("kind") != "attention_snapshot":
            continue
        rows.append(
            {
                "ticker": (s.get("ticker") or "").upper(),
                "ts": s["ts"],
                "mentions": md.get("mentions"),
                "ape_rank": md.get("rank"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ticker", "ts", "mentions", "ape_rank"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce")
    return df


def _session_metrics(
    ndf_t: pd.DataFrame, adf_t: pd.DataFrame, tref: datetime, weights: dict[str, float]
) -> dict[str, Any]:
    """Raw (pre-z) metrics for one ticker at one session cutoff."""
    lo = tref - timedelta(hours=DECAY_WINDOW_HOURS)
    win = ndf_t[(ndf_t.published_at > lo) & (ndf_t.published_at <= tref)]
    source_means: dict[str, float] = {}
    for sk, g in win.groupby("srckey"):
        ages = (tref - g.published_at).dt.total_seconds() / 3600.0
        w = ages.map(decay_weight)
        wsum = float(w.sum())
        source_means[sk] = float((w * g.sentiment).sum() / wsum) if wsum > 0 else float("nan")
    sent_raw, sources_used = combine_sources(source_means, weights)

    a_lo = tref - timedelta(hours=ATTENTION_WINDOW_HOURS)
    recent = ndf_t[(ndf_t.published_at > a_lo) & (ndf_t.published_at <= tref)]
    item_count = int(len(recent))

    ape_lo = tref - timedelta(hours=APEWISDOM_WINDOW_HOURS)
    ape_win = adf_t[(adf_t.ts > ape_lo) & (adf_t.ts <= tref)]
    if len(ape_win):
        last = ape_win.sort_values("ts").iloc[-1]
        mentions = float(last.mentions) if pd.notna(last.mentions) else float("nan")
        ape_rank = None if pd.isna(last.ape_rank) else int(last.ape_rank)
    else:
        mentions, ape_rank = float("nan"), None

    return {
        "sent_raw": sent_raw,
        "disagreement": float(win.sentiment.std(ddof=0)) if len(win) >= 2 else 0.0,
        "item_count": item_count,
        "n_evidence": int(len(win)),
        "mentions": mentions,
        "ape_rank": ape_rank,
        "sources_used": sources_used,
    }


async def _compute(
    universe: list[str], start: date, end: date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full per-(ticker, session) panel with z-scores. Returns (panel, news_df, ape_df)."""
    universe = [t.upper() for t in universe]
    weights = sentiment_source_weights()
    lo = datetime.combine(start - timedelta(days=4), datetime.min.time(), tzinfo=UTC)
    hi = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    ndf = _news_df(await store.get_news(universe, since=lo, until=hi, only_relevant=True))
    adf = _ape_df(await store.get_signals("sentinel", universe, since=lo, until=hi))
    sess = trading_calendar.sessions(start, end)

    panels: list[pd.DataFrame] = []
    for tk in universe:
        ndf_t = ndf[ndf.ticker == tk]
        adf_t = adf[adf.ticker == tk]
        rows = []
        for d in sess:
            m = _session_metrics(ndf_t, adf_t, _tref(d), weights)
            m.update({"date": d, "ticker": tk})
            rows.append(m)
        tdf = pd.DataFrame(rows).set_index("date").sort_index()
        sent_z, base_count = trailing_z(tdf["sent_raw"], std_floor=SENT_STD_FLOOR)
        z_count, _ = trailing_z(
            tdf["item_count"].astype(float), std_floor=COUNT_STD_FLOOR, poisson_floor=True
        )
        z_ment, _ = trailing_z(tdf["mentions"], std_floor=MENTIONS_STD_FLOOR, poisson_floor=True)
        tdf["sent_z"] = sent_z
        tdf["attention_z"] = pd.concat([z_count, z_ment], axis=1).mean(axis=1, skipna=True)
        tdf["immature"] = base_count < Z_WINDOW
        panels.append(tdf.reset_index())

    # A ticker with no news has all-NaN sent_raw/mentions columns; concat's all-NA dtype warning
    # is benign here (we want NaN preserved), so quiet it rather than change the values.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    return panel, ndf, adf


def _row_to_signal(row: pd.Series) -> dict:
    md = {
        "kind": AGG_KIND,
        "sent_raw": _nan_to_none(float(row.sent_raw)),
        "sent_z": _nan_to_none(float(row.sent_z)),
        "attention_z": _nan_to_none(float(row.attention_z)),
        "disagreement": _nan_to_none(float(row.disagreement)),
        "item_count": int(row.item_count),
        "n_evidence": int(row.n_evidence),
        "apewisdom_mentions": _nan_to_none(float(row.mentions)),
        "apewisdom_rank": None
        if row.ape_rank is None or pd.isna(row.ape_rank)
        else int(row.ape_rank),
        "sources_used": list(row.sources_used),
        "immature": bool(row.immature),
    }
    score = row.sent_raw
    score = 0.0 if (score is None or (isinstance(score, float) and math.isnan(score))) else score
    return {
        "agent_id": "sentinel",
        "ts": _tref(row.date),
        "ticker": row.ticker,
        "score": max(-1.0, min(1.0, float(score))),
        "metadata": md,
    }


async def aggregate(
    universe: list[str], start: date, end: date, persist: bool = False
) -> pd.DataFrame:
    """Compute the panel over [start, end]; if persist, replace the sentiment_aggregate signal
    rows in that range (delete-then-insert, so re-runs are idempotent)."""
    panel, _, _ = await _compute(universe, start, end)
    if persist and not panel.empty:
        rows = [_row_to_signal(r) for _, r in panel.iterrows()]
        await store.delete_signals(
            "sentinel",
            AGG_KIND,
            since=_tref(start) - timedelta(days=1),
            until=_tref(end) + timedelta(days=1),
        )
        await store.insert_signal_snapshots(rows)
    return panel


def _top_items(
    ndf: pd.DataFrame, ticker: str, tref: datetime, weights: dict[str, float]
) -> list[dict]:
    lo = tref - timedelta(hours=DECAY_WINDOW_HOURS)
    sub = ndf[(ndf.ticker == ticker) & (ndf.published_at > lo) & (ndf.published_at <= tref)].copy()
    if sub.empty:
        return []
    ages = (tref - sub.published_at).dt.total_seconds() / 3600.0
    w = ages.map(decay_weight) * sub.srckey.map(lambda k: weights.get(k, 0.0))
    sub["contrib"] = (w * sub.sentiment).abs()
    sub = sub.sort_values("contrib", ascending=False).head(TOP_ITEMS)
    return [
        {
            "title": r.title,
            "source": r.source,
            "score": round(float(r.sentiment), 4),
            "url": r.url,
        }
        for r in sub.itertuples()
    ]


async def latest_aggregates(universe: list[str], asof: datetime | None = None) -> dict[str, dict]:
    """Per-ticker aggregate at the session the agent should act on (previous_session(asof)),
    with top-5 contributing items attached for the decision rationale. This is what the runner
    injects as ctx.extras['signals']."""
    asof = asof or datetime.now(UTC)
    target = trading_calendar.previous_session(asof.date())
    start = target - timedelta(days=160)
    panel, ndf, _ = await _compute(universe, start, target)
    weights = sentiment_source_weights()
    tref = _tref(target)
    out: dict[str, dict] = {}
    if panel.empty:
        return out
    latest = panel[panel.date == target]
    for r in latest.itertuples():
        out[r.ticker] = {
            "sent_raw": _nan_to_none(float(r.sent_raw)),
            "sent_z": _nan_to_none(float(r.sent_z)),
            "attention_z": _nan_to_none(float(r.attention_z)),
            "disagreement": _nan_to_none(float(r.disagreement)),
            "n_evidence": int(r.n_evidence),
            "item_count": int(r.item_count),
            "apewisdom_rank": None
            if r.ape_rank is None or pd.isna(r.ape_rank)
            else int(r.ape_rank),
            "sources_used": list(r.sources_used),
            "immature": bool(r.immature),
            "session": target.isoformat(),
            "top_items": _top_items(ndf, r.ticker, tref, weights),
        }
    return out
