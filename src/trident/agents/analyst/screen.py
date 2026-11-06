"""Deterministic candidate screen (doc 05) — the $0, reproducible front of the graph.

Cross-sectional z-scores of four fundamental factors from the cached EDGAR statements, all
restricted to filings available at asof (`tools.visible_financials`, so the screen and the LLM's
tools share ONE anti-look-ahead rule rather than two that can drift). Composite = mean of the
available z-scores; the top SCREEN_TOP_N plus every currently-held name (holdings are always
re-underwritten) become the LLM candidates.

This is code, not the model: ranking a table is arithmetic (governing rule 1). The full ranked
table is returned so candidate selection is inspectable in the run artifact.

A factor the issuer never tagged stays NaN and z-scores to 0 (neutral) — it is neither a
penalty nor a reward, and `n_factors` on each row records how many of the four actually landed
so a name ranked on thin evidence is visible rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from trident.agents.analyst.tools import usable_shares, visible_financials
from trident.data import fundamentals as fund
from trident.data import market

SCREEN_TOP_N = 10  # deep-dive this many by composite (plus held); doc 09 cuts to 6 over-budget
_FACTORS = ["earnings_yield", "roic_proxy", "revenue_growth_yoy", "fcf_margin"]


@dataclass
class ScreenResult:
    table: pd.DataFrame  # full ranked composite table, for the artifact
    candidates: list[str]  # top_n by composite + every held name


def _last_close(ticker: str, asof: datetime) -> float | None:
    try:
        hist = market.get_history([ticker], asof.date() - timedelta(days=40), asof.date())
    except Exception:  # noqa: BLE001 — a missing price only drops earnings_yield for this name
        return None
    if hist is None or hist.empty:
        return None
    g = hist[hist["ticker"] == ticker.upper()]
    g = g[pd.to_datetime(g["date"]).dt.date <= asof.date()].sort_values("date")
    return float(g["close"].iloc[-1]) if not g.empty else None


def _factors_for(ticker: str, asof: datetime) -> dict:
    ticker = ticker.upper()
    row = {"ticker": ticker, **{f: np.nan for f in _FACTORS}}
    fin = visible_financials(ticker, asof)
    years = fin.get("fiscal_years") or []
    if not years:
        return row
    latest = years[0]
    metrics = (fund.derived_metrics(fin).get("fiscal_years") or [{}])[0].get("metrics", {})

    for name in ("roic_proxy", "revenue_growth_yoy", "fcf_margin"):
        v = (metrics.get(name) or {}).get("value")
        if v is not None:
            row[name] = float(v)

    # Same share-count plausibility guard the tools use: an issuer tagging the count in millions
    # would otherwise post an earnings yield ~1e9x too high and hijack the whole ranking.
    ni = latest.get("net_income")
    shares, _problem = usable_shares(latest.get("diluted_shares"))
    price = _last_close(ticker, asof) if shares else None
    if ni is not None and price and shares:
        mcap = price * shares
        if mcap:
            row["earnings_yield"] = ni / mcap
    return row


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; NaN -> 0 (neutral), zero/degenerate variance -> 0."""
    v = s.astype(float)
    sd = v.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((v - v.mean(skipna=True)) / sd).fillna(0.0)


def run_screen(
    universe: list[str], asof: datetime, held: list[str] | None = None, top_n: int = SCREEN_TOP_N
) -> ScreenResult:
    held = [h.upper() for h in (held or [])]
    table = pd.DataFrame([_factors_for(t, asof) for t in universe]).set_index("ticker")
    table["n_factors"] = table[_FACTORS].notna().sum(axis=1)
    for f in _FACTORS:
        table[f"z_{f}"] = _zscore(table[f])
    table["composite"] = table[[f"z_{f}" for f in _FACTORS]].mean(axis=1)
    table = table.sort_values("composite", ascending=False)

    universe_set = {t.upper() for t in universe}
    top = list(table.head(top_n).index)
    candidates = list(dict.fromkeys(top + [h for h in held if h in universe_set]))
    return ScreenResult(table=table.reset_index(), candidates=candidates)
