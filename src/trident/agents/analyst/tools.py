"""ORACLE's research tools (doc 05) — the disciplined lens between the LLM and the data layer.

Five read-only tools over the EXISTING data layer (`data/fundamentals.financials` +
`derived_metrics`, `data/market` bars, SENTINEL's ingested `news_items`). No new HTTP code.

Three invariants make these safe for an LLM to drive:

  1. Anti-look-ahead. Every tool is bound to `asof` via ToolContext: fiscal years are filtered
     on `filed <= asof` BEFORE ratios are derived (so year-over-year growth can only see
     filings that existed), prices stop at asof, news uses `until=asof`.
  2. Never guess (golden rule 8). Missing data returns {"available": false, "reason": …}; a
     concept nobody tagged stays None. Nothing is inferred, interpolated or carried forward.
  3. Everything derived carries provenance. Ratios keep the {"value","formula","inputs"} shape
     that `fundamentals._metric` produces, and the valuation ratios computed HERE build the same
     shape — so `citations.py` can recompute any cited number from raw reported inputs instead
     of trusting a summary field.

Data may be shared across agents; SIGNALS may not. get_recent_news reads ingested `news_items`;
`store.get_signals` is deliberately NOT imported here.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

from trident.config import load_sectors, sector_peers
from trident.data import fundamentals as fund
from trident.data import market, store

STALE_DAYS = 100  # cached companyfacts older than this flag `stale` (doc 05 freshness rule)
_PERIOD_DAYS = {"1m": 31, "3m": 93, "6m": 186, "1y": 372, "2y": 744, "5y": 1860}
_CONTEXT_DAYS = 420  # always fetch >=1y so the 52-week high/low context exists
_TRADING_YEAR = 252
_FETCH_YEARS = 12  # over-fetch fiscal years, then filter to what was filed by asof
_WINDOW_YEARS = 5  # ...and report at most this many

# Share counts are the one line item whose SCALE varies by issuer under an identical us-gaap tag:
# most of the universe reports WeightedAverageNumberOfDilutedSharesOutstanding in absolute units
# (AAPL: 15,004,697,000) but at least one reports it in millions (MCD FY2025: 716.4), which would
# turn a $191B market cap into $191K and hand the model a nonsense P/E. A listed US large-cap
# cannot have fewer than a million shares, so anything below this is a units fault, not a count.
# We REFUSE the derived number rather than rescale it — inferring the intended scale would be
# guessing (golden rule 8), and a confidently wrong valuation is worse than a blank one.
MIN_PLAUSIBLE_SHARES = 1_000_000.0


@dataclass(frozen=True)
class ToolContext:
    """The anti-look-ahead seam. Bound once per research run; every tool reads `asof` from it so
    no tool can reach past the decision timestamp."""

    asof: datetime


# ------------------------------------------------------------------ helpers
def _num(x, nd: int = 4):
    """A finite python float rounded to nd places, or None (kills NaN/Inf and numpy scalars)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _metric(value, formula: str, inputs: dict) -> dict:
    """A derived number with provenance — the same shape `fundamentals._metric` emits, so
    citations.py recomputes tool-computed valuation ratios exactly as it does filing ratios."""
    return {"value": _num(value), "formula": formula, "inputs": inputs}


def visible_financials(ticker: str, asof: datetime) -> dict:
    """`financials()` with fiscal years filtered to filings available at asof, newest first.

    The filter runs BEFORE derived_metrics so revenue_growth_yoy compares two years the analyst
    could actually have seen — filtering afterwards would leak a future year into the growth
    denominator.
    """
    fin = fund.financials(ticker, years=_FETCH_YEARS)
    cutoff = asof.date().isoformat()
    rows = [r for r in fin.get("fiscal_years", []) if (r.get("filed") or "") <= cutoff]
    fin = {**fin, "fiscal_years": rows[:_WINDOW_YEARS]}
    return fin


def _is_stale(fetched_at: str | None) -> bool:
    if not fetched_at:
        return True
    try:
        return (datetime.now(UTC) - datetime.fromisoformat(fetched_at)).days > STALE_DAYS
    except (TypeError, ValueError):
        return True


def usable_shares(shares) -> tuple[float | None, str | None]:
    """(share_count, problem). Returns (None, reason) for an absent or implausibly small count
    so every share-dependent valuation collapses to None instead of to a wrong number."""
    if shares is None:
        return None, "diluted_shares not tagged by this issuer"
    try:
        s = float(shares)
    except (TypeError, ValueError):
        return None, "diluted_shares not numeric"
    if not math.isfinite(s) or s <= 0:
        return None, "diluted_shares non-finite or non-positive"
    if s < MIN_PLAUSIBLE_SHARES:
        return None, (
            f"diluted_shares={s:g} is below {MIN_PLAUSIBLE_SHARES:g} — the issuer appears to tag "
            "the count in millions; refusing to infer the scale (share-dependent valuation "
            "suppressed)"
        )
    return s, None


async def _financials_async(ticker: str, asof: datetime) -> dict:
    return await asyncio.to_thread(visible_financials, ticker, asof)


# ============================================================================= tools
async def get_price_history(ctx: ToolContext, ticker: str, period: str = "1y") -> dict:
    """Price action up to the decision date: last close, period return, annualized volatility,
    max drawdown, and distance from the trailing 52-week high/low."""
    ticker = ticker.upper()
    if period not in _PERIOD_DAYS:
        period = "1y"
    lookback = max(_PERIOD_DAYS[period], _CONTEXT_DAYS)
    start = ctx.asof.date() - timedelta(days=lookback)
    try:
        hist = await asyncio.to_thread(market.get_history, [ticker], start, ctx.asof.date())
    except Exception as e:  # noqa: BLE001 — a data outage is a tool result, never a crash
        return {"available": False, "ticker": ticker, "reason": f"price fetch failed: {e}"}
    if hist is None or hist.empty:
        return {"available": False, "ticker": ticker, "reason": "no price bars <= asof"}
    g = hist[hist["ticker"] == ticker].copy()
    g = g[pd.to_datetime(g["date"]).dt.date <= ctx.asof.date()].sort_values("date")
    if g.empty:
        return {"available": False, "ticker": ticker, "reason": "no price bars <= asof"}

    close = g["close"].astype(float).reset_index(drop=True)
    last = float(close.iloc[-1])
    window = close.tail(_PERIOD_DAYS[period])
    first = float(window.iloc[0])
    yr = close.tail(_TRADING_YEAR)
    hi, lo = float(yr.max()), float(yr.min())
    daily = close.pct_change().dropna()
    vol = float(daily.std() * math.sqrt(_TRADING_YEAR)) if len(daily) > 1 else None
    dd = float((close / close.cummax() - 1.0).min())

    return {
        "available": True,
        "ticker": ticker,
        "period": period,
        "asof": ctx.asof.date().isoformat(),
        "n_sessions": int(len(g)),
        # reported
        "last_close": _num(last, 2),
        "period_first_close": _num(first, 2),
        "high_52w": _num(hi, 2),
        "low_52w": _num(lo, 2),
        # derived, with provenance
        "period_return": _metric(
            (last / first - 1.0) if first else None,
            "(last_close - period_first_close) / period_first_close",
            {"last_close": last, "period_first_close": first},
        ),
        "pct_below_52w_high": _metric(
            (last / hi - 1.0) if hi else None,
            "(last_close - high_52w) / high_52w",
            {"last_close": last, "high_52w": hi},
        ),
        "pct_above_52w_low": _metric(
            (last / lo - 1.0) if lo else None,
            "(last_close - low_52w) / low_52w",
            {"last_close": last, "low_52w": lo},
        ),
        "annualized_volatility": _num(vol),
        "max_drawdown": _num(dd),
        "note": "returns/margins are FRACTIONS (0.25 = 25%).",
    }


async def get_financials(ctx: ToolContext, ticker: str) -> dict:
    """Annual income statement / balance sheet / cash flow from SEC EDGAR companyfacts, newest
    fiscal year first, restricted to filings available at asof. `concepts_used` names the
    us-gaap tag behind each line item; `missing_concepts` names what the issuer never tagged."""
    ticker = ticker.upper()
    fin = await _financials_async(ticker, ctx.asof)
    if not fin.get("fiscal_years"):
        return {
            "available": False,
            "ticker": ticker,
            "reason": "no 10-K fiscal years filed on or before asof",
            "missing_concepts": fin.get("missing_concepts", []),
        }
    return {
        "available": True,
        "ticker": ticker,
        "asof": ctx.asof.date().isoformat(),
        "stale": _is_stale(fin.get("fetched_at")),
        "units": "USD; diluted_shares in shares; capex reported POSITIVE (a payment).",
        "fiscal_years": fin["fiscal_years"],
        "concepts_used": fin.get("concepts_used", {}),
        "missing_concepts": fin.get("missing_concepts", []),
    }


async def get_ratios(ctx: ToolContext, ticker: str) -> dict:
    """Per-fiscal-year quality ratios (margins, ROE, ROIC proxy, leverage, growth) plus current
    valuation from asof's price. Every ratio carries {"value","formula","inputs"} so each number
    is checkable against the raw reported figures it came from."""
    ticker = ticker.upper()
    fin = await _financials_async(ticker, ctx.asof)
    if not fin.get("fiscal_years"):
        return {"available": False, "ticker": ticker, "reason": "no fiscal years <= asof"}
    dm = fund.derived_metrics(fin)
    latest = fin["fiscal_years"][0]

    ph = await get_price_history(ctx, ticker, "1m")
    last_close = ph.get("last_close") if ph.get("available") else None
    shares, shares_problem = usable_shares(latest.get("diluted_shares"))
    ni = latest.get("net_income")
    ocf, capex = latest.get("operating_cash_flow"), latest.get("capital_expenditures")
    std, ltd = latest.get("short_term_debt"), latest.get("long_term_debt")
    cash, op = latest.get("cash_and_equivalents"), latest.get("depreciation_amortization")
    oper = latest.get("operating_income")

    mcap = (last_close * shares) if (last_close is not None and shares) else None
    fcf = (ocf - capex) if (ocf is not None and capex is not None) else None
    ebitda = (oper + op) if (oper is not None and op is not None) else None
    ev = None
    if mcap is not None:
        ev = mcap + (std or 0.0) + (ltd or 0.0) - (cash or 0.0)

    valuation = {
        "market_cap": _metric(
            mcap,
            "last_close * diluted_shares",
            {"last_close": last_close, "diluted_shares": shares},
        ),
        "pe": _metric(
            (mcap / ni) if (mcap is not None and ni) else None,
            "last_close * diluted_shares / net_income",
            {"last_close": last_close, "diluted_shares": shares, "net_income": ni},
        ),
        "fcf_yield": _metric(
            (fcf / mcap) if (fcf is not None and mcap) else None,
            "(operating_cash_flow - capital_expenditures) / (last_close * diluted_shares)",
            {
                "operating_cash_flow": ocf,
                "capital_expenditures": capex,
                "last_close": last_close,
                "diluted_shares": shares,
            },
        ),
        "ev_ebitda_proxy": _metric(
            (ev / ebitda) if (ev is not None and ebitda) else None,
            "(last_close * diluted_shares + short_term_debt + long_term_debt "
            "- cash_and_equivalents) / (operating_income + depreciation_amortization)",
            {
                "last_close": last_close,
                "diluted_shares": shares,
                "short_term_debt": std,
                "long_term_debt": ltd,
                "cash_and_equivalents": cash,
                "operating_income": oper,
                "depreciation_amortization": op,
            },
        ),
    }
    data_quality = []
    if shares_problem:
        data_quality.append(shares_problem)
        # Keep the raw reported figure visible in `inputs` while the published value stays None:
        # citations.py treats a recomputation the tool declined to stand behind as UNVERIFIABLE,
        # so a model quoting the suppressed number cannot have it counted as evidence.
        for name in ("market_cap", "pe", "fcf_yield", "ev_ebitda_proxy"):
            valuation[name]["inputs"]["diluted_shares"] = latest.get("diluted_shares")

    return {
        "available": True,
        "ticker": ticker,
        "asof": ctx.asof.date().isoformat(),
        "stale": _is_stale(fin.get("fetched_at")),
        "note": "ratios are FRACTIONS (0.25 = 25%); every metric carries formula + inputs.",
        "latest_fy": latest.get("fy"),
        "valuation": valuation,
        "quality_by_year": dm.get("fiscal_years", []),
        "data_quality": data_quality,
    }


async def get_recent_news(ctx: ToolContext, ticker: str, n: int = 8) -> dict:
    """The `n` most recent materially-relevant news items for `ticker`, published <= asof.
    Reads ingested DATA (news_items), never SENTINEL's derived signals."""
    ticker = ticker.upper()
    n = max(1, min(int(n), 25))
    rows = await store.get_news(tickers=[ticker], until=ctx.asof, only_relevant=True)
    items = [
        {
            "title": r.get("title"),
            "source": r.get("source"),
            "published_at": r["published_at"].isoformat() if r.get("published_at") else None,
            "url": r.get("url"),
        }
        for r in rows[-n:]  # get_news returns ASC; tail-n = the most recent
    ]
    items.reverse()  # newest first for the reader
    return {
        "available": True,
        "ticker": ticker,
        "asof": ctx.asof.date().isoformat(),
        "count": len(items),
        "items": items,
    }


async def compare_peers(ctx: ToolContext, ticker: str) -> dict:
    """Latest-year quality ratios for `ticker` beside its universe sector peers.

    Peer groups here are small by construction (Energy and Industrials hold two names), so the
    peer count is reported alongside every comparison — a one-peer "sector median" is not a
    median, and the caller must be able to see that.
    """
    ticker = ticker.upper()
    sector = load_sectors().get(ticker)
    if sector is None:
        return {"available": False, "ticker": ticker, "reason": "ticker has no sector mapping"}
    peers = sector_peers(ticker)

    async def latest_metrics(t: str) -> dict | None:
        fin = await _financials_async(t, ctx.asof)
        if not fin.get("fiscal_years"):
            return None
        years = fund.derived_metrics(fin).get("fiscal_years") or []
        return years[0]["metrics"] if years else None

    subject = await latest_metrics(ticker)
    if subject is None:
        return {"available": False, "ticker": ticker, "reason": "no fundamentals for subject"}

    wanted = ["gross_margin", "operating_margin", "net_margin", "fcf_margin", "roe", "roic_proxy"]
    rows: dict[str, dict] = {ticker: {k: subject[k] for k in wanted if k in subject}}
    for p in peers:
        pm = await latest_metrics(p)
        if pm:
            rows[p] = {k: pm[k] for k in wanted if k in pm}

    ranks: dict[str, dict] = {}
    for k in wanted:
        vals = {t: r[k]["value"] for t, r in rows.items() if r.get(k, {}).get("value") is not None}
        if ticker not in vals:
            continue
        ordered = sorted(vals, key=lambda t: vals[t], reverse=True)  # higher is better
        ranks[k] = {"rank": ordered.index(ticker) + 1, "of": len(ordered)}
    return {
        "available": True,
        "ticker": ticker,
        "sector": sector,
        "asof": ctx.asof.date().isoformat(),
        "peers_in_sector": len(peers),
        "peers_with_data": len(rows) - 1,
        "note": "small sector groups: report the peer count with any comparison.",
        "metrics": rows,
        "rank": ranks,
    }


# ------------------------------------------------------------------ dispatch + schemas
TOOLS = {
    "get_price_history": get_price_history,
    "get_financials": get_financials,
    "get_ratios": get_ratios,
    "get_recent_news": get_recent_news,
    "compare_peers": compare_peers,
}

_TICKER = {"ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL"}}

# Anthropic tool schemas — identical across all names, so they sit under the cached prefix.
TOOL_SCHEMAS = [
    {
        "name": "get_price_history",
        "description": "Price action up to the decision date: last close, period return, "
        "annualized volatility, max drawdown, distance from the 52-week high/low.",
        "input_schema": {
            "type": "object",
            "properties": {
                **_TICKER,
                "period": {"type": "string", "enum": list(_PERIOD_DAYS)},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financials",
        "description": "Annual income statement, balance sheet and cash flow from SEC EDGAR "
        "10-K filings, newest fiscal year first, with the us-gaap concept behind each line item.",
        "input_schema": {"type": "object", "properties": _TICKER, "required": ["ticker"]},
    },
    {
        "name": "get_ratios",
        "description": "Quality ratios per fiscal year (margins, ROE, ROIC proxy, leverage, "
        "revenue growth) plus current valuation (P/E, FCF yield, EV/EBITDA proxy). Each ratio "
        "includes the formula and the raw inputs it was computed from.",
        "input_schema": {"type": "object", "properties": _TICKER, "required": ["ticker"]},
    },
    {
        "name": "get_recent_news",
        "description": "The n most recent materially-relevant news items for the ticker, "
        "published on or before the decision date.",
        "input_schema": {
            "type": "object",
            "properties": {**_TICKER, "n": {"type": "integer", "minimum": 1, "maximum": 25}},
            "required": ["ticker"],
        },
    },
    {
        "name": "compare_peers",
        "description": "Latest-year quality ratios for the ticker beside its sector peers, with "
        "the ticker's rank on each and the number of peers the comparison rests on.",
        "input_schema": {"type": "object", "properties": _TICKER, "required": ["ticker"]},
    },
]
