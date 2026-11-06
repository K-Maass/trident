"""Fundamentals from SEC EDGAR companyfacts (doc 01/02).

EDGAR is the one production-grade free source here (public domain, 10 req/s, descriptive
User-Agent required via EDGAR_USER_AGENT). Cached to data/cache/fundamentals/ as JSON with
a _fetched_at stamp. Research/eval only; ORACLE reads these via its tools (Phase 5).

Extraction contract (Phase 5): `financials()` returns plain JSON-serializable dicts — they
feed an LLM tool result AND a JSONB column, so no DataFrames, no numpy scalars, no NaN.
A concept nobody tagged is reported as missing with None values; nothing is ever inferred,
interpolated or carried forward (golden rule 8).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from trident.config import CACHE_DIR, get_settings, load_cik_overrides, load_universe

FUND_DIR = CACHE_DIR / "fundamentals"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# --------------------------------------------------------------------- concepts
# Companies tag the same economics under different us-gaap concepts (taxonomy drift +
# industry conventions), so every line item is an ORDERED list, most specific/most common
# first; `annual_series` resolves the preference per fiscal year. Coverage notes below were
# measured over the last 5 FYs of the 39 cached universe tickers (2026-07-21); "n miss" =
# tickers where nothing in the list is tagged — a genuine gap, not a bug to paper over.

# --- income statement
# Revenue concept has drifted across GAAP taxonomy versions; try newest first.
_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
_COST_OF_REVENUE_CONCEPTS = [  # 9 miss (banks/payment networks report no COGS)
    "CostOfRevenue",
    "CostOfGoodsAndServicesSold",
    "CostOfGoodsSold",
    "CostOfServices",
]
_GROSS_PROFIT_CONCEPTS = ["GrossProfit"]  # 23 miss — many issuers never present it
_OPERATING_INCOME_CONCEPTS = ["OperatingIncomeLoss"]  # 9 miss (CVX/LLY/MRK/NKE/GE/banks)
_NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",  # attributable to the parent
    "ProfitLoss",  # incl. noncontrolling interests
    "NetIncomeLossAvailableToCommonStockholdersBasic",
]
_EPS_DILUTED_CONCEPTS = [  # units USD/shares
    "EarningsPerShareDiluted",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
]
_INTEREST_EXPENSE_CONCEPTS = [
    "InterestExpense",
    "InterestExpenseDebt",
    "InterestAndDebtExpense",
    # NOT InterestIncomeExpenseNet: for banks that is NET INTEREST INCOME (a revenue line), and
    # letting it fill a gap year would splice a revenue number into an expense series.
]
_INCOME_TAX_EXPENSE_CONCEPTS = ["IncomeTaxExpenseBenefit"]
_DEPRECIATION_AMORTIZATION_CONCEPTS = [  # cash-flow add-back; feeds the EBITDA proxy
    "DepreciationDepletionAndAmortization",
    "DepreciationAmortizationAndAccretionNet",
    "DepreciationAndAmortization",
    "Depreciation",
]

# --- balance sheet (instants at fiscal year-end)
_TOTAL_ASSETS_CONCEPTS = ["Assets"]
_TOTAL_LIABILITIES_CONCEPTS = ["Liabilities"]  # 13 miss — plenty of issuers omit the subtotal
_EQUITY_CONCEPTS = [
    "StockholdersEquity",  # parent only
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
_CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",  # incl. restricted
]
_SHORT_TERM_DEBT_CONCEPTS = [  # broadest current-debt measure first; scope varies by issuer
    "DebtCurrent",
    "LongTermDebtCurrent",
    "ShortTermBorrowings",
    "OtherShortTermBorrowings",
    "CommercialPaper",
]
_LONG_TERM_DEBT_CONCEPTS = ["LongTermDebtNoncurrent", "LongTermDebt"]
_INVENTORY_CONCEPTS = ["InventoryNet", "InventoryFinishedGoodsNetOfReserves"]

# --- cash flow
_OPERATING_CASH_FLOW_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
_CAPEX_CONCEPTS = [  # reported POSITIVE in us-gaap (a payment) -> SUBTRACTS from OCF
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "PaymentsToAcquireProductiveAssets",
    "PaymentsForCapitalImprovements",
    "PaymentsToAcquireOtherPropertyPlantAndEquipment",
]
_DIVIDENDS_PAID_CONCEPTS = ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"]
_SHARE_REPURCHASE_CONCEPTS = [
    "PaymentsForRepurchaseOfCommonStock",
    "PaymentsForRepurchaseOfEquity",
]

# --- share counts (units "shares")
_DILUTED_SHARES_CONCEPTS = [
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstandingBasicAndDiluted",
]

# (field name, ordered concepts, XBRL unit) — the schema of a `financials()` fiscal-year row.
_LINE_ITEMS: list[tuple[str, list[str], str]] = [
    ("revenue", _REVENUE_CONCEPTS, "USD"),
    ("cost_of_revenue", _COST_OF_REVENUE_CONCEPTS, "USD"),
    ("gross_profit", _GROSS_PROFIT_CONCEPTS, "USD"),
    ("operating_income", _OPERATING_INCOME_CONCEPTS, "USD"),
    ("net_income", _NET_INCOME_CONCEPTS, "USD"),
    ("eps_diluted", _EPS_DILUTED_CONCEPTS, "USD/shares"),
    ("interest_expense", _INTEREST_EXPENSE_CONCEPTS, "USD"),
    ("income_tax_expense", _INCOME_TAX_EXPENSE_CONCEPTS, "USD"),
    ("depreciation_amortization", _DEPRECIATION_AMORTIZATION_CONCEPTS, "USD"),
    ("total_assets", _TOTAL_ASSETS_CONCEPTS, "USD"),
    ("total_liabilities", _TOTAL_LIABILITIES_CONCEPTS, "USD"),
    ("stockholders_equity", _EQUITY_CONCEPTS, "USD"),
    ("cash_and_equivalents", _CASH_CONCEPTS, "USD"),
    ("short_term_debt", _SHORT_TERM_DEBT_CONCEPTS, "USD"),
    ("long_term_debt", _LONG_TERM_DEBT_CONCEPTS, "USD"),
    ("inventory", _INVENTORY_CONCEPTS, "USD"),
    ("operating_cash_flow", _OPERATING_CASH_FLOW_CONCEPTS, "USD"),
    ("capital_expenditures", _CAPEX_CONCEPTS, "USD"),
    ("dividends_paid", _DIVIDENDS_PAID_CONCEPTS, "USD"),
    ("share_repurchases", _SHARE_REPURCHASE_CONCEPTS, "USD"),
    ("diluted_shares", _DILUTED_SHARES_CONCEPTS, "shares"),
]

_ANNUAL_MIN_DAYS, _ANNUAL_MAX_DAYS = 330, 400
_NETWORK_ERRORS = (httpx.TransportError, httpx.TimeoutException)
_FACT_COLS = ["concept", "end", "val", "fy", "fp", "form", "filed", "frame"]


def _ua() -> str:
    settings = get_settings()
    settings.require("edgar_user_agent")
    return settings.edgar_user_agent


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception_type(_NETWORK_ERRORS),
    reraise=True,
)
def _get(url: str) -> httpx.Response:
    r = httpx.get(url, headers={"User-Agent": _ua()}, timeout=30.0)
    r.raise_for_status()
    return r


# --------------------------------------------------------------- ticker -> CIK
def cik_map(cache_dir: Path = FUND_DIR, refresh: bool = False) -> dict[str, str]:
    """{TICKER: zero-padded-10-digit CIK} from SEC's registry, cached locally."""
    path = cache_dir / "_ticker_cik.json"
    if not refresh and path.exists():
        data = json.loads(path.read_text())
    else:
        data = _get(_TICKERS_URL).json()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    return {row["ticker"].upper(): f"{int(row['cik_str']):010d}" for row in data.values()}


def cik_for(ticker: str, cache_dir: Path = FUND_DIR) -> str | None:
    """CIK for a Trident ticker, normalizing symbology at the SEC boundary.

    SEC's registry spells class shares with a dash (BRK-B) while universe.yaml and the rest
    of the system use a dot (BRK.B) — same idea as market._symbol_for. Only the SEC lookup
    is dashed; cache files stay keyed on the canonical repo ticker.

    A pinned override in universe.yaml wins over the registry: SEC's ticker->CIK file is
    itself occasionally wrong (see the XOM entry there), and a wrong CIK yields a filing
    shell with zero us-gaap facts rather than an error — silent, which golden rule 8 forbids.
    """
    t = ticker.upper()
    override = load_cik_overrides().get(t)
    if override:
        return override
    m = cik_map(cache_dir)
    return m.get(t) or m.get(t.replace(".", "-"))  # BRK.B -> BRK-B


# --------------------------------------------------------------- companyfacts
def fetch_companyfacts(
    ticker: str, cache_dir: Path = FUND_DIR, refresh: bool = False
) -> dict | None:
    """Raw EDGAR companyfacts for `ticker`, cache-first (JSON with _fetched_at)."""
    ticker = ticker.upper()
    path = cache_dir / f"{ticker}.json"
    if not refresh and path.exists():
        return json.loads(path.read_text())
    cik = cik_for(ticker, cache_dir)
    if cik is None:
        return None
    data = _get(_FACTS_URL.format(cik=cik)).json()
    data["_fetched_at"] = datetime.now(UTC).isoformat()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


# ------------------------------------------------------------------ extraction
def annual_concept(facts: dict, concepts: list[str], units: str = "USD") -> pd.DataFrame:
    """Annual (10-K, ~full-year) values for the first matching concept, one row per fiscal
    year-end (latest-filed wins on restatement), newest first. Pure — offline-testable."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in concepts:
        node = gaap.get(concept)
        if not node:
            continue
        recs = []
        for e in node.get("units", {}).get(units, []):
            if e.get("form") != "10-K" or not e.get("end"):
                continue
            if e.get("start"):
                days = (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
                if not (_ANNUAL_MIN_DAYS <= days <= _ANNUAL_MAX_DAYS):
                    continue  # skip quarterly / partial periods
            recs.append({k: e.get(k) for k in ("end", "val", "fy", "fp", "form", "filed", "frame")})
        if not recs:
            continue
        df = pd.DataFrame(recs)
        df.insert(0, "concept", concept)
        df["end"] = pd.to_datetime(df["end"])
        df = df.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        return df.sort_values("end", ascending=False).reset_index(drop=True)[_FACT_COLS]
    return pd.DataFrame(columns=_FACT_COLS)


def revenue(ticker: str, years: int = 5, cache_dir: Path = FUND_DIR) -> pd.DataFrame:
    """Last `years` fiscal years of annual revenue (the Phase 1 acceptance probe)."""
    facts = fetch_companyfacts(ticker, cache_dir=cache_dir)
    if facts is None:
        return pd.DataFrame(columns=_FACT_COLS)
    return annual_concept(facts, _REVENUE_CONCEPTS).head(years)


# ------------------------------------------------------- statements (ORACLE, Phase 5)
def _num(v: object) -> float | None:
    """Any XBRL value -> a plain JSON-safe float, or None. Never NaN/inf (JSONB + LLM)."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _fiscal_year(end: str) -> int:
    """Fiscal-year LABEL for a period end (ISO date): the calendar year the period ends in.

    Caveat: this is not always the company's own label (Home Depot calls the year ending
    2026-02-01 "fiscal 2025"). `end` is the authoritative field. The one adjustment: 52/53-week
    calendars that end in the first days of January cover the PRIOR calendar year (JNJ's FY2022
    ended 2023-01-01), so those roll back a year — otherwise one company shows two "2023"s.
    """
    d = date.fromisoformat(end)
    return d.year - 1 if (d.month == 1 and d.day <= 7) else d.year


def annual_series(facts: dict, concepts: list[str], units: str = "USD") -> dict[str, dict]:
    """One line item as {period-end ISO: {"value", "concept", "filed", "form"}}. Pure.

    Preference is resolved PER FISCAL YEAR, not once for the whole series: for each period end
    the first concept in `concepts` that reported it wins. Picking a single concept up front
    silently loses years, and both directions of that bug are live in the universe cache —
    Apple tags PaymentsOfDividendsCommonStock only in FY2016–17 and PaymentsOfDividends through
    FY2025 (whole-series first-match would return 2017 dividends), while Alphabet reports FY2025
    only under `Revenues` and FY2022 only under RevenueFromContractWithCustomerExcludingAssessedTax
    (whole-series pick loses one end or the other).

    The trade-off is that a series CAN mix concepts across years, which is only sound when the
    alternates mean the same thing — hence the lists are deliberately narrow, alternates are
    ordered most-specific-first, and every value carries the `concept` it came from so a mixed
    series is visible rather than implied. Values are never converted, scaled or interpolated.
    """
    out: dict[str, dict] = {}
    for concept in concepts:
        df = annual_concept(facts, [concept], units=units)
        for row in df.itertuples(index=False):
            end = row.end.date().isoformat()
            val = _num(row.val)
            if val is None or end in out:  # earlier concept in the list already won this year
                continue
            out[end] = {
                "value": val,
                "concept": concept,
                "filed": str(row.filed) if row.filed else None,
                "form": str(row.form) if row.form else None,
            }
    return out


def financials(ticker: str, years: int = 5, cache_dir: Path = FUND_DIR) -> dict:
    """Annual income statement / balance sheet / cash flow for `ticker`, newest FY first.

    Cache-first over the EDGAR companyfacts JSON (no network unless the cache is cold).
    JSON-serializable throughout: floats, strings, None — no DataFrames, no numpy, no NaN.

        {"ticker", "fetched_at", "fiscal_years": [{"fy", "end", "filed", "form", <line items>}],
         "concepts_used": {field: [us-gaap concept, ...]}, "missing_concepts": [...]}

    Rows are anchored on the period-end dates seen across all line items (exact-date match;
    within one 10-K every statement shares the fiscal year-end), newest first, at most `years`.
    A line item with no value in ANY returned year — never tagged, or only tagged in years
    outside the window — is listed in `missing_concepts` and is None in every row. Nothing is
    inferred, interpolated or carried forward (golden rule 8). `concepts_used` names the
    us-gaap tag(s) each line item came from (a list; >1 entry means the issuer switched tags
    mid-window), so any number here can be traced back to the filing. Annual/10-K only, per
    `annual_concept` — there is deliberately no quarterly path.
    """
    ticker = ticker.upper()
    facts = fetch_companyfacts(ticker, cache_dir=cache_dir)
    if facts is None:
        return {
            "ticker": ticker,
            "fetched_at": None,
            "fiscal_years": [],
            "concepts_used": {},
            "missing_concepts": [name for name, _, _ in _LINE_ITEMS],
        }

    series = {
        name: annual_series(facts, concepts, units=units) for name, concepts, units in _LINE_ITEMS
    }
    ends = sorted({e for by_end in series.values() for e in by_end}, reverse=True)[:years]

    rows: list[dict] = []
    used: dict[str, list[str]] = {}
    for end in ends:
        hits = [by_end[end] for by_end in series.values() if end in by_end]
        filed = max((h["filed"] for h in hits if h["filed"]), default=None)
        form = next((h["form"] for h in hits if h["filed"] == filed), None)
        row: dict = {"fy": _fiscal_year(end), "end": end, "filed": filed, "form": form}
        for name, _, _ in _LINE_ITEMS:
            hit = series[name].get(end)
            row[name] = hit["value"] if hit else None
            if hit and hit["concept"] not in used.setdefault(name, []):
                used[name].append(hit["concept"])
        rows.append(row)

    # "Missing" is judged over the RETURNED window: a concept tagged only in years we don't show
    # is missing as far as the caller is concerned, and saying so beats an all-None column.
    missing = [name for name, _, _ in _LINE_ITEMS if name not in used]
    return {
        "ticker": ticker,
        "fetched_at": facts.get("_fetched_at"),
        "fiscal_years": rows,
        "concepts_used": used,
        "missing_concepts": missing,
    }


# ------------------------------------------------------------------ derived metrics
def _safe_div(num: float | None, den: float | None) -> float | None:
    """Division that yields None on missing input or a zero denominator (never raises)."""
    if num is None or den is None or den == 0:
        return None
    out = num / den
    return None if (math.isnan(out) or math.isinf(out)) else out


def _metric(value: float | None, formula: str, inputs: dict) -> dict:
    """A derived number with its provenance: {"value", "formula", "inputs"}.

    `inputs` carries the RAW reported fields the value was computed from (None included, so a
    reader can see exactly what was and wasn't available). ORACLE's citation checker recomputes
    `formula` from `inputs` — a metric without this cannot be verified and counts as a
    candidate fabrication, so every metric below must build one of these.
    """
    return {"value": value, "formula": formula, "inputs": inputs}


def _sum_present(*vals: float | None) -> float | None:
    """Sum of the non-None values, or None if every value is missing."""
    present = [v for v in vals if v is not None]
    return sum(present) if present else None


def derived_metrics(fin: dict) -> dict:
    """Per-fiscal-year ratios computed from a `financials()` dict. Pure — no I/O, no network.

        {"ticker", "fiscal_years": [{"fy", "end", "metrics": {name: {"value","formula","inputs"}}}]}

    Every metric is None-safe (missing input -> value None) and returns None rather than
    raising on a zero denominator. Nothing here is estimated: a None value means the inputs
    weren't reported, not that the ratio is zero.

    Definitions and their limitations:
      - gross_margin — gross_profit / revenue; when the issuer never tags GrossProfit it falls
        back to (revenue - cost_of_revenue) / revenue. `formula` says which branch ran.
      - fcf — operating_cash_flow - capital_expenditures. us-gaap reports capex POSITIVE (it is
        a payment), so it subtracts. Excludes acquisitions, capitalized leases and, notably,
        does not add back stock-based compensation.
      - roe — net_income / stockholders_equity at the fiscal year END (not average equity), and
        equity is the parent-only figure when the issuer tags one, so buyback-heavy or
        NCI-heavy names read high.
      - roic_proxy — NOPAT / invested capital, i.e.
        operating_income * (1 - tax_rate) / (short_term_debt + long_term_debt +
        stockholders_equity - cash_and_equivalents), with
        tax_rate = income_tax_expense / (net_income + income_tax_expense).
        It is a PROXY, deliberately: the tax rate uses a pretax proxy (net income + tax expense)
        because pretax income isn't extracted, so discontinued operations, noncontrolling
        interests and one-off tax items distort it; invested capital uses ending balances, the
        parent-only equity, and ignores operating leases, goodwill and pension obligations.
        Returns None when operating income isn't tagged (6/39 universe names), when the pretax
        proxy is <= 0, or when the implied tax rate falls outside [0, 1] — a nonsense rate would
        produce a confidently wrong number, which is worse than a blank.
      - net_debt / debt_to_ebitda_proxy — total debt = short_term_debt + long_term_debt, where a
        component the issuer never tagged is treated as absent (Meta reports no current debt);
        `inputs` shows the None so the omission is visible. EBITDA proxy = operating_income +
        depreciation_amortization, which is not a GAAP measure and ignores one-off charges;
        None when EBITDA proxy <= 0.
      - revenue_growth_yoy — versus the immediately preceding row in this dict. None for the
        oldest row and whenever the two years aren't adjacent fiscal years in the data.
    """
    rows = fin.get("fiscal_years", []) or []
    out_years: list[dict] = []
    for i, r in enumerate(rows):
        prior = rows[i + 1] if i + 1 < len(rows) else {}  # rows are newest-first
        rev = r.get("revenue")
        gp, cor = r.get("gross_profit"), r.get("cost_of_revenue")
        op, ni = r.get("operating_income"), r.get("net_income")
        tax, eq = r.get("income_tax_expense"), r.get("stockholders_equity")
        ocf, capex = r.get("operating_cash_flow"), r.get("capital_expenditures")
        cash, std, ltd = (
            r.get("cash_and_equivalents"),
            r.get("short_term_debt"),
            r.get("long_term_debt"),
        )
        da = r.get("depreciation_amortization")

        m: dict[str, dict] = {}

        # --- margins
        if gp is None and cor is not None and rev is not None:
            m["gross_margin"] = _metric(
                _safe_div(rev - cor, rev),
                "(revenue - cost_of_revenue) / revenue",
                {"revenue": rev, "cost_of_revenue": cor},
            )
        else:
            m["gross_margin"] = _metric(
                _safe_div(gp, rev), "gross_profit / revenue", {"gross_profit": gp, "revenue": rev}
            )
        m["operating_margin"] = _metric(
            _safe_div(op, rev),
            "operating_income / revenue",
            {"operating_income": op, "revenue": rev},
        )
        m["net_margin"] = _metric(
            _safe_div(ni, rev), "net_income / revenue", {"net_income": ni, "revenue": rev}
        )

        # --- cash generation (capex is reported positive -> it subtracts)
        fcf = None if (ocf is None or capex is None) else ocf - capex
        fcf_inputs = {"operating_cash_flow": ocf, "capital_expenditures": capex}
        m["fcf"] = _metric(fcf, "operating_cash_flow - capital_expenditures", fcf_inputs)
        m["fcf_margin"] = _metric(
            _safe_div(fcf, rev),
            "(operating_cash_flow - capital_expenditures) / revenue",
            {**fcf_inputs, "revenue": rev},
        )

        # --- returns
        m["roe"] = _metric(
            _safe_div(ni, eq),
            "net_income / stockholders_equity",
            {"net_income": ni, "stockholders_equity": eq},
        )

        pretax_proxy = None if (ni is None or tax is None) else ni + tax
        tax_rate = _safe_div(tax, pretax_proxy) if (pretax_proxy or 0) > 0 else None
        if tax_rate is not None and not (0.0 <= tax_rate <= 1.0):
            tax_rate = None  # nonsense effective rate -> refuse to publish a number
        invested = _sum_present(std, ltd, eq)
        invested = None if (invested is None or cash is None) else invested - cash
        nopat = None if (op is None or tax_rate is None) else op * (1.0 - tax_rate)
        m["roic_proxy"] = _metric(
            _safe_div(nopat, invested),
            "operating_income * (1 - income_tax_expense / (net_income + income_tax_expense)) / "
            "(short_term_debt + long_term_debt + stockholders_equity - cash_and_equivalents)",
            {
                "operating_income": op,
                "income_tax_expense": tax,
                "net_income": ni,
                "short_term_debt": std,
                "long_term_debt": ltd,
                "stockholders_equity": eq,
                "cash_and_equivalents": cash,
            },
        )

        # --- leverage
        total_debt = _sum_present(std, ltd)
        debt_inputs = {"short_term_debt": std, "long_term_debt": ltd}
        m["net_debt"] = _metric(
            None if (total_debt is None or cash is None) else total_debt - cash,
            "short_term_debt + long_term_debt - cash_and_equivalents",
            {**debt_inputs, "cash_and_equivalents": cash},
        )
        ebitda_proxy = None if (op is None or da is None) else op + da
        m["debt_to_ebitda_proxy"] = _metric(
            _safe_div(total_debt, ebitda_proxy) if (ebitda_proxy or 0) > 0 else None,
            "(short_term_debt + long_term_debt) / (operating_income + depreciation_amortization)",
            {**debt_inputs, "operating_income": op, "depreciation_amortization": da},
        )

        # --- growth
        prior_rev = prior.get("revenue")
        m["revenue_growth_yoy"] = _metric(
            None if (rev is None or prior_rev is None) else _safe_div(rev - prior_rev, prior_rev),
            "(revenue - revenue_prior) / revenue_prior",
            {"revenue": rev, "revenue_prior": prior_rev, "prior_fy": prior.get("fy")},
        )

        out_years.append({"fy": r.get("fy"), "end": r.get("end"), "metrics": m})

    return {"ticker": fin.get("ticker"), "fiscal_years": out_years}


def sync(tickers: list[str] | None = None, cache_dir: Path = FUND_DIR) -> dict[str, bool]:
    """Pull + cache companyfacts for `tickers` (default universe). {ticker: ok}."""
    out: dict[str, bool] = {}
    for t in [t.upper() for t in (tickers or load_universe())]:
        try:
            out[t] = fetch_companyfacts(t, cache_dir=cache_dir, refresh=True) is not None
        except Exception:  # noqa: BLE001 — one bad ticker never aborts the sync
            out[t] = False
    return out
