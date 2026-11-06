"""EDGAR fundamentals tests with synthetic annual revenue and ticker fixtures.

Companyfacts examples are built inline. Reads use a temporary cache and need no SEC calls.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest

from trident.data import fundamentals as f

FIX = Path(__file__).parent / "fixtures"


def _seed_cache(cache_dir: Path):
    """Drop the fixtures where fundamentals.py's cache-first reads expect them."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / "edgar_ticker_cik.json", cache_dir / "_ticker_cik.json")
    shutil.copy(FIX / "edgar_demo_revenue.json", cache_dir / "DEMO.json")


# --------------------------------------------------- synthetic companyfacts builders
def _entry(end: str, val: float, filed: str) -> dict:
    """One annual 10-K fact in EDGAR's shape (365-day period -> passes the annual filter)."""
    start = (date.fromisoformat(end) - timedelta(days=364)).isoformat()
    return {
        "start": start,
        "end": end,
        "val": val,
        "fy": date.fromisoformat(end).year,
        "fp": "FY",
        "form": "10-K",
        "filed": filed,
        "frame": None,
    }


def _cf(
    series: dict[str, dict[str, float]],
    unit_of: dict[str, str] | None = None,
    filed: str = "2026-02-15",
) -> dict:
    """{concept: {period-end: value}} -> a minimal companyfacts document."""
    gaap = {
        concept: {
            "units": {
                (unit_of or {}).get(concept, "USD"): [_entry(e, v, filed) for e, v in years.items()]
            }
        }
        for concept, years in series.items()
    }
    return {
        "cik": 1,
        "entityName": "TEST CO",
        "facts": {"us-gaap": gaap},
        "_fetched_at": "2026-07-21T00:00:00+00:00",
    }


def _seed_synthetic(cache_dir: Path, ticker: str, facts: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{ticker}.json").write_text(json.dumps(facts))


def _fin(*years: dict) -> dict:
    """A financials()-shaped dict from partial rows (newest first); unset fields are None."""
    rows = []
    for i, y in enumerate(years):
        row = {name: None for name, _, _ in f._LINE_ITEMS}
        row |= {"fy": 2025 - i, "end": f"{2025 - i}-12-31", "filed": "2026-02-15", "form": "10-K"}
        row |= y
        rows.append(row)
    return {
        "ticker": "TEST",
        "fetched_at": "2026-07-21T00:00:00+00:00",
        "fiscal_years": rows,
        "concepts_used": {},
        "missing_concepts": [],
    }


# ------------------------------------------------------------------ CIK map
def test_cik_lookup_zero_pads(tmp_path):
    _seed_cache(tmp_path)
    assert f.cik_for("DEMO", cache_dir=tmp_path) == "0000000123"
    assert f.cik_for("demo", cache_dir=tmp_path) == "0000000123"  # case-insensitive
    assert f.cik_for("MOCK", cache_dir=tmp_path) == "0000000456"
    assert f.cik_for("NOTREAL", cache_dir=tmp_path) is None


def test_cik_lookup_normalizes_dot_to_dash(tmp_path):
    """SEC spells class shares BRK-B; universe.yaml (and everything downstream) says BRK.B."""
    registry = {
        "0": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
        "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "_ticker_cik.json").write_text(json.dumps(registry))

    assert f.cik_for("BRK.B", cache_dir=tmp_path) == "0001067983"  # the bug: was None
    assert f.cik_for("brk.b", cache_dir=tmp_path) == "0001067983"
    assert f.cik_for("BRK-B", cache_dir=tmp_path) == "0001067983"  # SEC spelling still works
    assert f.cik_for("AAPL", cache_dir=tmp_path) == "0000320193"  # dotless names unaffected
    assert f.cik_for("NOPE.X", cache_dir=tmp_path) is None  # normalization isn't a wildcard


@pytest.mark.skipif(
    not (f.FUND_DIR / "_ticker_cik.json").exists(),
    reason="local EDGAR ticker cache absent (would require network)",
)
def test_cik_lookup_normalizes_against_real_cached_registry():
    """Same check against the real cached SEC registry — still offline (cache-first)."""
    assert f.cik_map().get("BRK.B") is None  # SEC genuinely has no dotted key
    assert f.cik_for("BRK.B") == "0001067983"


def test_cik_override_beats_the_registry(monkeypatch, tmp_path):
    """A pinned CIK wins over SEC's registry — the registry is not always right.

    SEC maps ticker XOM to a 2026 filing shell (CIK 2115436) whose companyfacts carry ZERO
    us-gaap facts, so an unpinned XOM reads as 'reports nothing' instead of raising. That is
    a silent data loss, which is why the override exists at all (golden rule 8).
    """
    registry = {"0": {"cik_str": 2115436, "ticker": "XOM", "title": "EXXON MOBIL CORP"}}
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "_ticker_cik.json").write_text(json.dumps(registry))

    monkeypatch.setattr(f, "load_cik_overrides", lambda: {})
    assert f.cik_for("XOM", cache_dir=tmp_path) == "0002115436"  # unpinned: the wrong shell

    monkeypatch.setattr(f, "load_cik_overrides", lambda: {"XOM": "0000034088"})
    assert f.cik_for("XOM", cache_dir=tmp_path) == "0000034088"  # pinned: the real company
    # an override for one ticker must not shadow every other lookup
    assert f.cik_for("NOTREAL", cache_dir=tmp_path) is None


def test_shipped_cik_override_config_is_wellformed():
    """The real universe.yaml pin parses and is zero-padded — config typos fail here, not live."""
    from trident.config import load_cik_overrides

    assert load_cik_overrides()["XOM"] == "0000034088"


# --------------------------------------------------------------- extraction
def test_annual_concept_dedupes_by_year_and_sorts_desc():
    facts = json.loads((FIX / "edgar_demo_revenue.json").read_text())
    df = f.annual_concept(facts, f._REVENUE_CONCEPTS)
    assert not df.empty
    assert list(df.columns) == f._FACT_COLS
    # newest first, all 10-K, one row per fiscal year-end (no dupes)
    assert df["end"].is_monotonic_decreasing
    assert (df["form"] == "10-K").all()
    assert df["end"].dt.year.is_unique or df["end"].nunique() == len(df)


def test_revenue_returns_five_years_with_expected_values(tmp_path):
    _seed_cache(tmp_path)
    rev = f.revenue("DEMO", years=5, cache_dir=tmp_path)
    assert len(rev) == 5
    assert rev["end"].dt.year.tolist() == [2025, 2024, 2023, 2022, 2021]
    # Use the later filing for FY2025 and ignore the quarterly fact.
    assert rev["val"].tolist() == [450, 400, 350, 300, 250]


def test_missing_concept_returns_empty_frame():
    df = f.annual_concept({"facts": {"us-gaap": {}}}, f._REVENUE_CONCEPTS)
    assert df.empty
    assert list(df.columns) == f._FACT_COLS


# ------------------------------------------------------- statements (financials)
def test_financials_falls_back_to_second_concept_in_the_list(tmp_path):
    """Nothing tags the preferred concept -> the next name in the list supplies the year."""
    assert f._REVENUE_CONCEPTS[1] == "Revenues"  # the fallback under test
    facts = _cf({"Revenues": {"2025-12-31": 500.0, "2024-12-31": 400.0}})
    _seed_synthetic(tmp_path, "TEST", facts)

    fin = f.financials("TEST", cache_dir=tmp_path)

    assert [r["revenue"] for r in fin["fiscal_years"]] == [500.0, 400.0]
    assert fin["concepts_used"]["revenue"] == ["Revenues"]
    assert "revenue" not in fin["missing_concepts"]


def test_financials_resolves_concept_preference_per_year(tmp_path):
    """Alphabet's live pattern: FY2025 only under `Revenues`, FY2023 only under the
    contract-revenue tag. Both years must survive, with both tags disclosed."""
    facts = _cf(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "2024-12-31": 350.0,
                "2023-12-31": 307.0,
            },
            "Revenues": {"2025-12-31": 402.0, "2024-12-31": 350.0},
        }
    )
    _seed_synthetic(tmp_path, "TEST", facts)

    fin = f.financials("TEST", cache_dir=tmp_path)

    assert [(r["fy"], r["revenue"]) for r in fin["fiscal_years"]] == [
        (2025, 402.0),
        (2024, 350.0),
        (2023, 307.0),
    ]
    # FY2024 is tagged both ways -> the preferred concept wins; the switch is disclosed.
    assert fin["concepts_used"]["revenue"] == [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ]


def test_financials_reports_missing_concepts_without_crashing(tmp_path):
    """A wholly untagged line item -> named in missing_concepts, None in every row."""
    facts = _cf({"Revenues": {"2025-12-31": 500.0}, "NetIncomeLoss": {"2025-12-31": 50.0}})
    _seed_synthetic(tmp_path, "TEST", facts)

    fin = f.financials("TEST", cache_dir=tmp_path)

    row = fin["fiscal_years"][0]
    for field in ("gross_profit", "inventory", "capital_expenditures", "eps_diluted"):
        assert field in fin["missing_concepts"]
        assert row[field] is None
    assert "revenue" not in fin["missing_concepts"]
    # every declared line item is present as a key, so the row shape is stable for JSONB
    assert {name for name, _, _ in f._LINE_ITEMS} <= set(row)


def test_financials_is_json_serializable_and_bounded(tmp_path):
    _seed_cache(tmp_path)  # the synthetic fixture carries the revenue concept only
    fin = f.financials("DEMO", years=3, cache_dir=tmp_path)

    assert fin["ticker"] == "DEMO"
    assert len(fin["fiscal_years"]) == 3
    assert fin["fiscal_years"][0]["end"] == "2025-12-31"
    assert fin["fiscal_years"][0]["revenue"] == 450
    assert [r["end"] for r in fin["fiscal_years"]] == sorted(
        (r["end"] for r in fin["fiscal_years"]), reverse=True
    )
    assert json.loads(json.dumps(fin)) == fin  # no numpy scalars, no NaN, no Timestamps
    assert json.loads(json.dumps(f.derived_metrics(fin))) == f.derived_metrics(fin)


def test_financials_unknown_ticker_returns_empty_shape(tmp_path):
    """No cache and no CIK -> a well-formed empty answer, not an exception or a fetch."""
    (tmp_path / "_ticker_cik.json").write_text(json.dumps({}))
    fin = f.financials("NOTREAL", cache_dir=tmp_path)
    assert fin["fiscal_years"] == []
    assert fin["missing_concepts"] == [name for name, _, _ in f._LINE_ITEMS]


def test_units_are_respected_for_per_share_and_share_counts(tmp_path):
    """EPS lives in USD/shares and share counts in `shares` — a USD-only read loses both."""
    facts = _cf(
        {
            "Revenues": {"2025-12-31": 500.0},
            "EarningsPerShareDiluted": {"2025-12-31": 7.46},
            "WeightedAverageNumberOfDilutedSharesOutstanding": {"2025-12-31": 15.0e9},
        },
        unit_of={
            "EarningsPerShareDiluted": "USD/shares",
            "WeightedAverageNumberOfDilutedSharesOutstanding": "shares",
        },
    )
    _seed_synthetic(tmp_path, "TEST", facts)

    row = f.financials("TEST", cache_dir=tmp_path)["fiscal_years"][0]
    assert row["eps_diluted"] == 7.46
    assert row["diluted_shares"] == 15.0e9


# ------------------------------------------------------------- derived metrics
def test_derived_metrics_core_values():
    fin = _fin(
        {
            "revenue": 1000.0,
            "cost_of_revenue": 600.0,
            "gross_profit": 400.0,
            "operating_income": 250.0,
            "net_income": 200.0,
            "income_tax_expense": 50.0,
            "stockholders_equity": 800.0,
            "cash_and_equivalents": 100.0,
            "short_term_debt": 50.0,
            "long_term_debt": 150.0,
            "depreciation_amortization": 50.0,
            "operating_cash_flow": 300.0,
            "capital_expenditures": 100.0,
        },
        {"revenue": 800.0},
    )
    m = f.derived_metrics(fin)["fiscal_years"][0]["metrics"]

    assert m["gross_margin"]["value"] == pytest.approx(0.4)
    assert m["operating_margin"]["value"] == pytest.approx(0.25)
    assert m["net_margin"]["value"] == pytest.approx(0.2)
    assert m["fcf"]["value"] == pytest.approx(200.0)
    assert m["fcf_margin"]["value"] == pytest.approx(0.2)
    assert m["roe"]["value"] == pytest.approx(0.25)
    # NOPAT = 250 * (1 - 50/250) = 200; invested = 50 + 150 + 800 - 100 = 900
    assert m["roic_proxy"]["value"] == pytest.approx(200.0 / 900.0)
    assert m["net_debt"]["value"] == pytest.approx(100.0)
    assert m["debt_to_ebitda_proxy"]["value"] == pytest.approx(200.0 / 300.0)
    assert m["revenue_growth_yoy"]["value"] == pytest.approx(0.25)
    # oldest row has no prior year to compare against — None, not an exception
    assert (
        f.derived_metrics(fin)["fiscal_years"][1]["metrics"]["revenue_growth_yoy"]["value"] is None
    )


def test_fcf_subtracts_capex_reported_positive():
    """us-gaap reports capex as a positive payment, so FCF = OCF - capex (not OCF + capex)."""
    fin = _fin({"revenue": 1000.0, "operating_cash_flow": 300.0, "capital_expenditures": 120.0})
    m = f.derived_metrics(fin)["fiscal_years"][0]["metrics"]

    assert m["fcf"]["value"] == pytest.approx(180.0)
    assert m["fcf"]["value"] < 300.0  # the sign error would give 420
    assert m["fcf_margin"]["value"] == pytest.approx(0.18)
    assert m["fcf"]["inputs"] == {"operating_cash_flow": 300.0, "capital_expenditures": 120.0}


def test_gross_margin_falls_back_to_revenue_minus_cost_of_revenue():
    """23/39 universe names never tag GrossProfit; the fallback must say which branch ran."""
    fin = _fin({"revenue": 1000.0, "cost_of_revenue": 550.0})
    gm = f.derived_metrics(fin)["fiscal_years"][0]["metrics"]["gross_margin"]

    assert gm["value"] == pytest.approx(0.45)
    assert gm["formula"] == "(revenue - cost_of_revenue) / revenue"
    assert gm["inputs"] == {"revenue": 1000.0, "cost_of_revenue": 550.0}


def test_derived_metrics_return_none_on_zero_denominator():
    """Divide-by-zero yields None, never a ZeroDivisionError (fail safe, not silent)."""
    fin = _fin(
        {
            "revenue": 0.0,
            "gross_profit": 10.0,
            "operating_income": 0.0,
            "net_income": 5.0,
            "income_tax_expense": 0.0,
            "stockholders_equity": 0.0,
            "cash_and_equivalents": 0.0,
            "short_term_debt": 0.0,
            "long_term_debt": 0.0,
            "depreciation_amortization": 0.0,
            "operating_cash_flow": 10.0,
            "capital_expenditures": 4.0,
        },
        {"revenue": 0.0},
    )
    m = f.derived_metrics(fin)["fiscal_years"][0]["metrics"]

    for name in (
        "gross_margin",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "roe",
        "roic_proxy",
        "debt_to_ebitda_proxy",
        "revenue_growth_yoy",
    ):
        assert m[name]["value"] is None, name
    assert m["fcf"]["value"] == pytest.approx(6.0)  # a difference, not a ratio — still computable


def test_derived_metrics_are_none_safe_on_a_totally_empty_year():
    m = f.derived_metrics(_fin({}))["fiscal_years"][0]["metrics"]
    assert all(v["value"] is None for v in m.values())
    assert f.derived_metrics({"ticker": "X", "fiscal_years": []})["fiscal_years"] == []
    assert f.derived_metrics({})["fiscal_years"] == []


def test_every_derived_metric_carries_input_provenance():
    """Load-bearing: ORACLE's citation checker recomputes cited numbers from `inputs`.
    A metric without them cannot be verified and counts as a candidate fabrication."""
    fin = _fin(
        {
            "revenue": 1000.0,
            "gross_profit": 400.0,
            "operating_income": 250.0,
            "net_income": 200.0,
            "income_tax_expense": 50.0,
            "stockholders_equity": 800.0,
            "cash_and_equivalents": 100.0,
            "short_term_debt": 50.0,
            "long_term_debt": 150.0,
            "depreciation_amortization": 50.0,
            "operating_cash_flow": 300.0,
            "capital_expenditures": 100.0,
        },
        {"revenue": 800.0},
    )
    dm = f.derived_metrics(fin)
    line_items = {name for name, _, _ in f._LINE_ITEMS}

    for year in dm["fiscal_years"]:
        assert year["metrics"], year
        for name, metric in year["metrics"].items():
            assert set(metric) == {"value", "formula", "inputs"}, name
            assert isinstance(metric["formula"], str) and metric["formula"], name
            assert isinstance(metric["inputs"], dict) and metric["inputs"], name
            # inputs name RAW reported fields (plus the growth metric's prior-year pointers)
            assert set(metric["inputs"]) <= line_items | {"revenue_prior", "prior_fy"}, name

    # and the recomputation actually closes, which is the whole point
    gm = dm["fiscal_years"][0]["metrics"]["gross_margin"]
    assert gm["inputs"]["gross_profit"] / gm["inputs"]["revenue"] == pytest.approx(gm["value"])


def test_provenance_survives_missing_inputs():
    """A None value still shows what was looked for — a blank with no trail is unauditable."""
    m = f.derived_metrics(_fin({"revenue": 1000.0}))["fiscal_years"][0]["metrics"]
    assert m["roe"]["value"] is None
    assert m["roe"]["inputs"] == {"net_income": None, "stockholders_equity": None}
    assert m["net_debt"]["inputs"]["cash_and_equivalents"] is None


def test_roic_proxy_refuses_a_nonsense_tax_rate():
    """A negative pretax proxy (loss year with a tax expense) would invert NOPAT — return None."""
    fin = _fin(
        {
            "revenue": 1000.0,
            "operating_income": 100.0,
            "net_income": -200.0,
            "income_tax_expense": 50.0,
            "stockholders_equity": 500.0,
            "cash_and_equivalents": 50.0,
            "long_term_debt": 100.0,
        }
    )
    roic = f.derived_metrics(fin)["fiscal_years"][0]["metrics"]["roic_proxy"]
    assert roic["value"] is None
    assert roic["inputs"]["net_income"] == -200.0  # still auditable
