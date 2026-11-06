"""Market-bar cache tests — offline. A synthetic parquet cache stands in for the
providers; `_fetch` is monkeypatched to prove the cache-first contract never hits the
network when the cache already covers the range (Phase 1 acceptance: warm reads)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from trident.data import market


def _write_cache(cache_dir, ticker, start, end):
    idx = pd.bdate_range(start, end, name="date")
    df = pd.DataFrame(
        {
            "open": range(len(idx)),
            "high": range(len(idx)),
            "low": range(len(idx)),
            "close": [100 + i for i in range(len(idx))],
            "volume": [1_000 + i for i in range(len(idx))],
        },
        index=idx,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(market._cache_path(ticker, cache_dir))
    return df


# --------------------------------------------------------------- symbology
def test_symbol_for_maps_per_provider():
    assert market._symbol_for("BRK.B", "alpaca") == "BRK.B"
    assert market._symbol_for("BRK.B", "yfinance") == "BRK-B"
    assert market._symbol_for("BRK.B", "stooq") == "brk-b.us"
    assert market._symbol_for("aapl", "yfinance") == "AAPL"  # upper-cased


# ------------------------------------------------------- cache-first reads
def test_warm_read_serves_from_cache_without_fetching(tmp_path, monkeypatch):
    _write_cache(tmp_path, "AAPL", "2024-01-01", "2024-03-01")

    def _boom(*a, **k):
        raise AssertionError("_fetch must NOT be called when the cache covers the range")

    monkeypatch.setattr(market, "_fetch", _boom)
    out = market.get_history(["AAPL"], "2024-01-10", "2024-02-10", cache_dir=tmp_path)
    assert list(out.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert out["ticker"].unique().tolist() == ["AAPL"]
    assert out["date"].min().date() >= date(2024, 1, 10)
    assert out["date"].max().date() <= date(2024, 2, 10)
    assert out["date"].is_monotonic_increasing


def test_cache_miss_triggers_fetch_then_caches(tmp_path, monkeypatch):
    idx = pd.bdate_range("2024-01-01", "2024-01-31", name="date")
    synthetic = pd.DataFrame(
        {c: range(len(idx)) for c in ["open", "high", "low", "close", "volume"]}, index=idx
    )
    calls = []

    def _fake_fetch(ticker, start, end):
        calls.append(ticker)
        return synthetic

    monkeypatch.setattr(market, "_fetch", _fake_fetch)
    out = market.get_history(["MSFT"], "2024-01-05", "2024-01-20", cache_dir=tmp_path)
    assert calls == ["MSFT"]  # fetched once
    assert market._cache_path("MSFT", tmp_path).exists()  # and cached
    assert not out.empty and out["ticker"].iloc[0] == "MSFT"


def test_missing_data_returns_empty_tidy_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(market, "_fetch", lambda *a, **k: None)  # provider has nothing
    out = market.get_history(["ZZZZ"], "2024-01-01", "2024-01-31", cache_dir=tmp_path)
    assert out.empty
    assert list(out.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]


def test_multi_ticker_tidy_sorted(tmp_path, monkeypatch):
    _write_cache(tmp_path, "AAPL", "2024-01-01", "2024-03-01")
    _write_cache(tmp_path, "MSFT", "2024-01-01", "2024-03-01")
    monkeypatch.setattr(market, "_fetch", lambda *a, **k: None)
    out = market.get_history(["MSFT", "AAPL"], "2024-02-01", "2024-02-15", cache_dir=tmp_path)
    # grouped by ticker, each ascending by date
    assert out["ticker"].tolist() == sorted(out["ticker"].tolist())
    for _, g in out.groupby("ticker"):
        assert g["date"].is_monotonic_increasing


def test_refresh_forces_fetch_even_with_cache(tmp_path, monkeypatch):
    _write_cache(tmp_path, "AAPL", "2024-01-01", "2024-03-01")
    idx = pd.bdate_range("2024-01-01", "2024-03-01", name="date")
    synthetic = pd.DataFrame(
        {c: range(len(idx)) for c in ["open", "high", "low", "close", "volume"]}, index=idx
    )
    called = []
    monkeypatch.setattr(market, "_fetch", lambda t, s, e: called.append(t) or synthetic)
    market.get_history(["AAPL"], "2024-02-01", "2024-02-15", cache_dir=tmp_path, refresh=True)
    assert called == ["AAPL"]
