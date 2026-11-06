"""Daily bars for the universe (doc 01/02).

Alpaca primary (IEX daily bars), yfinance backfill to 2015; parquet cache under
data/cache/bars/. `get_history` returns a tidy long DataFrame [date, ticker, open,
high, low, close, volume], corporate-action-adjusted (splits + dividends).

Rules:
  - NEVER used at execution time — the broker's live price is the only execution price
    (golden rule 8). yfinance/Stooq are unofficial: research/backtest only.
  - Stored bars reflect TODAY's corp-action history, so they carry look-ahead if used
    naively in a backtest — backtests must apply point-in-time logic (backtest-protocol).
  - Symbology differs per provider (BRK.B): normalized at the boundary (`_symbol_for`).
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from trident.config import CACHE_DIR, load_universe
from trident.data import calendar as trading_calendar

BARS_DIR = CACHE_DIR / "bars"
DEFAULT_START = date(2015, 1, 1)  # yfinance backfill floor (doc 02)
_OHLCV = ["open", "high", "low", "close", "volume"]
_COVER_TOL_DAYS = 7  # allow the requested start to be a weekend/holiday near a provider's first bar


# ------------------------------------------------------------------ helpers
def _as_date(d: str | date | datetime) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def _symbol_for(ticker: str, provider: str) -> str:
    """Map a Trident ticker to a provider's symbology. BRK.B is the live example."""
    t = ticker.upper()
    if provider == "yfinance":
        return t.replace(".", "-")  # BRK.B -> BRK-B
    if provider == "stooq":
        return t.replace(".", "-").lower() + ".us"  # BRK.B -> brk-b.us
    return t  # alpaca uses BRK.B as-is


def _cache_path(ticker: str, cache_dir: Path) -> Path:
    return cache_dir / f"{ticker.upper()}.parquet"


def _safe_last_completed(end: date) -> date:
    """The latest date bars should exist for, capped at `end`; degrades to `end` on error."""
    try:
        return min(end, trading_calendar.last_completed_session())
    except Exception:  # noqa: BLE001 — calendar is best-effort here
        return end


# ---------------------------------------------------------------- providers
# Each returns a DatetimeIndex(date, tz-naive UTC midnight) DataFrame with _OHLCV, or None.
def _fetch_alpaca(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        from alpaca.data.enums import Adjustment
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        from trident.config import get_settings

        key, secret, _ = get_settings().broker_credentials()
        client = StockHistoricalDataClient(key, secret)
        req = StockBarsRequest(
            symbol_or_symbols=_symbol_for(ticker, "alpaca"),
            timeframe=TimeFrame.Day,
            start=str(start),
            end=str(end),
            adjustment=Adjustment.ALL,  # splits + dividends
        )
        df = client.get_stock_bars(req).df
        if df is None or df.empty:
            return None
        df = df.reset_index()  # MultiIndex (symbol, timestamp) -> columns
        idx = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
        out = df[_OHLCV].copy()
        out.index = pd.DatetimeIndex(idx, name="date")
        return out.sort_index()
    except Exception:  # noqa: BLE001 — any provider failure -> try the next
        return None


def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        import yfinance as yf

        # yfinance `end` is exclusive; +1 day to include it.
        h = yf.Ticker(_symbol_for(ticker, "yfinance")).history(
            start=str(start),
            end=str(pd.Timestamp(end) + pd.Timedelta(days=1)),
            auto_adjust=True,  # splits + dividends
            raise_errors=False,
        )
        if h is None or h.empty:
            return None
        h = h.rename(columns=str.lower)
        if not set(_OHLCV).issubset(h.columns):
            return None
        out = h[_OHLCV].copy()
        out.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        out.index.name = "date"
        return out.sort_index()
    except Exception:  # noqa: BLE001
        return None


def _fetch(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """Alpaca primary; fall back to yfinance when Alpaca doesn't reach `start` (doc 02)."""
    alp = _fetch_alpaca(ticker, start, end)
    if alp is not None and not alp.empty:
        covers_start = (alp.index.min().date() - start).days <= _COVER_TOL_DAYS
        if covers_start:
            return alp
    yfin = _fetch_yfinance(ticker, start, end)
    if yfin is not None and not yfin.empty:
        return yfin
    return alp  # partial Alpaca history beats nothing


# -------------------------------------------------------------------- cache
def _read_cache(ticker: str, cache_dir: Path) -> pd.DataFrame | None:
    path = _cache_path(ticker, cache_dir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def _merge_write(
    ticker: str, cached: pd.DataFrame | None, fresh: pd.DataFrame, cache_dir: Path
) -> pd.DataFrame:
    """Union cached + fresh (fresh wins on overlap — it is re-adjusted/newer), then persist."""
    combined = fresh if cached is None else pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_cache_path(ticker, cache_dir))
    return combined


# ------------------------------------------------------------------ public
def sync(
    tickers: list[str] | None = None,
    start: str | date = DEFAULT_START,
    end: str | date | None = None,
    cache_dir: Path = BARS_DIR,
) -> dict[str, int]:
    """Refresh the parquet cache for `tickers` (default: universe) over [start, end].
    Returns {ticker: rows_cached}. The cold path; run once then nightly."""
    tickers = [t.upper() for t in (tickers or load_universe())]
    start = _as_date(start)
    end = _as_date(end) if end is not None else datetime.now(UTC).date()
    result: dict[str, int] = {}
    for t in tickers:
        fresh = _fetch(t, start, end)
        if fresh is None or fresh.empty:
            result[t] = 0
            continue
        merged = _merge_write(t, _read_cache(t, cache_dir), fresh, cache_dir)
        result[t] = len(merged)
    return result


def get_history(
    tickers: list[str] | str,
    start: str | date,
    end: str | date,
    cache_dir: Path = BARS_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Tidy long bars [date, ticker, open, high, low, close, volume] for [start, end].

    Cache-first: a ticker whose cache is missing or doesn't cover the range is fetched
    (from DEFAULT_START so the cache accrues full history) and cached, then served.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [t.upper() for t in tickers]
    start, end = _as_date(start), _as_date(end)
    required_end = _safe_last_completed(end)

    frames: list[pd.DataFrame] = []
    for t in tickers:
        cached = None if refresh else _read_cache(t, cache_dir)
        # Fetch only when the cache is missing, stale at the top, or the request reaches
        # BELOW our backfill floor (which a present cache is always filled to). A start
        # that merely predates a ticker's first-ever bar is NOT a miss — no earlier data
        # exists — so we must not refetch there (that was the 40x warm-read regression).
        need = (
            cached is None
            or cached.index.max().date() < required_end
            or (start < DEFAULT_START and cached.index.min().date() > start)
        )
        if need:
            fresh = _fetch(t, min(start, DEFAULT_START), end)
            if fresh is not None and not fresh.empty:
                cached = _merge_write(t, cached, fresh, cache_dir)
        if cached is None or cached.empty:
            continue
        # Label-slice a sorted DatetimeIndex (O(log n)) instead of a per-row .date mask.
        sub = cached.loc[str(start) : str(end)].copy()
        if sub.empty:
            continue
        sub.insert(0, "ticker", t)
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", *_OHLCV])
    out = pd.concat(frames).reset_index().rename(columns={"index": "date"})
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
