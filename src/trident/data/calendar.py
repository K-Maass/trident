"""NYSE trading calendar (XNYS) — the scheduler's source of truth (doc 01/09).

Thin wrapper over pandas-market-calendars. All returned datetimes are tz-aware UTC;
sessions are plain dates. Used for: backtest date ranges, "was yesterday a trading day",
freshness bounds, and market-state checks the broker clock doesn't cover.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from functools import lru_cache

import pandas_market_calendars as mcal

_CAL_NAME = "XNYS"
_HORIZON = timedelta(days=10)  # enough to cross any holiday cluster to the next session
_LOOKBACK = timedelta(days=10)


@lru_cache(maxsize=1)
def _calendar():
    return mcal.get_calendar(_CAL_NAME)


def _now_utc(ts: datetime | None) -> datetime:
    if ts is None:
        return datetime.now(UTC)
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def sessions(start: date, end: date) -> list[date]:
    """Trading days (dates) in [start, end] inclusive."""
    days = _calendar().valid_days(start_date=start, end_date=end)
    return [d.date() for d in days]


def is_session(day: date) -> bool:
    """True if `day` is a full or half trading session."""
    return len(sessions(day, day)) == 1


def is_open(ts: datetime | None = None) -> bool:
    """True if the exchange is in its regular session at `ts` (defaults to now, UTC).

    Half-days are handled (schedule carries the early close). Never guesses: outside a
    session or on a holiday returns False.
    """
    ts = _now_utc(ts)
    sched = _calendar().schedule(start_date=ts.date(), end_date=ts.date())
    if sched.empty:
        return False
    open_ = sched["market_open"].iloc[0].to_pydatetime()
    close_ = sched["market_close"].iloc[0].to_pydatetime()
    return open_ <= ts <= close_


def next_open(ts: datetime | None = None) -> datetime:
    """Next regular-session open at/after `ts` (UTC)."""
    ts = _now_utc(ts)
    sched = _calendar().schedule(start_date=ts.date(), end_date=(ts + _HORIZON).date())
    for open_ in sched["market_open"]:
        o = open_.to_pydatetime()
        if o >= ts:
            return o
    raise RuntimeError("no session open within lookahead horizon")


def next_close(ts: datetime | None = None) -> datetime:
    """Next regular-session close at/after `ts` (UTC)."""
    ts = _now_utc(ts)
    sched = _calendar().schedule(start_date=ts.date(), end_date=(ts + _HORIZON).date())
    for close_ in sched["market_close"]:
        c = close_.to_pydatetime()
        if c >= ts:
            return c
    raise RuntimeError("no session close within lookahead horizon")


def previous_session(asof: date | None = None) -> date:
    """Most recent trading day strictly before `asof` (defaults to today)."""
    asof = asof or datetime.now(UTC).date()
    days = sessions(asof - _LOOKBACK, asof)
    prior = [d for d in days if d < asof]
    if not prior:
        raise RuntimeError("no session within lookback horizon")
    return prior[-1]


def last_completed_session(ts: datetime | None = None) -> date:
    """The latest session whose regular close has already passed at `ts` — i.e. the most
    recent date for which end-of-day bars should exist. Drives the freshness job."""
    ts = _now_utc(ts)
    sched = _calendar().schedule(start_date=(ts - _LOOKBACK).date(), end_date=ts.date())
    completed = [
        row_close.to_pydatetime().date()
        for row_close in sched["market_close"]
        if row_close.to_pydatetime() <= ts
    ]
    if not completed:
        raise RuntimeError("no completed session within lookback horizon")
    return completed[-1]
