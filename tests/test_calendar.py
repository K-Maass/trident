"""XNYS calendar wrapper tests — offline, against known 2026 NYSE dates.
2026-07-03 (Fri) is the observed Independence Day holiday (July 4 falls on Saturday)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from trident.data import calendar as cal

UTC = UTC


def test_sessions_excludes_weekend_and_holiday():
    days = cal.sessions(date(2026, 7, 2), date(2026, 7, 6))
    assert days == [date(2026, 7, 2), date(2026, 7, 6)]  # Jul 3 holiday, Jul 4/5 weekend


def test_is_session():
    assert cal.is_session(date(2026, 7, 6)) is True  # Monday
    assert cal.is_session(date(2026, 7, 3)) is False  # observed holiday
    assert cal.is_session(date(2026, 7, 4)) is False  # Saturday


def test_is_open_regular_hours_utc():
    assert cal.is_open(datetime(2026, 7, 6, 14, 0, tzinfo=UTC)) is True  # 10:00 ET
    assert cal.is_open(datetime(2026, 7, 6, 2, 0, tzinfo=UTC)) is False  # overnight
    assert cal.is_open(datetime(2026, 7, 3, 15, 0, tzinfo=UTC)) is False  # holiday, midday


def test_next_open_and_close_cross_holiday_weekend():
    # From Saturday, the next open is Monday 09:30 ET = 13:30 UTC
    assert cal.next_open(datetime(2026, 7, 4, 12, 0, tzinfo=UTC)) == datetime(
        2026, 7, 6, 13, 30, tzinfo=UTC
    )
    # 16:00 ET close = 20:00 UTC
    assert cal.next_close(datetime(2026, 7, 6, 18, 0, tzinfo=UTC)) == datetime(
        2026, 7, 6, 20, 0, tzinfo=UTC
    )


def test_previous_session_skips_holiday_cluster():
    # The trading day before Mon Jul 6 is Thu Jul 2 (Jul 3 holiday, Jul 4/5 weekend)
    assert cal.previous_session(date(2026, 7, 6)) == date(2026, 7, 2)


def test_last_completed_session():
    # Mid-session Mon Jul 6 (14:00 UTC): last COMPLETED session is Thu Jul 2 (Jul 6 not closed yet)
    assert cal.last_completed_session(datetime(2026, 7, 6, 14, 0, tzinfo=UTC)) == date(2026, 7, 2)
    # After Jul 6 close (21:00 UTC): last completed is Jul 6 itself
    assert cal.last_completed_session(datetime(2026, 7, 6, 21, 0, tzinfo=UTC)) == date(2026, 7, 6)


def test_naive_datetime_treated_as_utc():
    assert cal.is_open(datetime(2026, 7, 6, 14, 0)) is True  # naive -> UTC
