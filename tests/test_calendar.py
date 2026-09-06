"""Calendar and expiry resolution -- the most error-prone part of Indian F&O."""

import datetime as dt

import pytest

from quantsutra import calendar_in as cal


def test_weekends_are_not_trading_days():
    assert not cal.is_trading_day(dt.date(2025, 9, 6))   # Saturday
    assert not cal.is_trading_day(dt.date(2025, 9, 7))   # Sunday
    assert cal.is_trading_day(dt.date(2025, 9, 8))       # Monday


def test_known_holidays_are_excluded():
    assert not cal.is_trading_day(dt.date(2025, 8, 15))  # Independence Day
    assert not cal.is_trading_day(dt.date(2025, 12, 25))  # Christmas


def test_expiry_weekday_changed_with_the_2025_rationalisation():
    """NIFTY expired Thursday before Sept 2025 and Tuesday after."""
    before = cal.next_expiry("NIFTY", dt.date(2024, 6, 3))
    assert before.weekday() == 3, "pre-Sept-2025 NIFTY weeklies expired on Thursday"
    after = cal.next_expiry("NIFTY", dt.date(2025, 9, 8))
    assert after.weekday() == 1, "post-Sept-2025 NIFTY weeklies expire on Tuesday"


def test_expiry_rolls_backward_over_a_holiday():
    """Diwali 2025-10-21 is a Tuesday holiday; expiry must move to Monday."""
    expiry = cal.next_expiry("NIFTY", dt.date(2025, 10, 20))
    assert expiry == dt.date(2025, 10, 20)
    assert expiry.weekday() == 0
    assert cal.is_trading_day(expiry)


def test_banknifty_has_no_weeklies_after_nov_2024():
    assert cal.has_weekly("BANKNIFTY", dt.date(2024, 1, 1))
    assert not cal.has_weekly("BANKNIFTY", dt.date(2025, 9, 8))
    # With no weekly, the nearest expiry must be the monthly.
    nearest = cal.next_expiry("BANKNIFTY", dt.date(2025, 9, 8))
    monthly = cal.monthly_expiry_on_or_after("BANKNIFTY", dt.date(2025, 9, 8))
    assert nearest == monthly


def test_lot_size_is_effective_dated():
    assert cal.lot_size("NIFTY", dt.date(2024, 1, 5)) == 50
    assert cal.lot_size("NIFTY", dt.date(2025, 9, 8)) == 75


def test_sensex_expires_on_a_different_day_to_nifty():
    day = dt.date(2025, 9, 8)
    assert cal.next_expiry("SENSEX", day) != cal.next_expiry("NIFTY", day)


def test_expiry_chain_is_strictly_increasing():
    chain = cal.expiry_chain("NIFTY", dt.date(2025, 9, 8), count=6)
    assert len(chain) == 6
    assert chain == sorted(chain)
    assert len(set(chain)) == 6
    assert all(cal.is_trading_day(e) for e in chain)


def test_trading_days_between_excludes_holidays():
    # 2025-08-14 (Thu) to 2025-08-18 (Mon): the 15th is a holiday, 16-17 a weekend.
    assert cal.trading_days_between(dt.date(2025, 8, 14), dt.date(2025, 8, 18)) == 1


def test_days_to_expiry_uses_trading_days_by_default():
    day = dt.date(2025, 9, 8)
    assert cal.days_to_expiry("NIFTY", day) <= cal.days_to_expiry("NIFTY", day, calendar=True)


def test_session_phase_boundaries():
    def at(hour, minute):
        return dt.datetime(2025, 9, 8, hour, minute, tzinfo=cal.IST)

    assert cal.session_phase(at(9, 5)) == "PRE_OPEN"
    assert cal.session_phase(at(9, 20)) == "OPENING_AUCTION_DRIFT"
    assert cal.session_phase(at(10, 0)) == "MORNING_TREND"
    assert cal.session_phase(at(11, 30)) == "MIDDAY"
    assert cal.session_phase(at(14, 0)) == "AFTERNOON_TREND"
    assert cal.session_phase(at(15, 20)) == "CLOSING_HOUR"
    assert cal.session_phase(at(16, 30)) == "POST_CLOSE"
    assert cal.session_phase(dt.datetime(2025, 9, 6, 11, 0, tzinfo=cal.IST)) == "HOLIDAY"


def test_unknown_instrument_raises():
    with pytest.raises(KeyError):
        cal.spec("NOTANINDEX")


def test_expiry_context_flags_gamma_risk():
    expiry = cal.next_expiry("NIFTY", dt.date(2025, 9, 8))
    assert cal.expiry_context("NIFTY", expiry).gamma_risk == "EXTREME"
    far = cal.expiry_context("NIFTY", dt.date(2025, 9, 10))
    assert far.gamma_risk in ("NORMAL", "ELEVATED", "HIGH")
