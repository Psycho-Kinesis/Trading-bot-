"""Historical events, calendar risk and analogue search."""

import datetime as dt

import pytest

from quantsutra.knowledge import (
    MARKET_EVENTS,
    RECURRING_EVENTS,
    event_risk,
    events_between,
    find_analogues,
    lessons_for_regime,
    shape_distance,
)


def test_every_event_carries_a_lesson():
    assert len(MARKET_EVENTS) >= 10
    for event in MARKET_EVENTS:
        assert event.lesson and len(event.lesson) > 40
        assert event.description
        assert event.category


def test_events_are_chronological_and_in_the_past():
    dates = [e.date for e in MARKET_EVENTS]
    assert dates == sorted(dates)
    assert all(d < dt.date.today() for d in dates)


def test_events_between_filters_by_date():
    found = events_between(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    assert found
    assert all(2020 == e.date.year for e in found)


def test_recurring_events_declare_their_iv_impact():
    for item in RECURRING_EVENTS:
        assert item["iv_impact"] in ("LOW", "MEDIUM", "HIGH", "EXTREME")
        assert item["note"]


def test_expiry_day_is_flagged_high_risk():
    from quantsutra.calendar_in import next_expiry
    expiry = next_expiry("NIFTY", dt.date(2025, 9, 8))
    risk = event_risk(expiry, "NIFTY")
    assert risk["level"] == "HIGH"
    assert any("xpiry" in flag for flag in risk["flags"])


def test_budget_window_is_flagged():
    risk = event_risk(dt.date(2026, 2, 2), "NIFTY")
    assert risk["level"] in ("HIGH", "ELEVATED")
    assert any("Budget" in flag for flag in risk["flags"])


def test_non_trading_day_is_reported():
    assert event_risk(dt.date(2025, 9, 6))["level"] == "MARKET_CLOSED"


def test_normal_day_still_warns_about_unscheduled_risk():
    risk = event_risk(dt.date(2025, 11, 12), "NIFTY")
    assert any("never zero" in flag for flag in risk["flags"])


def test_lessons_are_regime_specific():
    assert lessons_for_regime("STRONG_UPTREND", "EXTREME")
    assert lessons_for_regime("RANGE", "LOW")
    assert lessons_for_regime("", "") == []


def test_shape_distance_is_scale_invariant():
    import numpy as np
    base = np.array([1.0, 2, 3, 2, 1])
    assert shape_distance(base, base * 1000 + 5000) == pytest.approx(0.0, abs=1e-9)


def test_analogues_report_dispersion_not_just_a_mean(synthetic):
    result = find_analogues(synthetic, window=40, forward=20, top_n=5)
    assert result["sample_size"] > 0
    assert "forward_std_pct" in result
    assert "forward_range_pct" in result
    assert "not a forecast" in result["note"]


def test_analogue_windows_do_not_overlap(synthetic):
    result = find_analogues(synthetic, window=40, forward=20, top_n=5)
    starts = [m["start"] for m in result["matches"]]
    assert len(set(starts)) == len(starts)


def test_analogues_refuse_insufficient_history(synthetic):
    result = find_analogues(synthetic.tail(50))
    assert result["sample_size"] == 0
    assert "Need at least" in result["note"]
