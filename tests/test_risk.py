"""Transaction costs, position sizing and portfolio risk limits."""

import datetime as dt

import pytest

from quantsutra.risk.costs import CostModel
from quantsutra.risk.manager import RiskLimits, RiskManager
from quantsutra.risk.sizing import (fixed_fractional, kelly_fraction,
                                    size_option_position, volatility_target_size)


def test_option_costs_charge_stt_on_the_sell_side_only():
    model = CostModel("NSE")
    buy = model.option_leg(200, 75, "BUY")
    sell = model.option_leg(200, 75, "SELL")
    assert buy.stt == 0
    assert sell.stt > 0
    assert buy.stamp_duty > 0        # stamp duty is the buy-side charge
    assert sell.stamp_duty == 0


def test_gst_applies_to_fees_not_to_stt():
    model = CostModel("NSE")
    leg = model.option_leg(200, 75, "SELL")
    expected = (leg.brokerage + leg.exchange_charge + leg.sebi_fee) * 0.18
    assert leg.gst == pytest.approx(expected)


def test_letting_an_itm_option_expire_costs_more_than_squaring_off():
    model = CostModel("NSE")
    squared = model.option_leg(150, 75, "SELL", "ITM")
    expired = model.option_leg(150, 75, "SELL", "ITM", exercised_intrinsic=150)
    assert expired.stt > squared.stt
    assert expired.total > squared.total


def test_cheaper_options_need_a_larger_percentage_move_to_break_even():
    model = CostModel("NSE")
    expensive = model.breakeven_move(300, 75, "ATM")
    cheap = model.breakeven_move(20, 75, "FAR_OTM")
    assert cheap["premium_move_pct"] > expensive["premium_move_pct"]


def test_expiry_day_widens_the_assumed_spread():
    model = CostModel("NSE")
    normal = model.breakeven_move(50, 75, "OTM", is_expiry_day=False)
    expiry = model.breakeven_move(50, 75, "OTM", is_expiry_day=True)
    assert expiry["total_cost"] > normal["total_cost"]


def test_costs_can_be_disabled_for_a_gross_run():
    assert CostModel("NSE", include_slippage=False).option_leg(200, 75, "BUY").slippage == 0


def test_sizing_returns_zero_when_one_lot_breaches_the_risk_limit():
    result = size_option_position(50_000, 0.01, 200, 75, delta=0.5, index_stop_distance=150)
    assert result.lots == 0
    assert any("ZERO" in reason for reason in result.reasons)
    assert result.warnings, "a zero-size result must explain what capital would be needed"


def test_sizing_scales_with_capital():
    small = size_option_position(500_000, 0.02, 100, 75, delta=0.5, index_stop_distance=100)
    large = size_option_position(5_000_000, 0.02, 100, 75, delta=0.5, index_stop_distance=100)
    assert large.lots > small.lots


def test_sizing_never_exceeds_the_premium_outlay_cap():
    result = size_option_position(1_000_000, 0.50, 40, 75, delta=0.2,
                                  index_stop_distance=100, max_premium_pct=0.10)
    assert result.notional <= 1_000_000 * 0.10 + 1e-6


def test_sizing_uses_the_more_conservative_of_two_stop_bases():
    """The full premium is the risk when the index stop would take it to zero."""
    result = size_option_position(2_000_000, 0.02, 100, 75, delta=0.5,
                                  index_stop_distance=1000)
    assert result.risk_amount == pytest.approx(result.notional, rel=1e-6)


def test_fixed_fractional_rounds_down_to_whole_lots():
    assert fixed_fractional(100_000, 0.01, 10, 75) == 1
    assert fixed_fractional(10_000, 0.01, 10, 75) == 0


def test_kelly_is_capped_and_rejects_a_negative_edge():
    good = kelly_fraction(0.55, 1.8, 1.0)
    assert 0 < good["capped"] <= 0.25
    assert good["capped"] < good["kelly"], "must be fractional Kelly, not full"
    assert kelly_fraction(0.40, 1.0, 1.0)["capped"] == 0


def test_volatility_targeting_shrinks_as_volatility_rises():
    quiet = volatility_target_size(1_000_000, 0.15, 0.09, 25000, 75, leverage=8)
    wild = volatility_target_size(1_000_000, 0.15, 0.30, 25000, 75, leverage=8)
    assert quiet > wild


def _manager(**kwargs):
    manager = RiskManager(capital=500_000, limits=RiskLimits(**kwargs))
    manager.roll_day(dt.date(2025, 9, 8))
    return manager


def test_risk_manager_allows_a_normal_trade():
    assert _manager().check("UP", 4000, 15000, minutes_into_session=60).allowed


def test_risk_manager_blocks_an_oversized_trade():
    check = _manager().check("UP", 9000, 15000, minutes_into_session=60)
    assert not check.allowed
    assert "max_risk_amount" in check.adjustments


def test_daily_loss_limit_halts_trading():
    manager = _manager()
    manager.record_trade(-16_000)
    assert not manager.check("UP", 1000, 0, minutes_into_session=60).allowed
    assert manager.halted_reason


def test_consecutive_loss_breaker_resets_the_next_day():
    """Without a daily reset this rule deadlocks: it blocks every trade and
    only clears on a win that can then never happen."""
    manager = _manager()
    for _ in range(3):
        manager.record_trade(-1000)
    assert not manager.check("UP", 1000, 0, minutes_into_session=60).allowed
    manager.roll_day(dt.date(2025, 9, 9))
    assert manager.consecutive_losses == 0
    assert manager.check("UP", 1000, 0, minutes_into_session=60).allowed


def test_opening_and_closing_windows_are_blocked():
    manager = _manager()
    assert not manager.check("UP", 1000, 0, minutes_into_session=5).allowed
    assert not manager.check("UP", 1000, 0, minutes_into_session=370).allowed
    assert manager.check("UP", 1000, 0, minutes_into_session=120).allowed


def test_expiry_day_premium_selling_is_blocked():
    manager = _manager()
    check = manager.check("UP", 1000, 0, minutes_into_session=60,
                          is_expiry_day=True, is_short_premium=True)
    assert not check.allowed
    assert "gamma" in check.reasons[0].lower()


def test_correlated_positions_are_capped():
    manager = _manager(max_positions_same_direction=1)
    manager.open_positions.append({"direction": "UP"})
    assert not manager.check("UP", 1000, 0, minutes_into_session=60).allowed


def test_profit_target_stops_further_trading():
    manager = _manager()
    manager.record_trade(31_000)
    assert not manager.check("UP", 1000, 0, minutes_into_session=60).allowed
