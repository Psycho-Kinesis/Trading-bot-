"""Options pricing, greeks and chain analytics.

Pricing is checked against identities that must hold exactly (put-call parity,
the delta relation, gamma/vega equality) rather than against hardcoded numbers,
so the tests stay meaningful if the implementation changes.
"""

import math

import numpy as np
import pytest

from quantsutra.constants import Action
from quantsutra.options.chain import classify_oi_buildup, iv_rank, max_pain, put_call_ratio
from quantsutra.options.pricing import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RATE,
    bs_price,
    delta_to_strike,
    greeks,
    implied_volatility,
    moneyness,
    time_to_expiry,
)
from quantsutra.options.selection import select_strike
from quantsutra.options.strategies import build_strategy, choose_structure, strategy_metrics

S, K, T, V = 25000.0, 25000.0, time_to_expiry(5), 0.14


def test_put_call_parity_holds_exactly():
    call = bs_price(S, K, T, V, "CE")
    put = bs_price(S, K, T, V, "PE")
    expected = S * math.exp(-DEFAULT_DIVIDEND_YIELD * T) - K * math.exp(-DEFAULT_RATE * T)
    assert abs((call - put) - expected) < 1e-8


@pytest.mark.parametrize("strike", [23000, 24500, 25000, 25500, 27000])
def test_parity_across_strikes(strike):
    call = bs_price(S, strike, T, V, "CE")
    put = bs_price(S, strike, T, V, "PE")
    expected = S * math.exp(-DEFAULT_DIVIDEND_YIELD * T) - strike * math.exp(-DEFAULT_RATE * T)
    assert abs((call - put) - expected) < 1e-8


def test_delta_relation_and_shared_greeks():
    call = greeks(S, K, T, V, "CE")
    put = greeks(S, K, T, V, "PE")
    assert abs((call.delta - put.delta) - math.exp(-DEFAULT_DIVIDEND_YIELD * T)) < 1e-9
    assert abs(call.gamma - put.gamma) < 1e-12
    assert abs(call.vega - put.vega) < 1e-12


@pytest.mark.parametrize("vol", [0.08, 0.14, 0.25, 0.60])
@pytest.mark.parametrize("strike,kind", [(25000, "CE"), (25600, "CE"), (24400, "PE")])
def test_implied_volatility_round_trips(vol, strike, kind):
    price = bs_price(S, strike, T, vol, kind)
    assert abs(implied_volatility(price, S, strike, T, kind) - vol) < 1e-5


def test_iv_returns_nan_outside_no_arbitrage_bounds():
    assert np.isnan(implied_volatility(1.0, S, 24000, T, "CE"))   # below intrinsic
    assert np.isnan(implied_volatility(0.0, S, K, T, "CE"))       # zero price
    assert np.isnan(implied_volatility(S + 1, S, K, T, "CE"))     # above the underlying


def test_theta_and_gamma_increase_as_expiry_approaches():
    far = greeks(S, K, time_to_expiry(10), V, "CE")
    near = greeks(S, K, time_to_expiry(1), V, "CE")
    assert abs(near.theta) > abs(far.theta)
    assert near.gamma > far.gamma


def test_at_expiry_price_is_intrinsic():
    assert greeks(S, 24900, 0, V, "CE").price == pytest.approx(100.0)
    assert greeks(S, 25100, 0, V, "PE").price == pytest.approx(100.0)
    assert greeks(S, 25100, 0, V, "CE").price == 0.0


def test_long_option_theta_is_negative():
    assert greeks(S, K, T, V, "CE").theta < 0
    assert greeks(S, K, T, V, "PE").theta < 0


def test_delta_targeting_finds_a_lower_delta_further_otm():
    atm = delta_to_strike(0.50, S, T, V, "CE", 50)
    otm = delta_to_strike(0.20, S, T, V, "CE", 50)
    assert otm > atm
    assert abs(greeks(S, otm, T, V, "CE").delta - 0.20) < 0.06


def test_moneyness_labels():
    assert moneyness(25000, 25000, "CE") == "ATM"
    assert moneyness(25000, 25500, "CE") == "OTM"
    assert moneyness(25000, 25500, "PE") == "ITM"


def test_oi_buildup_matrix():
    assert classify_oi_buildup(1, 1) == "LONG_BUILDUP"
    assert classify_oi_buildup(-1, 1) == "SHORT_BUILDUP"
    assert classify_oi_buildup(1, -1) == "SHORT_COVERING"
    assert classify_oi_buildup(-1, -1) == "LONG_UNWINDING"
    assert classify_oi_buildup(0, 1) == "NEUTRAL"


def test_chain_summary_is_complete(option_chain):
    summary = option_chain.summary()
    for key in ("atm_strike", "pcr_oi", "max_pain", "iv_skew", "max_call_oi_strike"):
        assert summary.get(key) is not None, f"{key} missing from chain summary"
    assert 0 < summary["pcr_oi"] < 10


def test_chain_recovers_iv_from_prices(option_chain):
    option_chain.data["ce_iv"] = np.nan
    option_chain.compute_ivs()
    atm = option_chain.atm_strike
    assert 5 < float(option_chain.data.at[atm, "ce_iv"]) < 40


def test_max_pain_sits_inside_the_strike_range(option_chain):
    result = max_pain(option_chain)
    strikes = option_chain.data.index
    assert strikes.min() <= result["max_pain"] <= strikes.max()


def test_pcr_reading_is_directional(option_chain):
    result = put_call_ratio(option_chain)
    assert result["pcr_oi"] > 0
    assert isinstance(result["pcr_reading"], str)


def test_naked_short_reports_unlimited_risk():
    strategy = build_strategy(Action.SELL_CALL, S, {"short_ce": 25300},
                              {"short_ce": 90.0}, 75)
    metrics = strategy_metrics(strategy)
    assert metrics["max_loss"] == "UNLIMITED"
    assert metrics["risk_defined"] is False


def test_spread_reports_defined_risk():
    strategy = build_strategy(Action.BULL_CALL_SPREAD, S,
                              {"long_ce": 25000, "short_ce": 25300},
                              {"long_ce": 210.0, "short_ce": 90.0}, 75)
    metrics = strategy_metrics(strategy)
    assert metrics["risk_defined"] is True
    assert metrics["max_loss"] == pytest.approx(-(210.0 - 90.0) * 75, rel=1e-3)
    assert metrics["max_profit"] == pytest.approx((300 - 120) * 75, rel=1e-2)


def test_iron_condor_has_two_breakevens():
    strategy = build_strategy(
        Action.IRON_CONDOR, S,
        {"short_pe": 24700, "long_pe": 24500, "short_ce": 25300, "long_ce": 25500},
        {"short_pe": 74.0, "long_pe": 35.0, "short_ce": 90.0, "long_ce": 45.0}, 75)
    metrics = strategy_metrics(strategy)
    assert len(metrics["breakevens"]) == 2
    assert metrics["risk_defined"] is True


def test_structure_choice_respects_iv_and_time():
    # Rich IV -> a spread, not a naked long.
    action, _ = choose_structure(1, iv_percentile=85, dte_trading=5, trend_strength=0.8)
    assert action == Action.BULL_CALL_SPREAD
    # Cheap IV -> a naked long is fine.
    action, _ = choose_structure(1, iv_percentile=15, dte_trading=5, trend_strength=0.8)
    assert action == Action.BUY_CALL
    # No direction, rich IV, time left -> a range structure.
    action, _ = choose_structure(0, iv_percentile=85, dte_trading=5, trend_strength=0.3)
    assert action == Action.IRON_CONDOR
    # No direction, no volatility edge -> no trade.
    action, _ = choose_structure(0, iv_percentile=50, dte_trading=5, trend_strength=0.3)
    assert action == Action.NO_TRADE


def test_strike_selection_targets_delta_and_checks_liquidity(option_chain):
    choice = select_strike(option_chain, "CE", target_delta=0.30)
    assert choice is not None
    assert abs(abs(choice.delta) - 0.30) < 0.10
    assert choice.liquidity_ok
    assert any("delta" in reason for reason in choice.reasons)


def test_illiquid_chain_is_flagged(option_chain):
    option_chain.data["ce_oi"] = 100          # far below any sane floor
    choice = select_strike(option_chain, "CE", target_delta=0.40, min_oi=25_000)
    assert choice is not None
    assert not choice.liquidity_ok
    assert any("illiquid" in r.lower() or "OI floor" in r for r in choice.reasons)


def test_iv_rank_flags_expensive_and_cheap():
    history = np.concatenate([np.full(200, 12.0), np.full(100, 20.0)])
    assert iv_rank(25.0, history)["iv_reading"] == "IV_EXPENSIVE_FAVOUR_SELLING"
    assert iv_rank(10.0, history)["iv_reading"] == "IV_CHEAP_FAVOUR_BUYING"
    assert iv_rank(15.0, np.array([1, 2, 3]))["iv_reading"] == "INSUFFICIENT_HISTORY"
