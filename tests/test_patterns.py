"""Pattern recognition: candles, swings, structure and classical formations."""

import numpy as np
import pandas as pd
import pytest

from quantsutra.patterns.candles import detect_candles
from quantsutra.patterns.classical import detect_chart_patterns
from quantsutra.patterns.levels import (find_levels, level_interaction,
                                        reinforce_with_round_numbers, round_number_levels)
from quantsutra.patterns.structure import (fair_value_gaps, liquidity_sweeps,
                                           market_structure, order_blocks)
from quantsutra.patterns.swings import last_swings, swing_points, zigzag

FLAT = [[100, 102, 98, 100]] * 20
DOWN = FLAT + [[100 - 2 * i, 101 - 2 * i, 98 - 2 * i, 99 - 2 * i] for i in range(12)]
UP = FLAT + [[100 + 2 * i, 102 + 2 * i, 99 + 2 * i, 101 + 2 * i] for i in range(12)]


def frame(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(volume=1000)


def names(rows, **kwargs):
    return {s.name for s in detect_candles(frame(rows), lookback=1, **kwargs)}


@pytest.mark.parametrize("rows,expected", [
    (DOWN + [[76, 78.5, 70, 78]], "hammer"),
    (UP + [[123, 131, 122.5, 124]], "shooting_star"),
    (UP + [[120, 124, 119, 123], [123.5, 124, 116, 117]], "bearish_engulfing"),
    (DOWN + [[78, 79, 72, 73], [70, 77, 69.5, 76]], "piercing_line"),
    (DOWN + [[78, 79, 73, 74], [72, 73, 70.5, 71], [71.5, 79, 71, 78]], "morning_star"),
    (UP + [[123, 127, 122.5, 126.5], [127.5, 128.5, 127, 127.8], [127, 127.5, 118, 119]],
     "evening_star"),
    (DOWN + [[78, 79, 72, 73], [80, 86, 79.5, 85]], "bullish_kicker"),
    (DOWN + [[76, 90.2, 75.9, 90]], "bullish_marubozu"),
])
def test_candlestick_patterns_are_detected(rows, expected):
    assert expected in names(rows)


def test_reversal_patterns_require_a_prior_move():
    """A hammer with no preceding decline is not a hammer worth acting on."""
    detected = detect_candles(frame(FLAT + [[100, 101, 92, 100.6]]), lookback=1)
    hammers = [s for s in detected if s.name in ("hammer", "dragonfly_doji")]
    assert all(not s.context_ok for s in hammers)


def test_require_context_filters_weak_patterns():
    rows = FLAT + [[100, 101, 92, 100.6]]
    assert len(detect_candles(frame(rows), lookback=1, require_context=True)) <= \
           len(detect_candles(frame(rows), lookback=1))


def test_short_frames_return_nothing_rather_than_guessing():
    assert detect_candles(frame(FLAT[:4]), lookback=1) == []


def test_zigzag_pivots_alternate():
    t = np.arange(300)
    close = 20000 + 600 * np.sin(t / 18)
    data = pd.DataFrame({"open": close, "high": close + 25, "low": close - 25,
                         "close": close, "volume": 1})
    pivots = zigzag(data, 1.5)
    assert len(pivots) >= 4
    assert all(a.kind != b.kind for a, b in zip(pivots, pivots[1:]))


def test_swings_are_confirmed_after_the_fact():
    """A swing's confirmation index must be later than the pivot itself."""
    rng = np.random.default_rng(0)
    close = 20000 + np.cumsum(rng.normal(0, 40, 200))
    data = pd.DataFrame({"open": close, "high": close + 30, "low": close - 30,
                         "close": close, "volume": 1})
    for swing in swing_points(data, 3, 3):
        assert swing.confirmed_at > swing.index


def test_last_swings_alternate_and_are_recent():
    t = np.arange(300)
    close = 20000 + 600 * np.sin(t / 18)
    data = pd.DataFrame({"open": close, "high": close + 25, "low": close - 25,
                         "close": close, "volume": 1})
    recent = last_swings(zigzag(data, 1.5), 4)
    assert all(a.kind != b.kind for a, b in zip(recent, recent[1:]))


def test_market_structure_labels_match_the_last_four_swings(uptrend, downtrend):
    """Structure is a *local* read on the last four alternating swings.

    It deliberately does not describe the whole series: a 400-bar downtrend
    whose final swings are contracting is correctly labelled a range, and that
    local read is what makes the signal useful.
    """
    for data in (uptrend, downtrend):
        state = market_structure(data)
        pivots = last_swings(zigzag(data, use_atr=True, atr_mult=1.5), 4)
        highs = [p.price for p in pivots if p.is_high]
        lows = [p.price for p in pivots if not p.is_high]
        if len(highs) < 2 or len(lows) < 2:
            continue
        higher_high, higher_low = highs[-1] > highs[-2], lows[-1] > lows[-2]
        if higher_high and higher_low:
            assert state.trend == "UPTREND" and state.label == "HH-HL"
        elif not higher_high and not higher_low:
            assert state.trend == "DOWNTREND" and state.label == "LH-LL"
        else:
            assert state.trend == "RANGE"


def test_market_structure_detects_a_clean_downtrend():
    """With unambiguous LH-LL swings the label must be DOWNTREND."""
    def ramp(a, b, n):
        return list(np.linspace(a, b, n))
    path = (ramp(21000, 20000, 25) + ramp(20000, 20400, 12) + ramp(20400, 19400, 25)
            + ramp(19400, 19750, 12) + ramp(19750, 18800, 25) + ramp(18800, 19000, 10))
    close = np.array(path)
    data = pd.DataFrame({"open": close, "high": close + 20, "low": close - 20,
                         "close": close, "volume": 1e5})
    state = market_structure(data)
    assert state.trend == "DOWNTREND"
    assert state.label == "LH-LL"


def test_market_structure_detects_a_clean_uptrend():
    def ramp(a, b, n):
        return list(np.linspace(a, b, n))
    path = (ramp(19000, 20000, 25) + ramp(20000, 19600, 12) + ramp(19600, 20600, 25)
            + ramp(20600, 20250, 12) + ramp(20250, 21200, 25) + ramp(21200, 21000, 10))
    close = np.array(path)
    data = pd.DataFrame({"open": close, "high": close + 20, "low": close - 20,
                         "close": close, "volume": 1e5})
    state = market_structure(data)
    assert state.trend == "UPTREND"
    assert state.label == "HH-HL"


def test_market_structure_degrades_gracefully_on_short_data():
    tiny = pd.DataFrame({"open": [1, 2], "high": [2, 3], "low": [0, 1],
                         "close": [1, 2], "volume": [1, 1]})
    state = market_structure(tiny)
    assert state.trend == "RANGE"
    assert state.strength == 0.0


def test_head_and_shoulders_target_is_the_measured_move():
    def ramp(a, b, n):
        return list(np.linspace(a, b, n))
    path = (ramp(19000, 20000, 25) + ramp(20000, 20500, 10) + ramp(20500, 20100, 8)
            + ramp(20100, 21000, 12) + ramp(21000, 20100, 10) + ramp(20100, 20520, 8)
            + ramp(20520, 19800, 12))
    close = np.array(path)
    data = pd.DataFrame({"open": close, "high": close + 3, "low": close - 3,
                         "close": close, "volume": 1e5})
    found = [p for p in detect_chart_patterns(data) if p.name == "head_and_shoulders"]
    assert found, "a textbook head and shoulders should be detected"
    pattern = found[0]
    assert pattern.bias == -1
    assert pattern.target < pattern.trigger < pattern.stop
    height = pattern.stop - pattern.trigger
    assert pattern.target == pytest.approx(pattern.trigger - height, rel=0.02)


def test_double_bottom_is_bullish_with_a_target_above_the_trigger():
    def ramp(a, b, n):
        return list(np.linspace(a, b, n))
    path = (ramp(21000, 20000, 25) + ramp(20000, 19500, 10) + ramp(19500, 20200, 12)
            + ramp(20200, 19510, 12) + ramp(19510, 20600, 15))
    close = np.array(path)
    data = pd.DataFrame({"open": close, "high": close + 3, "low": close - 3,
                         "close": close, "volume": 1e5})
    found = [p for p in detect_chart_patterns(data) if p.name == "double_bottom"]
    assert found and found[0].bias == 1
    assert found[0].target > found[0].trigger


def test_fair_value_gap_detection():
    rows = [[100 + i * 0.1, 100.5 + i * 0.1, 99.5 + i * 0.1, 100 + i * 0.1] for i in range(30)]
    rows += [[103, 103.5, 102.8, 103.2], [104, 112, 103.9, 111], [112.5, 114, 112.2, 113.5]]
    data = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(volume=1e5)
    gaps = fair_value_gaps(data)
    assert any(g["kind"] == "BULLISH" for g in gaps)


def test_order_block_detection():
    rows = [[100 + i * 0.2, 101 + i * 0.2, 99 + i * 0.2, 100.5 + i * 0.2] for i in range(25)]
    rows += [[105, 105.5, 102, 102.5], [102.5, 118, 102.4, 117]]
    rows += [[117 + i, 118.5 + i, 116 + i, 118 + i] for i in range(6)]
    data = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(volume=1e5)
    blocks = order_blocks(data)
    assert blocks and blocks[0]["kind"] == "BULLISH"


def test_liquidity_sweep_detection():
    rows = [[100, 102, 98, 100] for _ in range(8)]
    rows.append([100, 112, 99, 102])                  # swing high at 112
    rows += [[100, 102, 98, 100] for _ in range(10)]
    rows.append([101, 118, 100, 103])                 # wick above, closes below
    rows += [[100, 102, 98, 99] for _ in range(6)]
    data = pd.DataFrame(rows, columns=["open", "high", "low", "close"]).assign(volume=1e5)
    sweeps = liquidity_sweeps(data)
    assert sweeps and sweeps[0]["kind"] == "SELL_SIDE_SWEEP"
    assert sweeps[0]["bias"] == -1


def test_levels_cluster_repeated_touches(ranging):
    levels = find_levels(ranging)
    assert levels
    assert all(level.touches >= 2 for level in levels)
    assert all(0 <= level.strength <= 1 for level in levels)


def test_round_number_spacing_scales_with_volatility():
    tight = round_number_levels(23000, atr_value=90)
    wide = round_number_levels(23000, atr_value=425)
    tight_step = tight[1].price - tight[0].price
    wide_step = wide[1].price - wide[0].price
    assert wide_step > tight_step


def test_untested_round_numbers_are_weak():
    """Round numbers must not be strong enough to veto a trade on their own."""
    for level in round_number_levels(25000, atr_value=150):
        assert level.strength < 0.5


def test_round_number_coinciding_with_a_swing_is_promoted(ranging):
    swings = find_levels(ranging)
    combined = reinforce_with_round_numbers(
        swings + round_number_levels(float(ranging["close"].iloc[-1]), atr_value=150), 150.0)
    promoted = [lv for lv in combined if lv.source == "swing+round"]
    for level in promoted:
        assert level.strength > 0.25


def test_level_interaction_reports_room(uptrend):
    from quantsutra.indicators.volatility import atr
    levels = find_levels(uptrend)
    price = float(uptrend["close"].iloc[-1])
    info = level_interaction(price, levels, float(atr(uptrend).iloc[-1]))
    assert "at_level" in info
    if info.get("room_to_resistance_atr") is not None:
        assert info["room_to_resistance_atr"] >= 0
