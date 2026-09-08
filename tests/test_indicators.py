"""Indicator correctness.

Checked against mathematical properties and known-answer cases rather than
snapshots, so the tests catch real regressions instead of formatting changes.
"""

import numpy as np
import pandas as pd
import pytest

from quantsutra import indicators as ind
from quantsutra.indicators._util import crossover, crossunder, wilder_smooth
from quantsutra.indicators.momentum import cci, rsi, stochastic
from quantsutra.indicators.trend import adx, ema, macd, sma, supertrend
from quantsutra.indicators.volatility import atr, bollinger, choppiness, historical_volatility
from quantsutra.indicators.volume import has_usable_volume, obv, volume_profile, vwap


def test_sma_known_answer():
    series = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert sma(series, 3).iloc[-1] == pytest.approx(4.0)


def test_ema_converges_to_a_constant_series():
    assert ema(pd.Series([100.0] * 100), 20).iloc[-1] == pytest.approx(100.0)


def test_rsi_bounds_and_extremes():
    rising = pd.Series(np.arange(100, 200, dtype=float))
    assert rsi(rising).iloc[-1] == pytest.approx(100.0)
    falling = pd.Series(np.arange(200, 100, -1, dtype=float))
    assert rsi(falling).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_stays_within_zero_and_hundred(uptrend):
    values = rsi(uptrend["close"]).dropna()
    assert values.between(0, 100).all()


def test_wilder_smoothing_differs_from_a_plain_ema():
    series = pd.Series(np.arange(1, 101, dtype=float))
    assert abs(wilder_smooth(series, 14).iloc[-1] - ema(series, 14).iloc[-1]) > 0.5


def test_atr_is_positive_and_tracks_range(uptrend):
    values = atr(uptrend).dropna()
    assert (values > 0).all()
    wide = uptrend.copy()
    wide["high"] = wide["high"] * 1.02
    wide["low"] = wide["low"] * 0.98
    assert atr(wide).iloc[-1] > values.iloc[-1]


def test_bollinger_bands_are_ordered_and_contain_price(ranging):
    bands = bollinger(ranging["close"])
    valid = bands["bb_upper"].notna()
    assert (bands.loc[valid, "bb_upper"] > bands.loc[valid, "bb_mid"]).all()
    assert (bands.loc[valid, "bb_mid"] > bands.loc[valid, "bb_lower"]).all()
    close = ranging["close"]
    inside = ((close >= bands["bb_lower"]) & (close <= bands["bb_upper"]))[valid]
    # Financial series are not iid normal, so the textbook 95% does not hold
    # exactly; the band should still contain the large majority of bars.
    assert 0.80 < inside.mean() < 1.0

    wider = bollinger(close, std=3.0)
    wide_valid = wider["bb_upper"].notna()
    wide_inside = ((close >= wider["bb_lower"]) & (close <= wider["bb_upper"]))[wide_valid]
    assert wide_inside.mean() > inside.mean(), "wider bands must contain more bars"


def test_adx_is_higher_in_a_trend_than_in_a_range(uptrend, ranging):
    assert adx(uptrend)["adx"].iloc[-1] > adx(ranging)["adx"].iloc[-1]


def test_choppiness_is_higher_in_a_range_than_in_a_trend(uptrend, ranging):
    assert choppiness(ranging).iloc[-1] > choppiness(uptrend).iloc[-1]


def test_supertrend_direction_matches_the_trend(uptrend, downtrend):
    assert supertrend(uptrend)["direction"].iloc[-1] == 1
    assert supertrend(downtrend)["direction"].iloc[-1] == -1


def test_supertrend_flips_are_direction_changes(uptrend):
    result = supertrend(uptrend)
    flips = result.index[result["flip"].fillna(False)]
    for stamp in flips:
        position = result.index.get_loc(stamp)
        assert result["direction"].iloc[position] != result["direction"].iloc[position - 1]


def test_macd_histogram_equals_line_minus_signal(uptrend):
    result = macd(uptrend["close"]).dropna()
    assert np.allclose(result["hist"], result["macd"] - result["signal"])


def test_stochastic_is_bounded(ranging):
    values = stochastic(ranging)["stoch_k"].dropna()
    assert values.between(0, 100).all()


def test_obv_moves_with_direction():
    frame = pd.DataFrame({
        "open": [100, 101, 102], "high": [101, 102, 103],
        "low": [99, 100, 101], "close": [100, 102, 101],
        "volume": [1000, 2000, 3000],
    })
    values = obv(frame)
    assert values.iloc[1] > values.iloc[0]      # up bar adds volume
    assert values.iloc[2] < values.iloc[1]      # down bar subtracts it


def test_vwap_lies_within_the_session_range(uptrend):
    values = vwap(uptrend).dropna()
    assert (values >= uptrend["low"].min()).all()
    assert (values <= uptrend["high"].max()).all()


def test_volume_detection_handles_missing_volume(uptrend):
    assert has_usable_volume(uptrend)
    assert not has_usable_volume(uptrend.assign(volume=0))


def test_volume_profile_poc_sits_inside_the_range(uptrend):
    profile = volume_profile(uptrend, bins=30)
    assert uptrend["low"].min() <= profile["poc"] <= uptrend["high"].max()
    assert profile["val"] <= profile["poc"] <= profile["vah"]


def test_historical_volatility_is_annualised():
    rng = np.random.default_rng(0)
    daily_sigma = 0.01
    series = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, daily_sigma, 2000))))
    result = historical_volatility(series, 250).iloc[-1]
    expected = daily_sigma * np.sqrt(252) * 100
    assert expected * 0.6 < result < expected * 1.4


def test_crossover_helpers():
    a = pd.Series([1, 2, 3, 2, 1], dtype=float)
    b = pd.Series([2, 2, 2, 2, 2], dtype=float)
    # A cross fires only on a strict inequality: at index 3 a equals b, so the
    # cross is not confirmed until index 4.
    assert crossover(a, b).tolist() == [False, False, True, False, False]
    assert crossunder(a, b).tolist() == [False, False, False, False, True]


def test_compute_all_produces_a_wide_frame_without_lookahead(uptrend):
    features = ind.compute_all(uptrend)
    assert len(features.columns) > 100
    assert len(features) == len(uptrend)
    # Truncating the input must not change any earlier value: every indicator
    # here is causal, and a forward-looking one would break this.
    truncated = ind.compute_all(uptrend.iloc[:-20])
    for column in ("rsi_14", "ema_20", "adx", "atr_14", "st10_direction"):
        left = features[column].iloc[:-20].dropna()
        right = truncated[column].dropna()
        common = left.index.intersection(right.index)
        assert len(common) > 50
        assert np.allclose(left.loc[common], right.loc[common], equal_nan=True), \
            f"{column} changed when future bars were removed -- look-ahead bias"


def test_compute_all_skips_volume_when_absent(uptrend):
    features = ind.compute_all(uptrend.drop(columns=["volume"]))
    assert features.attrs["volume_available"] is False
    assert "cmf_20" not in features.columns


def test_snapshot_is_json_safe(uptrend):
    import json
    snapshot = ind.snapshot(ind.compute_all(uptrend))
    json.dumps(snapshot)          # must not raise
    assert isinstance(snapshot["squeeze_on"], bool)


def test_cci_uses_mean_absolute_deviation():
    """A constant series has zero deviation, so CCI must be NaN, not infinite."""
    flat = pd.DataFrame({"open": [100.0] * 40, "high": [100.0] * 40,
                         "low": [100.0] * 40, "close": [100.0] * 40,
                         "volume": [1] * 40})
    assert np.isnan(cci(flat).iloc[-1])
