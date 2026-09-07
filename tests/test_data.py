"""Data feeds, validation and parsing.

Network feeds are tested against recorded response shapes rather than live
calls, so the suite is deterministic and works offline.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quantsutra.data import CsvFeed, FeedCache, FeedError, get_feed
from quantsutra.data.base import normalise_frame, resample_ohlcv, validate_frame
from quantsutra.data.nse import NseFeed
from quantsutra.data.synthetic import (generate_intraday_series,
                                       generate_vix_series)
from quantsutra.data.yahoo import YahooFeed


def test_synthetic_series_is_internally_consistent(synthetic):
    assert (synthetic["high"] >= synthetic[["open", "close"]].max(axis=1)).all()
    assert (synthetic["low"] <= synthetic[["open", "close"]].min(axis=1)).all()
    assert (synthetic[["open", "high", "low", "close"]] > 0).all().all()
    assert validate_frame(synthetic, "SYN") == []


def test_synthetic_volatility_is_plausible(synthetic):
    annual = synthetic["close"].pct_change().std() * np.sqrt(252) * 100
    assert 8 < annual < 40, "generated series should look like an equity index"


@pytest.mark.parametrize("seed", [7, 11, 42, 2024, 99])
def test_synthetic_tails_match_a_real_index(seed):
    """Lock the tail calibration.

    Long-option payoffs are convex, so an overstated tail inflates every
    options-mode backtest run against this data. An earlier version compounded
    three multiplicative tail sources and produced kurtosis near 18 with 9-12%
    single days -- which made the engine look far better than it is.
    """
    from quantsutra.data import generate_index_series

    returns = generate_index_series(days=1200, seed=seed)["close"].pct_change().dropna()
    annual_vol = returns.std() * np.sqrt(252) * 100
    assert 11 < annual_vol < 19, f"annualised vol {annual_vol:.1f}% is not index-like"

    over_2 = (returns.abs() > 0.02).mean()
    over_3 = (returns.abs() > 0.03).mean()
    over_5 = (returns.abs() > 0.05).mean()
    assert over_2 < 0.08, f"{over_2:.1%} of days move >2%; real indices are nearer 3-5%"
    assert over_3 < 0.025, f"{over_3:.1%} of days move >3%; real indices are nearer 1%"
    assert over_5 < 0.006, f"{over_5:.2%} of days move >5%; that is crisis frequency"
    assert returns.min() > -0.09, f"worst day {returns.min():.1%} is a crisis print"
    assert returns.kurtosis() < 8, "tails are too fat to be a fair options-backtest proxy"


def test_tail_bounds_are_tunable_and_actually_bind():
    """Raising the caps must widen the tails -- otherwise they are decoration."""
    from quantsutra.data import generate_index_series

    tight = generate_index_series(days=800, seed=3, max_vol_multiplier=1.5,
                                  max_shock_sigma=2.0)["close"].pct_change().dropna()
    loose = generate_index_series(days=800, seed=3, max_vol_multiplier=4.0,
                                  max_shock_sigma=6.0)["close"].pct_change().dropna()
    assert loose.abs().max() > tight.abs().max()


def test_synthetic_vix_is_negatively_correlated_with_returns(synthetic):
    vix = generate_vix_series(synthetic)
    correlation = np.corrcoef(vix.pct_change().fillna(0),
                              synthetic["close"].pct_change().fillna(0))[0, 1]
    assert correlation < -0.3, "VIX must rise when the index falls"


def test_intraday_expansion_respects_the_daily_range(synthetic):
    """Each expanded day must reproduce the daily bar it came from -- both ends.

    Checking only the high once let a clamp regression through: the low bound
    had been dropped and intraday bars were printing below the day's low.
    """
    daily = synthetic.tail(3)
    intraday = generate_intraday_series(daily, minutes=5)
    assert len(intraday) == 3 * 75

    for offset in range(3):
        day = daily.index[offset].date()
        bars = intraday[intraday.index.date == day]
        assert len(bars) == 75
        assert bars["high"].max() == pytest.approx(float(daily["high"].iloc[offset]), rel=1e-6)
        assert bars["low"].min() == pytest.approx(float(daily["low"].iloc[offset]), rel=1e-6)
        assert bars["high"].max() <= float(daily["high"].iloc[offset]) + 1e-6
        assert bars["low"].min() >= float(daily["low"].iloc[offset]) - 1e-6


def test_validator_catches_corrupt_bars(synthetic):
    bad = synthetic.copy()
    bad.iloc[5, bad.columns.get_loc("high")] = bad.iloc[5]["low"] - 100
    bad.iloc[9, bad.columns.get_loc("close")] = bad.iloc[9]["close"] * 1.5
    issues = validate_frame(bad, "BAD")
    assert any("high < low" in issue for issue in issues)
    assert any("outside the high-low range" in issue for issue in issues)


def test_validator_reports_missing_volume(synthetic):
    issues = validate_frame(synthetic.drop(columns=["volume"]).assign(volume=np.nan), "X")
    assert any("volume" in issue for issue in issues)


def test_csv_round_trip_preserves_the_timestamp_index(tmp_path, synthetic):
    """A CSV round trip once reinterpreted a RangeIndex as epoch nanoseconds
    and put every bar in 1970."""
    feed = CsvFeed(tmp_path)
    feed.save(synthetic, "NIFTY", "1d")
    restored = feed.history("NIFTY", "1d", lookback=len(synthetic))
    assert isinstance(restored.index, pd.DatetimeIndex)
    assert restored.index[-1] == synthetic.index[-1]
    assert restored.index[0].year > 2000


def test_csv_feed_accepts_broker_style_column_names(tmp_path, synthetic):
    export = synthetic.reset_index().rename(columns={
        "index": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume"})
    export.to_csv(tmp_path / "BANKNIFTY_1d.csv", index=False)
    frame = CsvFeed(tmp_path).history("BANKNIFTY", "1d")
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(frame.index, pd.DatetimeIndex)


def test_csv_feed_error_lists_what_it_looked_for(tmp_path):
    with pytest.raises(FeedError, match="No file for"):
        CsvFeed(tmp_path).history("NOTHERE", "1d")


def test_normalise_frame_rejects_data_with_no_timestamps():
    with pytest.raises(FeedError, match="no usable timestamp"):
        normalise_frame(pd.DataFrame({"open": [1, 2], "high": [2, 3],
                                      "low": [0, 1], "close": [1, 2]}))


def test_cache_round_trips_without_pyarrow(tmp_path, synthetic):
    cache = FeedCache(tmp_path / "cache", ttl_seconds=60)
    cache.put_frame(synthetic, "k", "NIFTY", "1d")
    restored = cache.get_frame("k", "NIFTY", "1d")
    assert restored is not None
    assert len(restored) == len(synthetic)


def test_cache_respects_ttl(tmp_path, synthetic):
    cache = FeedCache(tmp_path / "cache", ttl_seconds=0)
    cache.put_frame(synthetic, "k", "NIFTY", "1d")
    # ttl of 0 means every entry is immediately stale.
    assert FeedCache(tmp_path / "cache", ttl_seconds=-1).get_frame("k", "NIFTY", "1d") is None


def test_resample_aggregates_correctly(synthetic):
    weekly = resample_ohlcv(synthetic, "1W")
    assert len(weekly) < len(synthetic)
    assert weekly["high"].max() <= synthetic["high"].max() + 1e-6
    assert weekly["volume"].sum() == pytest.approx(synthetic["volume"].sum(), rel=1e-6)


# --- Yahoo parser, offline ------------------------------------------------

YAHOO_PAYLOAD = {
    "chart": {"error": None, "result": [{
        "meta": {"symbol": "^NSEI", "currency": "INR", "regularMarketPrice": 24850.5,
                 "chartPreviousClose": 24700.1},
        "timestamp": [1725417000, 1725503400, 1725589800],
        "indicators": {"quote": [{"open": [24700, 24760, 24810],
                                  "high": [24780, 24830, 24880],
                                  "low": [24660, 24720, 24770],
                                  "close": [24760, 24810, 24790],
                                  "volume": [240000, 255000, 231000]}]}}]}}


def test_yahoo_parser_produces_an_ist_frame():
    frame = YahooFeed._parse_chart(YAHOO_PAYLOAD, "^NSEI", 500)
    assert len(frame) == 3
    assert str(frame.index.tz) == "Asia/Kolkata"
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]


def test_yahoo_symbol_resolution():
    assert YahooFeed.resolve("NIFTY") == "^NSEI"
    assert YahooFeed.resolve("SENSEX") == "^BSESN"
    assert YahooFeed.resolve("RELIANCE.NS") == "RELIANCE.NS"


@pytest.mark.parametrize("payload,message", [
    ({"chart": {"error": {"code": "Not Found"}, "result": None}}, "error"),
    ({"chart": {"result": []}}, "no data"),
    ({"chart": {"result": [{"timestamp": [], "indicators": {"quote": [{}]}}]}}, "empty series"),
])
def test_yahoo_parser_fails_loudly(payload, message):
    with pytest.raises(FeedError, match=message):
        YahooFeed._parse_chart(payload, "^BAD", 10)


# --- NSE parser, offline --------------------------------------------------

def _nse_payload():
    first, second = dt.date(2025, 9, 9), dt.date(2025, 9, 16)
    rows = []
    for strike in range(24500, 25550, 50):
        for expiry in (first, second):
            rows.append({
                "strikePrice": strike, "expiryDate": expiry.strftime("%d-%b-%Y"),
                "CE": {"lastPrice": max(1, 25000 - strike + 120),
                       "openInterest": 300000 - abs(strike - 25000) * 80,
                       "changeinOpenInterest": 5000, "totalTradedVolume": 90000,
                       "impliedVolatility": 0 if strike == 25200 else 13.5},
                "PE": {"lastPrice": max(1, strike - 25000 + 120),
                       "openInterest": 350000 - abs(strike - 25000) * 80,
                       "changeinOpenInterest": 9000, "totalTradedVolume": 120000,
                       "impliedVolatility": 14.2},
            })
    return {"records": {"data": rows, "underlyingValue": 25012.4,
                        "expiryDates": [first.strftime("%d-%b-%Y"),
                                        second.strftime("%d-%b-%Y")],
                        "timestamp": "09-Sep-2025 15:29:57"}}


def test_nse_parser_selects_the_nearest_expiry_by_default():
    chain = NseFeed.parse_option_chain(_nse_payload(), "NIFTY")
    assert chain.expiry == dt.date(2025, 9, 9)
    assert chain.spot == pytest.approx(25012.4)
    assert chain.lot_size == 75


def test_nse_parser_treats_zero_iv_as_missing():
    """NSE reports IV as 0 for illiquid strikes; trusting that would produce
    a nonsense price."""
    chain = NseFeed.parse_option_chain(_nse_payload(), "NIFTY")
    assert np.isnan(chain.data.at[25200.0, "ce_iv"])


def test_nse_parser_can_select_a_later_expiry():
    chain = NseFeed.parse_option_chain(_nse_payload(), "NIFTY", expiry=dt.date(2025, 9, 16))
    assert chain.expiry == dt.date(2025, 9, 16)


def test_nse_parser_lists_available_expiries_when_asked_for_a_bad_one():
    with pytest.raises(FeedError, match="Available expiries"):
        NseFeed.parse_option_chain(_nse_payload(), "NIFTY", expiry=dt.date(2030, 1, 1))


def test_nse_parser_warns_that_the_snapshot_is_delayed():
    chain = NseFeed.parse_option_chain(_nse_payload(), "NIFTY")
    assert any("delayed" in warning for warning in chain.warnings)


def test_feed_factory():
    assert get_feed("yahoo").name == "yahoo"
    assert get_feed("csv", directory=".").name == "csv"
    with pytest.raises(ValueError, match="unknown feed"):
        get_feed("nope")
