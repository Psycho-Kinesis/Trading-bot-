"""Core enums, market constants and shared vocabulary for the Indian market engine.

Anything in here that the exchanges can change (lot sizes, expiry weekdays, tax
rates) lives in ``config/instruments.yaml`` instead, keyed by an effective date,
so that backtests over historical windows use the rules that actually applied
then.  This module holds only the things that do not drift.
"""

from __future__ import annotations

from enum import Enum
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Regular equity/derivatives session on NSE and BSE.
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)
PRE_OPEN_START = (9, 0)
PRE_OPEN_END = (9, 8)
# Index option settlement is on the close of the expiry day.
TICK_SIZE = 0.05


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Direction(str, Enum):
    """What the engine thinks price does next."""

    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class Action(str, Enum):
    """The trade the engine recommends in option terms."""

    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    SELL_CALL = "SELL_CALL"
    SELL_PUT = "SELL_PUT"
    BULL_CALL_SPREAD = "BULL_CALL_SPREAD"
    BEAR_PUT_SPREAD = "BEAR_PUT_SPREAD"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"
    SHORT_STRADDLE = "SHORT_STRADDLE"
    LONG_STRADDLE = "LONG_STRADDLE"
    NO_TRADE = "NO_TRADE"


class OptionType(str, Enum):
    CALL = "CE"
    PUT = "PE"


class Regime(str, Enum):
    """Coarse market state.  Every playbook declares which regimes it is valid in."""

    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    RANGE = "RANGE"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"
    VOLATILE_CHOP = "VOLATILE_CHOP"
    SQUEEZE = "SQUEEZE"


class Timeframe(str, Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"
    W1 = "1wk"


TF_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "1d": 375, "1wk": 1875,
}


class Bias(str, Enum):
    """Strength buckets used when a rule reports its opinion."""

    STRONG_BULL = "STRONG_BULL"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    STRONG_BEAR = "STRONG_BEAR"


BIAS_SCORE: dict[Bias, float] = {
    Bias.STRONG_BULL: 1.0,
    Bias.BULL: 0.5,
    Bias.NEUTRAL: 0.0,
    Bias.BEAR: -0.5,
    Bias.STRONG_BEAR: -1.0,
}


def score_to_bias(score: float) -> Bias:
    """Inverse of :data:`BIAS_SCORE`, bucketing a continuous score in [-1, 1]."""
    if score >= 0.75:
        return Bias.STRONG_BULL
    if score >= 0.20:
        return Bias.BULL
    if score <= -0.75:
        return Bias.STRONG_BEAR
    if score <= -0.20:
        return Bias.BEAR
    return Bias.NEUTRAL


# Canonical OHLCV column names used everywhere in the package.
OHLCV = ["open", "high", "low", "close", "volume"]

# Index symbols this package knows about, mapped to their Yahoo Finance tickers.
INDEX_YAHOO = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "^NSEMDCP50",
    "NIFTYNXT50": "^NSMIDCP",
    "INDIAVIX": "^INDIAVIX",
}
