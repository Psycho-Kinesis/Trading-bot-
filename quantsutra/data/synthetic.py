"""Synthetic market data generator.

Exists so the package can be demonstrated, tested and profiled without a
network connection or a paid data subscription.  The generator is a
regime-switching model with fat tails, volatility clustering and overnight
gaps, which is closer to an index than plain geometric Brownian motion.

Nothing produced here is real market data, and a strategy that works on it has
demonstrated nothing except that it runs.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..calendar_in import trading_days
from ..constants import IST

__all__ = ["generate_index_series", "generate_intraday_series", "generate_vix_series"]


def generate_index_series(
    start_price: float = 22000.0, days: int = 750, seed: int = 7,
    start_date: dt.date | None = None, annual_drift: float = 0.11,
    base_vol: float = 0.13, regime_persistence: float = 0.985,
) -> pd.DataFrame:
    """Daily OHLCV with volatility clustering, regime shifts and gaps."""
    rng = np.random.default_rng(seed)
    today = dt.date.today()
    start_date = start_date or (today - dt.timedelta(days=int(days * 1.45)))
    horizon = min(start_date + dt.timedelta(days=int(days * 1.6)), today)
    dates = trading_days(start_date, horizon)[:days]
    n = len(dates)
    if n == 0:
        raise ValueError("no trading days generated; check the date range")

    # Two-state regime: calm and stressed, with stressed states also negative drift.
    state = np.zeros(n, dtype=int)
    for i in range(1, n):
        stay = regime_persistence if state[i - 1] == 0 else 0.94
        state[i] = state[i - 1] if rng.random() < stay else 1 - state[i - 1]

    vol_mult = np.where(state == 1, 2.3, 1.0)
    drift = np.where(state == 1, -annual_drift * 1.4, annual_drift)

    # GARCH-like persistence on top of the regime.
    daily_vol = np.zeros(n)
    daily_vol[0] = base_vol / np.sqrt(252)
    for i in range(1, n):
        shock = abs(rng.standard_t(df=4)) * 0.0012
        daily_vol[i] = 0.90 * daily_vol[i - 1] + 0.10 * (base_vol / np.sqrt(252)) + 0.05 * shock
    daily_vol *= vol_mult

    returns = drift / 252 + daily_vol * rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
    close = start_price * np.exp(np.cumsum(returns))

    # Overnight gap, then an intraday path around it.
    gap = daily_vol * rng.standard_normal(n) * 0.55
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + gap[1:])

    intraday_range = close * daily_vol * (1.4 + 0.8 * rng.random(n))
    upper = rng.random(n)
    high = np.maximum(open_, close) + intraday_range * upper * 0.6
    low = np.minimum(open_, close) - intraday_range * (1 - upper) * 0.6

    base_volume = 2.2e5
    volume = base_volume * (1 + 1.6 * (intraday_range / close) / daily_vol) * (0.7 + 0.6 * rng.random(n))

    index = pd.DatetimeIndex([dt.datetime(d.year, d.month, d.day, 15, 30, tzinfo=IST)
                              for d in dates])
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume.astype(int),
    }, index=index)


def generate_intraday_series(
    daily: pd.DataFrame, minutes: int = 5, seed: int = 7
) -> pd.DataFrame:
    """Expand daily bars into an intraday path that respects each day's OHLC.

    Uses a Brownian bridge from open to close, then stretches the path so the
    day's true high and low are actually touched -- so intraday backtests see
    the same daily geometry the daily-bar tests do.
    """
    rng = np.random.default_rng(seed)
    per_day = max(1, 375 // minutes)
    rows, stamps = [], []

    for ts, bar in daily.iterrows():
        o, h, lo, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        day_volume = float(bar.get("volume", 0) or 0)

        steps = np.cumsum(rng.standard_normal(per_day))
        steps -= np.linspace(0, steps[-1], per_day)          # bridge: ends at zero
        path = np.linspace(o, c, per_day) + steps * (h - lo) * 0.18

        span = path.max() - path.min()
        if span > 0:
            path = lo + (path - path.min()) * (h - lo) / span
        path[0], path[-1] = o, c

        # U-shaped volume profile: heavy at the open and into the close.
        shape = np.linspace(-1, 1, per_day)
        weights = 0.55 + 0.9 * shape**2
        weights /= weights.sum()

        session_start = ts.replace(hour=9, minute=15, second=0, microsecond=0)
        for i in range(per_day):
            prev = path[i - 1] if i else o
            hi = max(prev, path[i]) + abs(rng.normal(0, (h - lo) * 0.03))
            lo = min(prev, path[i]) - abs(rng.normal(0, (h - lo) * 0.03))
            rows.append((prev, min(hi, h), max(lo, lo), path[i], day_volume * weights[i]))
            stamps.append(session_start + dt.timedelta(minutes=minutes * (i + 1)))

    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                         index=pd.DatetimeIndex(stamps))
    frame["high"] = frame[["open", "high", "close"]].max(axis=1)
    frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    frame["volume"] = frame["volume"].astype(int)
    return frame


def generate_vix_series(daily: pd.DataFrame, seed: int = 7,
                        base: float = 14.0) -> pd.Series:
    """A plausible India VIX path: mean-reverting, spiking on down days.

    Real VIX is strongly negatively correlated with index returns and reverts
    fast, so that asymmetry is modelled explicitly.
    """
    rng = np.random.default_rng(seed)
    returns = daily["close"].pct_change().fillna(0).to_numpy()
    vix = np.zeros(len(daily))
    vix[0] = base
    for i in range(1, len(daily)):
        shock = -70 * returns[i] if returns[i] < 0 else -22 * returns[i]
        vix[i] = max(9.0, 0.90 * vix[i - 1] + 0.10 * base + shock + rng.normal(0, 0.35))
    return pd.Series(vix, index=daily.index, name="india_vix")
