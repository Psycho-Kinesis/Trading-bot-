r"""Synthetic market data generator.

Exists so the package can be demonstrated, tested and profiled without a
network connection or a paid data subscription.  The generator is a
regime-switching model with fat tails, volatility clustering and overnight
gaps, which is closer to an index than plain geometric Brownian motion.

**The tails are calibrated deliberately.**  An earlier version compounded three
multiplicative sources of tail risk (a fat-tailed volatility shock, a
fat-tailed return draw and a regime multiplier) and produced kurtosis near 18
with 9-12% single days.  That is not a harmless inaccuracy for this package:
long-option payoffs are convex, so an overstated tail inflates every
options-mode backtest run against it.  The volatility process is now bounded
and the standardised shock clipped, giving roughly:

===========================  ==================  =========================
Statistic                    This generator      NIFTY 50, long run
===========================  ==================  =========================
Annualised volatility        14-16%              13-18% (calm), 20-30%+ (stressed)
Daily \|move\| > 2%            3-5% of days       ~3-5% of days
Daily \|move\| > 3%            ~1% of days        ~1% of days
Daily \|move\| > 5%            <0.3% of days      rare outside crises
Worst single day             around -5%          -13% (Mar 2020) in a crisis
===========================  ==================  =========================

It therefore does **not** reproduce crisis behaviour -- there is no March-2020
in it. Stress-test against real history, not against this.

Nothing produced here is real market data, and a strategy that works on it has
demonstrated nothing except that it runs.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..calendar_in import today_ist, trading_days
from ..constants import IST

__all__ = ["generate_index_series", "generate_intraday_series", "generate_vix_series"]


def generate_index_series(
    start_price: float = 22000.0, days: int = 750, seed: int = 7,
    start_date: dt.date | None = None, annual_drift: float = 0.11,
    base_vol: float = 0.12, regime_persistence: float = 0.985,
    stress_vol_multiplier: float = 1.8, max_vol_multiplier: float = 2.8,
    max_shock_sigma: float = 3.8,
) -> pd.DataFrame:
    """Daily OHLCV with volatility clustering, regime shifts and gaps.

    ``max_vol_multiplier`` and ``max_shock_sigma`` bound the tails. Raising
    them makes options-mode backtests look better without making them more
    realistic -- see the module docstring.
    """
    rng = np.random.default_rng(seed)
    today = today_ist()
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

    vol_mult = np.where(state == 1, stress_vol_multiplier, 1.0)
    drift = np.where(state == 1, -annual_drift * 1.4, annual_drift)

    # GARCH-like persistence on top of the regime.
    daily_base = base_vol / np.sqrt(252)
    daily_vol = np.zeros(n)
    daily_vol[0] = daily_base
    for i in range(1, n):
        shock = abs(rng.standard_t(df=5)) * 0.0012
        daily_vol[i] = 0.90 * daily_vol[i - 1] + 0.10 * daily_base + 0.05 * shock
    # Index volatility mean-reverts hard; it does not compound without limit.
    daily_vol = np.clip(daily_vol * vol_mult, daily_base * 0.35,
                        daily_base * max_vol_multiplier)

    # Student-t rescaled to unit variance, then clipped: without the clip the
    # fat vol process and the fat return draw multiply into moves no index
    # delivers outside a crisis.
    df_returns = 6
    shocks = rng.standard_t(df=df_returns, size=n) / np.sqrt(df_returns / (df_returns - 2))
    shocks = np.clip(shocks, -max_shock_sigma, max_shock_sigma)
    returns = drift / 252 + daily_vol * shocks
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
        day_open = float(bar["open"])
        day_high = float(bar["high"])
        day_low = float(bar["low"])
        day_close = float(bar["close"])
        day_range = day_high - day_low
        day_volume = float(bar.get("volume", 0) or 0)

        steps = np.cumsum(rng.standard_normal(per_day))
        steps -= np.linspace(0, steps[-1], per_day)          # bridge: ends at zero
        path = np.linspace(day_open, day_close, per_day) + steps * day_range * 0.18

        span = path.max() - path.min()
        if span > 0:
            path = day_low + (path - path.min()) * day_range / span
        # The endpoints are fixed to the day's open and close, which can destroy
        # the extremes the stretch just created. Clip back into range, then plant
        # the day's high and low on interior bars so the expanded session really
        # does reproduce the daily bar it came from.
        path[0], path[-1] = day_open, day_close
        path = np.clip(path, day_low, day_high)
        if per_day >= 4 and day_range > 0:
            high_at = 1 + int(np.argmax(path[1:-1]))
            low_at = 1 + int(np.argmin(path[1:-1]))
            if high_at == low_at:
                low_at = 1 if high_at != 1 else per_day - 2
            path[high_at] = day_high
            path[low_at] = day_low

        # U-shaped volume profile: heavy at the open and into the close.
        shape = np.linspace(-1, 1, per_day)
        weights = 0.55 + 0.9 * shape**2
        weights /= weights.sum()

        session_start = ts.replace(hour=9, minute=15, second=0, microsecond=0)
        for i in range(per_day):
            prev = path[i - 1] if i else day_open
            wick = day_range * 0.03
            bar_high = max(prev, path[i]) + abs(rng.normal(0, wick))
            bar_low = min(prev, path[i]) - abs(rng.normal(0, wick))
            # Clamp each bar inside the day's true range so the expanded series
            # still reproduces the daily OHLC it came from.
            rows.append((prev, min(bar_high, day_high), max(bar_low, day_low),
                         path[i], day_volume * weights[i]))
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
