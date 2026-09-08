"""Trend and moving-average family indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import ensure_ohlcv, series, true_range, wilder_smooth

__all__ = [
    "adx",
    "alma",
    "aroon",
    "dema",
    "dpo",
    "ema",
    "hma",
    "ichimoku",
    "kama",
    "ma_ribbon_score",
    "macd",
    "mass_index",
    "psar",
    "sma",
    "supertrend",
    "tema",
    "trix",
    "vortex",
    "vwma",
    "wma",
]


def sma(s: pd.Series, length: int = 20) -> pd.Series:
    return series(s).rolling(length).mean()


def ema(s: pd.Series, length: int = 20) -> pd.Series:
    return series(s).ewm(span=length, adjust=False, min_periods=length).mean()


def wma(s: pd.Series, length: int = 20) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return series(s).rolling(length).apply(
        lambda w: float(np.dot(w, weights) / weights.sum()), raw=True
    )


def hma(s: pd.Series, length: int = 21) -> pd.Series:
    """Hull MA -- much less lag than an EMA of the same length."""
    half, root = max(1, length // 2), max(1, int(np.sqrt(length)))
    return wma(2 * wma(s, half) - wma(s, length), root)


def dema(s: pd.Series, length: int = 20) -> pd.Series:
    e1 = ema(s, length)
    return 2 * e1 - ema(e1, length)


def tema(s: pd.Series, length: int = 20) -> pd.Series:
    e1 = ema(s, length)
    e2 = ema(e1, length)
    return 3 * e1 - 3 * e2 + ema(e2, length)


def kama(s: pd.Series, length: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive MA -- speeds up in trends, flattens in chop."""
    s = series(s).astype(float)
    change = (s - s.shift(length)).abs()
    volatility = s.diff().abs().rolling(length).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0.0)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2

    values = s.to_numpy(dtype=float)
    smoothing = sc.to_numpy(dtype=float)
    out = np.full(len(s), np.nan)
    if len(s) <= length:
        return pd.Series(out, index=s.index)
    out[length] = values[length]
    for i in range(length + 1, len(s)):
        prev = out[i - 1]
        if np.isnan(prev) or np.isnan(values[i]):
            out[i] = values[i]
        else:
            out[i] = prev + smoothing[i] * (values[i] - prev)
    return pd.Series(out, index=s.index)


def vwma(df: pd.DataFrame, length: int = 20) -> pd.Series:
    d = ensure_ohlcv(df, need_volume=True)
    pv = (d["close"] * d["volume"]).rolling(length).sum()
    return pv / d["volume"].rolling(length).sum()


def alma(s: pd.Series, length: int = 21, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux MA -- Gaussian weights, tunable lag/smoothness tradeoff."""
    m = offset * (length - 1)
    sd = length / sigma
    weights = np.exp(-((np.arange(length) - m) ** 2) / (2 * sd**2))
    weights /= weights.sum()
    return series(s).rolling(length).apply(lambda w: float(np.dot(w, weights)), raw=True)


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Wilder's ADX/DMI.  ADX > 25 is the usual "there is a trend" gate."""
    d = ensure_ohlcv(df)
    up = d["high"].diff()
    down = -d["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr = wilder_smooth(true_range(d), length)
    plus_di = 100 * wilder_smooth(pd.Series(plus_dm, index=d.index), length) / atr
    minus_di = 100 * wilder_smooth(pd.Series(minus_dm, index=d.index), length) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({
        "adx": wilder_smooth(dx, length),
        "plus_di": plus_di,
        "minus_di": minus_di,
    })


def supertrend(df: pd.DataFrame, length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend -- the default trailing-stop line for Indian intraday desks.

    Returns the line, the direction (+1 long / -1 short) and the flip flag.
    """
    d = ensure_ohlcv(df)
    atr = wilder_smooth(true_range(d), length)
    hl2 = (d["high"] + d["low"]) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    close = d["close"].to_numpy(dtype=float)
    ub = upper.to_numpy(dtype=float)
    lb = lower.to_numpy(dtype=float)
    n = len(d)
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    trend = np.full(n, np.nan)

    start = int(np.argmax(~np.isnan(ub))) if not np.isnan(ub).all() else n
    if start >= n:
        return pd.DataFrame(
            {"supertrend": np.nan, "direction": np.nan, "flip": False}, index=d.index
        )

    final_ub[start], final_lb[start], trend[start] = ub[start], lb[start], 1.0
    for i in range(start + 1, n):
        # Bands only ratchet in the favourable direction while the trend holds.
        final_ub[i] = ub[i] if (ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = lb[i] if (lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]) else final_lb[i - 1]
        if trend[i - 1] == 1:
            trend[i] = -1.0 if close[i] < final_lb[i] else 1.0
        else:
            trend[i] = 1.0 if close[i] > final_ub[i] else -1.0

    line = np.where(trend == 1, final_lb, final_ub)
    direction = pd.Series(trend, index=d.index)
    return pd.DataFrame({
        "supertrend": pd.Series(line, index=d.index),
        "direction": direction,
        "flip": direction.ne(direction.shift(1)) & direction.notna() & direction.shift(1).notna(),
    })


def psar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    """Parabolic SAR (Wilder).  Returns the dot and the trend direction."""
    d = ensure_ohlcv(df)
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    n = len(d)
    out = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    if n < 2:
        return pd.DataFrame({"psar": out, "direction": trend}, index=d.index)

    bull = True
    af = step
    ep = high[0]
    sar = low[0]
    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if bull:
            sar = min(sar, low[i - 1], low[max(0, i - 2)])
            if low[i] < sar:                      # flip to bearish
                bull, sar, ep, af = False, ep, low[i], step
            elif high[i] > ep:
                ep, af = high[i], min(af + step, max_step)
        else:
            sar = max(sar, high[i - 1], high[max(0, i - 2)])
            if high[i] > sar:                     # flip to bullish
                bull, sar, ep, af = True, ep, high[i], step
            elif low[i] < ep:
                ep, af = low[i], min(af + step, max_step)
        out[i] = sar
        trend[i] = 1.0 if bull else -1.0
    return pd.DataFrame({"psar": out, "direction": trend}, index=d.index)


def ichimoku(
    df: pd.DataFrame, conversion: int = 9, base: int = 26, span_b: int = 52, displacement: int = 26
) -> pd.DataFrame:
    """Ichimoku Kinko Hyo.  Span A/B are forward-shifted (the cloud)."""
    d = ensure_ohlcv(df)

    def _mid(length: int) -> pd.Series:
        return (d["high"].rolling(length).max() + d["low"].rolling(length).min()) / 2

    tenkan = _mid(conversion)
    kijun = _mid(base)
    return pd.DataFrame({
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": ((tenkan + kijun) / 2).shift(displacement),
        "senkou_b": _mid(span_b).shift(displacement),
        "chikou": d["close"].shift(-displacement),
    })


def aroon(df: pd.DataFrame, length: int = 25) -> pd.DataFrame:
    d = ensure_ohlcv(df)
    up = d["high"].rolling(length + 1).apply(lambda w: float(np.argmax(w)) / length * 100, raw=True)
    dn = d["low"].rolling(length + 1).apply(lambda w: float(np.argmin(w)) / length * 100, raw=True)
    return pd.DataFrame({"aroon_up": up, "aroon_down": dn, "aroon_osc": up - dn})


def vortex(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    d = ensure_ohlcv(df)
    tr_sum = true_range(d).rolling(length).sum()
    vm_plus = (d["high"] - d["low"].shift(1)).abs().rolling(length).sum()
    vm_minus = (d["low"] - d["high"].shift(1)).abs().rolling(length).sum()
    return pd.DataFrame({"vi_plus": vm_plus / tr_sum, "vi_minus": vm_minus / tr_sum})


def trix(s: pd.Series, length: int = 15, signal: int = 9) -> pd.DataFrame:
    triple = ema(ema(ema(s, length), length), length)
    line = triple.pct_change() * 100
    return pd.DataFrame({"trix": line, "trix_signal": ema(line, signal)})


def dpo(s: pd.Series, length: int = 20) -> pd.Series:
    """Detrended price oscillator -- isolates the cycle from the trend."""
    shift = length // 2 + 1
    return series(s) - sma(s, length).shift(shift)


def mass_index(df: pd.DataFrame, length: int = 9, sum_length: int = 25) -> pd.Series:
    """Reversal warning: a bulge above 27 that then drops below 26.5."""
    d = ensure_ohlcv(df)
    rng = d["high"] - d["low"]
    ratio = ema(rng, length) / ema(ema(rng, length), length)
    return ratio.rolling(sum_length).sum()


def ma_ribbon_score(s: pd.Series, lengths: tuple[int, ...] = (5, 10, 20, 50, 100, 200)) -> pd.Series:
    """Fraction of the MA stack in bullish order, mapped to [-1, +1].

    A single number for "is the moving-average stack aligned", which is far more
    robust than any one crossover.
    """
    mas = [ema(s, n) for n in lengths]
    ordered = pd.Series(0.0, index=series(s).index)
    pairs = 0
    for i in range(len(mas) - 1):
        ordered = ordered.add((mas[i] > mas[i + 1]).astype(float), fill_value=0)
        pairs += 1
    price_above = (series(s) > mas[-1]).astype(float)
    raw = (ordered + price_above) / (pairs + 1)
    return raw * 2 - 1
