"""Momentum and oscillator family."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import ensure_ohlcv, series, wilder_smooth
from .trend import ema, sma

__all__ = [
    "rsi", "stoch_rsi", "stochastic", "cci", "roc", "momentum", "williams_r",
    "ultimate_oscillator", "awesome_oscillator", "tsi", "cmo", "ppo",
    "connors_rsi", "rsi_divergence", "coppock",
]


def rsi(s: pd.Series, length: int = 14) -> pd.Series:
    """Wilder RSI."""
    delta = series(s).diff()
    gain = wilder_smooth(delta.clip(lower=0), length)
    loss = wilder_smooth(-delta.clip(upper=0), length)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # All-gain windows have zero average loss -> RSI is 100 by definition.
    return out.where(loss != 0, 100.0).where(gain.notna())


def stoch_rsi(s: pd.Series, length: int = 14, k: int = 3, d: int = 3) -> pd.DataFrame:
    r = rsi(s, length)
    lo = r.rolling(length).min()
    hi = r.rolling(length).max()
    raw = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    k_line = raw.rolling(k).mean()
    return pd.DataFrame({"stochrsi_k": k_line, "stochrsi_d": k_line.rolling(d).mean()})


def stochastic(df: pd.DataFrame, k_length: int = 14, k_smooth: int = 3, d: int = 3) -> pd.DataFrame:
    d_ = ensure_ohlcv(df)
    lo = d_["low"].rolling(k_length).min()
    hi = d_["high"].rolling(k_length).max()
    raw = 100 * (d_["close"] - lo) / (hi - lo).replace(0, np.nan)
    k_line = raw.rolling(k_smooth).mean()
    return pd.DataFrame({"stoch_k": k_line, "stoch_d": k_line.rolling(d).mean()})


def cci(df: pd.DataFrame, length: int = 20, constant: float = 0.015) -> pd.Series:
    d = ensure_ohlcv(df)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    ma = tp.rolling(length).mean()
    # CCI uses mean absolute deviation, not standard deviation.
    mad = tp.rolling(length).apply(lambda w: float(np.abs(w - w.mean()).mean()), raw=True)
    return (tp - ma) / (constant * mad.replace(0, np.nan))


def roc(s: pd.Series, length: int = 12) -> pd.Series:
    return series(s).pct_change(length) * 100


def momentum(s: pd.Series, length: int = 10) -> pd.Series:
    return series(s).diff(length)


def williams_r(df: pd.DataFrame, length: int = 14) -> pd.Series:
    d = ensure_ohlcv(df)
    hi = d["high"].rolling(length).max()
    lo = d["low"].rolling(length).min()
    return -100 * (hi - d["close"]) / (hi - lo).replace(0, np.nan)


def ultimate_oscillator(
    df: pd.DataFrame, short: int = 7, medium: int = 14, long: int = 28
) -> pd.Series:
    d = ensure_ohlcv(df)
    prev_close = d["close"].shift(1)
    true_low = pd.concat([d["low"], prev_close], axis=1).min(axis=1)
    true_high = pd.concat([d["high"], prev_close], axis=1).max(axis=1)
    bp = d["close"] - true_low
    tr = true_high - true_low

    def _avg(n: int) -> pd.Series:
        return bp.rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)

    return 100 * (4 * _avg(short) + 2 * _avg(medium) + _avg(long)) / 7


def awesome_oscillator(df: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.Series:
    d = ensure_ohlcv(df)
    median = (d["high"] + d["low"]) / 2
    return sma(median, fast) - sma(median, slow)


def tsi(s: pd.Series, long: int = 25, short: int = 13, signal: int = 13) -> pd.DataFrame:
    """True Strength Index -- double-smoothed momentum, very clean on indices."""
    diff = series(s).diff()
    smoothed = ema(ema(diff, long), short)
    abs_smoothed = ema(ema(diff.abs(), long), short)
    line = 100 * smoothed / abs_smoothed.replace(0, np.nan)
    return pd.DataFrame({"tsi": line, "tsi_signal": ema(line, signal)})


def cmo(s: pd.Series, length: int = 14) -> pd.Series:
    """Chande Momentum Oscillator, range [-100, 100]."""
    delta = series(s).diff()
    up = delta.clip(lower=0).rolling(length).sum()
    down = (-delta.clip(upper=0)).rolling(length).sum()
    return 100 * (up - down) / (up + down).replace(0, np.nan)


def ppo(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Percentage price oscillator -- MACD normalised, so comparable across
    NIFTY (~25k) and BANKNIFTY (~55k) without rescaling thresholds."""
    line = 100 * (ema(s, fast) - ema(s, slow)) / ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"ppo": line, "ppo_signal": sig, "ppo_hist": line - sig})


def connors_rsi(s: pd.Series, rsi_len: int = 3, streak_len: int = 2, rank_len: int = 100) -> pd.Series:
    """Connors RSI -- a mean-reversion oscillator that works well on index ETFs
    and on NIFTY at the daily timeframe."""
    s = series(s)
    price_rsi = rsi(s, rsi_len)

    change = s.diff()
    streak = np.zeros(len(s))
    for i in range(1, len(s)):
        if change.iloc[i] > 0:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] > 0 else 1
        elif change.iloc[i] < 0:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] < 0 else -1
    streak_rsi = rsi(pd.Series(streak, index=s.index), streak_len)

    pct = s.pct_change()
    rank = pct.rolling(rank_len).apply(
        lambda w: float((w[:-1] < w[-1]).mean() * 100) if not np.isnan(w).any() else np.nan,
        raw=True,
    )
    return (price_rsi + streak_rsi + rank) / 3


def coppock(s: pd.Series, long: int = 14, short: int = 11, wma_len: int = 10) -> pd.Series:
    """Coppock curve -- a long-horizon bottom finder, useful on the NIFTY monthly."""
    from .trend import wma

    return wma(roc(s, long) + roc(s, short), wma_len)


def rsi_divergence(
    df: pd.DataFrame, length: int = 14, lookback: int = 40, pivot: int = 5
) -> pd.DataFrame:
    """Detect regular bullish/bearish RSI divergence against price pivots.

    Bearish: price makes a higher high while RSI makes a lower high.
    Bullish:  price makes a lower low while RSI makes a higher low.

    Pivots are confirmed ``pivot`` bars later, so the flag appears with that
    lag -- it is deliberately not repainted.
    """
    d = ensure_ohlcv(df)
    r = rsi(d["close"], length)
    n = len(d)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)

    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    rv = r.to_numpy(dtype=float)

    def _is_pivot_high(i: int) -> bool:
        lo_i, hi_i = i - pivot, i + pivot + 1
        return lo_i >= 0 and hi_i <= n and high[i] == np.nanmax(high[lo_i:hi_i])

    def _is_pivot_low(i: int) -> bool:
        lo_i, hi_i = i - pivot, i + pivot + 1
        return lo_i >= 0 and hi_i <= n and low[i] == np.nanmin(low[lo_i:hi_i])

    pivot_highs = [i for i in range(pivot, n - pivot) if _is_pivot_high(i)]
    pivot_lows = [i for i in range(pivot, n - pivot) if _is_pivot_low(i)]

    for idx_list, is_high in ((pivot_highs, True), (pivot_lows, False)):
        for a, b in zip(idx_list, idx_list[1:]):
            if b - a > lookback or np.isnan(rv[a]) or np.isnan(rv[b]):
                continue
            confirm = min(b + pivot, n - 1)
            if is_high and high[b] > high[a] and rv[b] < rv[a]:
                bear[confirm] = True
            elif not is_high and low[b] < low[a] and rv[b] > rv[a]:
                bull[confirm] = True

    return pd.DataFrame(
        {"rsi": r, "bullish_divergence": bull, "bearish_divergence": bear}, index=d.index
    )
