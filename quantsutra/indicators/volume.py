"""Volume, money-flow and market-profile style indicators.

Note on index volume: NIFTY/SENSEX spot "volume" from most free feeds is
unreliable or zero.  For volume-sensitive logic prefer futures volume or the
index ETF.  Every function here degrades to NaN rather than lying when volume
is absent, and :func:`has_usable_volume` lets callers check first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import ensure_ohlcv, percent_rank, true_range
from .trend import ema, sma

__all__ = [
    "has_usable_volume", "obv", "cmf", "mfi", "adl", "vwap", "anchored_vwap",
    "force_index", "ease_of_movement", "pvt", "volume_zscore", "volume_profile",
    "relative_volume", "klinger", "vwap_bands",
]


def has_usable_volume(df: pd.DataFrame, min_fraction: float = 0.8) -> bool:
    """True when volume is present and non-zero for most bars."""
    d = ensure_ohlcv(df)
    v = d["volume"]
    return bool(v.notna().mean() >= min_fraction and (v.fillna(0) > 0).mean() >= min_fraction)


def obv(df: pd.DataFrame) -> pd.Series:
    d = ensure_ohlcv(df, need_volume=True)
    direction = np.sign(d["close"].diff()).fillna(0.0)
    return (direction * d["volume"]).cumsum()


def _money_flow_multiplier(d: pd.DataFrame) -> pd.Series:
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    return ((d["close"] - d["low"]) - (d["high"] - d["close"])) / rng


def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Chaikin Money Flow, [-1, 1].  Sustained >0.15 = accumulation."""
    d = ensure_ohlcv(df, need_volume=True)
    mfv = _money_flow_multiplier(d).fillna(0.0) * d["volume"]
    return mfv.rolling(length).sum() / d["volume"].rolling(length).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Money Flow Index -- RSI weighted by volume."""
    d = ensure_ohlcv(df, need_volume=True)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    raw = tp * d["volume"]
    delta = tp.diff()
    pos = raw.where(delta > 0, 0.0).rolling(length).sum()
    neg = raw.where(delta < 0, 0.0).rolling(length).sum()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).where(neg != 0, 100.0)


def adl(df: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution line."""
    d = ensure_ohlcv(df, need_volume=True)
    return (_money_flow_multiplier(d).fillna(0.0) * d["volume"]).cumsum()


def vwap(df: pd.DataFrame, session_reset: bool = True) -> pd.Series:
    """Volume weighted average price.

    With ``session_reset`` (the default) the accumulator restarts each trading
    day, which is what an intraday VWAP means.  On a daily-bar frame it is a
    running VWAP instead.
    """
    d = ensure_ohlcv(df, need_volume=True)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    pv = tp * d["volume"]
    if session_reset and isinstance(d.index, pd.DatetimeIndex):
        day = pd.Series(d.index.normalize(), index=d.index)
        return pv.groupby(day).cumsum() / d["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv.cumsum() / d["volume"].cumsum().replace(0, np.nan)


def vwap_bands(df: pd.DataFrame, mults: tuple[float, ...] = (1.0, 2.0)) -> pd.DataFrame:
    """VWAP with standard-deviation bands -- the intraday mean-reversion map."""
    d = ensure_ohlcv(df, need_volume=True)
    v = vwap(d)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    if isinstance(d.index, pd.DatetimeIndex):
        day = pd.Series(d.index.normalize(), index=d.index)
        dev = ((tp - v) ** 2 * d["volume"]).groupby(day).cumsum() / d["volume"].groupby(day).cumsum()
    else:
        dev = ((tp - v) ** 2 * d["volume"]).cumsum() / d["volume"].cumsum()
    sd = np.sqrt(dev)
    out = {"vwap": v, "vwap_sd": sd}
    for m in mults:
        out[f"vwap_upper_{m:g}"] = v + m * sd
        out[f"vwap_lower_{m:g}"] = v - m * sd
    return pd.DataFrame(out)


def anchored_vwap(df: pd.DataFrame, anchor: int | str | pd.Timestamp) -> pd.Series:
    """VWAP anchored to a specific bar -- anchor to a swing high/low, an event
    day (budget, election result, RBI policy) or an expiry.
    """
    d = ensure_ohlcv(df, need_volume=True)
    if isinstance(anchor, int):
        pos = anchor if anchor >= 0 else len(d) + anchor
    else:
        pos = int(d.index.get_indexer([pd.Timestamp(anchor)], method="bfill")[0])
    if pos < 0 or pos >= len(d):
        raise ValueError(f"anchor {anchor!r} is outside the frame")
    tp = (d["high"] + d["low"] + d["close"]) / 3
    mask = np.arange(len(d)) >= pos
    pv = (tp * d["volume"]).where(mask, 0.0).cumsum()
    vv = d["volume"].where(mask, 0.0).cumsum()
    return (pv / vv.replace(0, np.nan)).where(mask)


def force_index(df: pd.DataFrame, length: int = 13) -> pd.Series:
    d = ensure_ohlcv(df, need_volume=True)
    return ema(d["close"].diff() * d["volume"], length)


def ease_of_movement(df: pd.DataFrame, length: int = 14, scale: float = 1e6) -> pd.Series:
    d = ensure_ohlcv(df, need_volume=True)
    distance = ((d["high"] + d["low"]) / 2).diff()
    box = (d["volume"] / scale) / (d["high"] - d["low"]).replace(0, np.nan)
    return (distance / box).rolling(length).mean()


def pvt(df: pd.DataFrame) -> pd.Series:
    d = ensure_ohlcv(df, need_volume=True)
    return (d["close"].pct_change() * d["volume"]).cumsum()


def klinger(df: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13) -> pd.DataFrame:
    d = ensure_ohlcv(df, need_volume=True)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    trend = np.sign(tp.diff()).fillna(0.0)
    vf = d["volume"] * trend
    line = ema(vf, fast) - ema(vf, slow)
    return pd.DataFrame({"kvo": line, "kvo_signal": ema(line, signal)})


def volume_zscore(df: pd.DataFrame, length: int = 20) -> pd.Series:
    d = ensure_ohlcv(df, need_volume=True)
    v = d["volume"]
    return (v - v.rolling(length).mean()) / v.rolling(length).std(ddof=0).replace(0, np.nan)


def relative_volume(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Volume vs its own average.  >2 on a breakout bar is real participation."""
    d = ensure_ohlcv(df, need_volume=True)
    return d["volume"] / d["volume"].rolling(length).mean().replace(0, np.nan)


def volume_profile(
    df: pd.DataFrame, bins: int = 50, lookback: int | None = None, value_area: float = 0.70
) -> dict:
    """Fixed-range volume profile.

    Returns the point of control (highest-volume price), the value area high/low
    containing ``value_area`` of traded volume, and the histogram.  POC and VAH
    /VAL act as magnets and are among the most reliable intraday levels.
    """
    d = ensure_ohlcv(df, need_volume=True)
    if lookback:
        d = d.tail(lookback)
    if d.empty or not np.isfinite(d["volume"].to_numpy(dtype=float)).any():
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "levels": [], "volumes": []}

    lo = float(d["low"].min())
    hi = float(d["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "levels": [], "volumes": []}

    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(bins)

    # Spread each bar's volume uniformly across the bins its range covers.
    for h, l, v in zip(d["high"].to_numpy(float), d["low"].to_numpy(float), d["volume"].to_numpy(float)):
        if not np.isfinite(v) or v <= 0 or not np.isfinite(h) or not np.isfinite(l):
            continue
        start = np.searchsorted(edges, l, side="right") - 1
        end = np.searchsorted(edges, h, side="left")
        start = max(0, min(start, bins - 1))
        end = max(start + 1, min(end, bins))
        hist[start:end] += v / (end - start)

    if hist.sum() <= 0:
        return {"poc": np.nan, "vah": np.nan, "val": np.nan, "levels": [], "volumes": []}

    poc_idx = int(np.argmax(hist))
    # Grow outward from the POC until value_area of total volume is enclosed.
    target = hist.sum() * value_area
    lo_i = hi_i = poc_idx
    acc = hist[poc_idx]
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        below = hist[lo_i - 1] if lo_i > 0 else -1.0
        above = hist[hi_i + 1] if hi_i < bins - 1 else -1.0
        if above >= below:
            hi_i += 1
            acc += hist[hi_i]
        else:
            lo_i -= 1
            acc += hist[lo_i]
    return {
        "poc": float(centers[poc_idx]),
        "vah": float(centers[hi_i]),
        "val": float(centers[lo_i]),
        "levels": centers.tolist(),
        "volumes": hist.tolist(),
    }
