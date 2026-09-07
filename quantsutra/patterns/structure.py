"""Market structure: trend structure, breaks of structure, imbalances and sweeps.

This is the price-action layer that answers "who is in control right now"
without reference to any indicator.  It is the highest-weight input to the
signal engine because structure leads indicators by construction -- an EMA
cross is a lagging restatement of a break of structure that already happened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators._util import ensure_ohlcv
from ..indicators.volatility import atr
from .swings import last_swings, zigzag


def _last_atr(d: pd.DataFrame, length: int = 14) -> float:
    """Latest ATR, reusing the ``atr_14`` column when the caller already has it.

    ``build_context`` passes the full feature frame to every pattern helper, so
    recomputing ATR in each of them was pure duplicated work.
    """
    if length == 14 and "atr_14" in d.columns:
        value = d["atr_14"].iloc[-1]
        if value is not None and np.isfinite(value) and value > 0:
            return float(value)
    series = atr(d, length)
    value = series.iloc[-1] if len(series) else np.nan
    return float(value) if np.isfinite(value) and value > 0 else float("nan")

__all__ = ["StructureState", "market_structure", "fair_value_gaps", "order_blocks",
           "liquidity_sweeps", "inside_outside_sequence"]


def _min_after(arr: np.ndarray, start: int, default: float) -> float:
    """min of arr[start:], or ``default`` when that slice is empty."""
    tail = arr[start:]
    return float(np.nanmin(tail)) if tail.size else default


def _max_after(arr: np.ndarray, start: int, default: float) -> float:
    tail = arr[start:]
    return float(np.nanmax(tail)) if tail.size else default


@dataclass
class StructureState:
    trend: str                    # "UPTREND" | "DOWNTREND" | "RANGE"
    label: str                    # "HH-HL" | "LH-LL" | "MIXED"
    last_event: str | None        # "BOS_UP" | "BOS_DOWN" | "CHOCH_UP" | "CHOCH_DOWN"
    last_event_index: int | None
    swing_high: float | None
    swing_low: float | None
    bars_since_event: int | None
    strength: float               # 0..1

    def to_dict(self) -> dict:
        return {
            "trend": self.trend, "label": self.label, "last_event": self.last_event,
            "swing_high": round(self.swing_high, 2) if self.swing_high else None,
            "swing_low": round(self.swing_low, 2) if self.swing_low else None,
            "bars_since_event": self.bars_since_event, "strength": round(self.strength, 2),
        }


def market_structure(df: pd.DataFrame, atr_mult: float = 1.5) -> StructureState:
    """Classify structure from the last four alternating swings.

    * BOS (break of structure) -- a new extreme in the direction of the trend.
    * CHoCH (change of character) -- the first break *against* the trend, which
      is the earliest reliable warning that a trend is over.
    """
    d = ensure_ohlcv(df)
    if len(d) < 30:
        return StructureState("RANGE", "MIXED", None, None, None, None, None, 0.0)

    pivots = zigzag(d, use_atr=True, atr_mult=atr_mult)
    swings = last_swings(pivots, 4)
    n = len(d)
    close = d["close"].to_numpy(float)

    highs = [s for s in swings if s.is_high]
    lows = [s for s in swings if not s.is_high]
    swing_high = highs[-1].price if highs else None
    swing_low = lows[-1].price if lows else None

    label, trend, strength = "MIXED", "RANGE", 0.2
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price
        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price
        if hh and hl:
            label, trend, strength = "HH-HL", "UPTREND", 0.8
        elif lh and ll:
            label, trend, strength = "LH-LL", "DOWNTREND", 0.8
        elif hh and ll:
            label, trend, strength = "EXPANDING", "RANGE", 0.3
        elif lh and hl:
            label, trend, strength = "CONTRACTING", "RANGE", 0.4

    # Walk the pivot sequence forward to find the most recent BOS/CHoCH.
    last_event: str | None = None
    last_event_index: int | None = None
    direction = 0
    prior_high = prior_low = None
    for s in pivots:
        if s.is_high:
            if prior_high is not None and s.price > prior_high:
                last_event = "BOS_UP" if direction >= 0 else "CHOCH_UP"
                last_event_index = s.confirmed_at
                direction = 1
            prior_high = s.price if prior_high is None else max(prior_high, s.price) if direction == 1 else s.price
        else:
            if prior_low is not None and s.price < prior_low:
                last_event = "BOS_DOWN" if direction <= 0 else "CHOCH_DOWN"
                last_event_index = s.confirmed_at
                direction = -1
            prior_low = s.price if prior_low is None else min(prior_low, s.price) if direction == -1 else s.price

    # A close beyond the most recent swing that the pivot list has not yet
    # confirmed is a live break -- report it immediately rather than waiting.
    if swing_high is not None and close[-1] > swing_high:
        last_event = "BOS_UP" if trend == "UPTREND" else "CHOCH_UP"
        last_event_index = n - 1
    elif swing_low is not None and close[-1] < swing_low:
        last_event = "BOS_DOWN" if trend == "DOWNTREND" else "CHOCH_DOWN"
        last_event_index = n - 1

    return StructureState(
        trend=trend, label=label, last_event=last_event, last_event_index=last_event_index,
        swing_high=swing_high, swing_low=swing_low,
        bars_since_event=(n - 1 - last_event_index) if last_event_index is not None else None,
        strength=strength,
    )


def fair_value_gaps(df: pd.DataFrame, lookback: int = 60, min_atr: float = 0.2) -> list[dict]:
    """Three-bar imbalances (fair value gaps).

    A bullish FVG exists when bar i-1's high is below bar i+1's low: price
    moved so fast that a band of prices was never traded through.  These bands
    act as magnets and as pullback entries.  Gaps already filled are dropped.
    """
    d = ensure_ohlcv(df).tail(lookback + 2)
    if len(d) < 5:
        return []
    a = _last_atr(d, 14)
    if not np.isfinite(a) or a <= 0:
        return []
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    n = len(d)
    out = []
    for i in range(1, n - 1):
        # bullish gap: nothing traded between bar i-1's high and bar i+1's low
        if low[i + 1] > high[i - 1]:
            top, bottom = float(low[i + 1]), float(high[i - 1])
            lowest_since = _min_after(low, i + 2, top)
            if top - bottom >= a * min_atr and lowest_since > bottom:
                out.append({"kind": "BULLISH", "top": round(top, 2), "bottom": round(bottom, 2),
                            "index": i, "bars_ago": n - 1 - i,
                            "size_atr": round((top - bottom) / a, 2),
                            "filled_pct": round(min(1.0, max(0.0, (top - lowest_since) / (top - bottom))), 2)})
        # bearish gap
        if high[i + 1] < low[i - 1]:
            top, bottom = float(low[i - 1]), float(high[i + 1])
            highest_since = _max_after(high, i + 2, bottom)
            if top - bottom >= a * min_atr and highest_since < top:
                out.append({"kind": "BEARISH", "top": round(top, 2), "bottom": round(bottom, 2),
                            "index": i, "bars_ago": n - 1 - i,
                            "size_atr": round((top - bottom) / a, 2),
                            "filled_pct": round(min(1.0, max(0.0, (highest_since - bottom) / (top - bottom))), 2)})
    return sorted(out, key=lambda g: g["bars_ago"])[:10]


def order_blocks(df: pd.DataFrame, lookback: int = 80, impulse_atr: float = 1.5) -> list[dict]:
    """The last opposing candle before an impulsive move.

    Practically: the zone institutional flow left behind. Only blocks that
    price has not yet traded back through are returned.
    """
    d = ensure_ohlcv(df).tail(lookback)
    if len(d) < 10:
        return []
    a = _last_atr(d, 14)
    if not np.isfinite(a) or a <= 0:
        return []
    o = d["open"].to_numpy(float)
    c = d["close"].to_numpy(float)
    h = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    n = len(d)
    out = []
    for i in range(1, n - 2):
        impulse = c[i + 1] - o[i + 1]
        if abs(impulse) < a * impulse_atr:
            continue
        if impulse > 0 and c[i] < o[i]:                # bullish OB: last down candle
            top, bottom = float(max(o[i], c[i])), float(lo[i])
            lowest_since = _min_after(lo, i + 2, top)
            if lowest_since > bottom:
                out.append({"kind": "BULLISH", "top": round(top, 2), "bottom": round(bottom, 2),
                            "index": i, "bars_ago": n - 1 - i,
                            "impulse_atr": round(float(abs(impulse)) / a, 2),
                            "mitigated": bool(lowest_since <= top)})
        elif impulse < 0 and c[i] > o[i]:              # bearish OB: last up candle
            top, bottom = float(h[i]), float(min(o[i], c[i]))
            highest_since = _max_after(h, i + 2, bottom)
            if highest_since < top:
                out.append({"kind": "BEARISH", "top": round(top, 2), "bottom": round(bottom, 2),
                            "index": i, "bars_ago": n - 1 - i,
                            "impulse_atr": round(float(abs(impulse)) / a, 2),
                            "mitigated": bool(highest_since >= bottom)})
    return sorted(out, key=lambda b: b["bars_ago"])[:8]


def liquidity_sweeps(df: pd.DataFrame, lookback: int = 60, pivot: int = 3) -> list[dict]:
    """Stop-hunt detection: a wick through a prior swing that closes back inside.

    On NIFTY this is extremely common in the first 15 minutes and just before
    3pm on expiry days, and it is the highest-quality reversal trigger the
    structure layer produces -- the market took liquidity and rejected the level.
    """
    d = ensure_ohlcv(df).tail(lookback)
    if len(d) < pivot * 2 + 5:
        return []
    from .swings import swing_points

    pivots = swing_points(d, pivot, pivot)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    close = d["close"].to_numpy(float)
    n = len(d)
    a = _last_atr(d, 14)
    if not np.isfinite(a) or a <= 0:
        return []

    out = []
    for s in pivots:
        for i in range(s.confirmed_at + 1, n):
            if s.is_high and high[i] > s.price and close[i] < s.price:
                out.append({"kind": "SELL_SIDE_SWEEP", "bias": -1, "level": round(s.price, 2),
                            "index": i, "bars_ago": n - 1 - i,
                            "wick_atr": round(float(high[i] - s.price) / a, 2),
                            "note": "buy stops above the swing high were taken, then rejected"})
                break
            if (not s.is_high) and low[i] < s.price and close[i] > s.price:
                out.append({"kind": "BUY_SIDE_SWEEP", "bias": 1, "level": round(s.price, 2),
                            "index": i, "bars_ago": n - 1 - i,
                            "wick_atr": round(float(s.price - low[i]) / a, 2),
                            "note": "sell stops below the swing low were taken, then rejected"})
                break
    return sorted(out, key=lambda x: x["bars_ago"])[:6]


def inside_outside_sequence(df: pd.DataFrame, lookback: int = 10) -> dict:
    """Compression/expansion state from consecutive inside and outside bars."""
    d = ensure_ohlcv(df).tail(lookback + 1)
    h = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    inside = outside = 0
    for i in range(len(d) - 1, 0, -1):
        if h[i] <= h[i - 1] and lo[i] >= lo[i - 1]:
            inside += 1
            outside = 0
        elif h[i] > h[i - 1] and lo[i] < lo[i - 1]:
            outside += 1
            break
        else:
            break
    return {
        "consecutive_inside_bars": inside,
        "outside_bar": outside > 0,
        "state": "COMPRESSING" if inside >= 2 else ("EXPANDING" if outside else "NORMAL"),
    }
