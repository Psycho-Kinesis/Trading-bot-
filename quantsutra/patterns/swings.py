"""Swing detection: fractals, ZigZag and swing-point series.

Everything downstream -- support/resistance, market structure, classical chart
patterns -- is built on a reliable set of swing points, so this module is
deliberately conservative.  A swing is only *confirmed* once ``right`` bars
have printed after it, and the confirmation index is reported separately from
the pivot index.  Nothing here repaints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators._util import ensure_ohlcv

__all__ = ["Swing", "fractals", "swing_points", "zigzag", "swing_series", "last_swings"]


@dataclass(frozen=True)
class Swing:
    index: int          # bar position of the pivot itself
    timestamp: object   # index label of the pivot
    price: float
    kind: str           # "HIGH" or "LOW"
    confirmed_at: int   # bar position at which it became known

    @property
    def is_high(self) -> bool:
        return self.kind == "HIGH"


def fractals(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.DataFrame:
    """Bill Williams style fractals: a bar higher/lower than its neighbours.

    Flags are placed on the *pivot* bar; use ``confirmed_at`` from
    :func:`swing_points` if you need to avoid look-ahead in a backtest.
    """
    d = ensure_ohlcv(df)
    high, low = d["high"], d["low"]
    window = left + right + 1
    roll_max = high.rolling(window, center=False).max().shift(-right)
    roll_min = low.rolling(window, center=False).min().shift(-right)
    return pd.DataFrame(
        {"fractal_high": (high >= roll_max) & high.notna(),
         "fractal_low": (low <= roll_min) & low.notna()},
        index=d.index,
    )


def swing_points(df: pd.DataFrame, left: int = 3, right: int = 3) -> list[Swing]:
    """All confirmed swing highs and lows, in chronological order."""
    d = ensure_ohlcv(df)
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    n = len(d)
    out: list[Swing] = []
    for i in range(left, n - right):
        window_h = high[i - left: i + right + 1]
        window_l = low[i - left: i + right + 1]
        if np.isnan(window_h).any() or np.isnan(window_l).any():
            continue
        if high[i] == window_h.max() and (window_h.argmax() == left):
            out.append(Swing(i, d.index[i], float(high[i]), "HIGH", i + right))
        if low[i] == window_l.min() and (window_l.argmin() == left):
            out.append(Swing(i, d.index[i], float(low[i]), "LOW", i + right))
    return sorted(out, key=lambda s: (s.index, s.kind))


def zigzag(df: pd.DataFrame, threshold_pct: float = 1.5, use_atr: bool = False,
           atr_mult: float = 2.0) -> list[Swing]:
    """ZigZag pivots using a percentage or ATR-scaled reversal threshold.

    ATR mode adapts to the instrument: a 1.5% swing means something very
    different on a quiet NIFTY week than on a volatile BANKNIFTY expiry.
    """
    d = ensure_ohlcv(df)
    if use_atr:
        from ..indicators.volatility import atr as _atr
        thresh = (atr_mult * _atr(d, 14) / d["close"] * 100).bfill().to_numpy(dtype=float)
    else:
        thresh = np.full(len(d), float(threshold_pct))

    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    n = len(d)
    if n < 3:
        return []

    pivots: list[Swing] = []
    direction = 0            # 0 = undecided, +1 = in an up-leg, -1 = down-leg
    hi_idx, hi_price = 0, high[0]
    lo_idx, lo_price = 0, low[0]
    ext_idx, ext_price = 0, high[0]

    for i in range(1, n):
        if direction == 0:
            # Track both extremes until one of them is retraced far enough to
            # establish which leg we are in.
            if high[i] >= hi_price:
                hi_idx, hi_price = i, high[i]
            if low[i] <= lo_price:
                lo_idx, lo_price = i, low[i]
            if hi_price > 0 and (hi_price - low[i]) / hi_price * 100 >= thresh[i]:
                pivots.append(Swing(hi_idx, d.index[hi_idx], float(hi_price), "HIGH", i))
                direction, ext_idx, ext_price = -1, i, low[i]
            elif lo_price > 0 and (high[i] - lo_price) / lo_price * 100 >= thresh[i]:
                pivots.append(Swing(lo_idx, d.index[lo_idx], float(lo_price), "LOW", i))
                direction, ext_idx, ext_price = 1, i, high[i]
        elif direction == 1:
            if high[i] >= ext_price:
                ext_idx, ext_price = i, high[i]
            elif ext_price > 0 and (ext_price - low[i]) / ext_price * 100 >= thresh[i]:
                pivots.append(Swing(ext_idx, d.index[ext_idx], float(ext_price), "HIGH", i))
                direction, ext_idx, ext_price = -1, i, low[i]
        else:
            if low[i] <= ext_price:
                ext_idx, ext_price = i, low[i]
            elif ext_price > 0 and (high[i] - ext_price) / ext_price * 100 >= thresh[i]:
                pivots.append(Swing(ext_idx, d.index[ext_idx], float(ext_price), "LOW", i))
                direction, ext_idx, ext_price = 1, i, high[i]
    return pivots


def swing_series(swings: list[Swing], kind: str | None = None) -> list[Swing]:
    return [s for s in swings if kind is None or s.kind == kind]


def last_swings(swings: list[Swing], count: int = 4) -> list[Swing]:
    """The most recent ``count`` alternating swing points, oldest first.

    Alternation matters -- two consecutive highs with no low between them make
    "higher high / higher low" structure tests meaningless.
    """
    ordered = sorted(swings, key=lambda s: s.index)
    alternating: list[Swing] = []
    for s in ordered:
        if alternating and alternating[-1].kind == s.kind:
            # keep the more extreme of the two same-kind pivots
            if (s.is_high and s.price > alternating[-1].price) or (
                not s.is_high and s.price < alternating[-1].price
            ):
                alternating[-1] = s
            continue
        alternating.append(s)
    return alternating[-count:]
