"""Pivot points and Central Pivot Range.

CPR (Central Pivot Range) is used very widely by Indian intraday traders: a
*narrow* CPR relative to yesterday's implies a trending day, a *wide* CPR
implies a range day.  That single classification is a genuinely useful daily
prior, so it is computed explicitly here rather than left to the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import ensure_ohlcv

__all__ = ["classic_pivots", "fibonacci_pivots", "camarilla_pivots", "woodie_pivots", "cpr", "nearest_levels"]


def _prev(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    d = ensure_ohlcv(df)
    return d["high"].shift(1), d["low"].shift(1), d["close"].shift(1), d["open"]


def classic_pivots(df: pd.DataFrame) -> pd.DataFrame:
    h, lo, c, _ = _prev(df)
    p = (h + lo + c) / 3
    rng = h - lo
    return pd.DataFrame({
        "pivot": p,
        "r1": 2 * p - lo, "s1": 2 * p - h,
        "r2": p + rng, "s2": p - rng,
        "r3": h + 2 * (p - lo), "s3": lo - 2 * (h - p),
        "r4": h + 3 * (p - lo), "s4": lo - 3 * (h - p),
    })


def fibonacci_pivots(df: pd.DataFrame) -> pd.DataFrame:
    h, lo, c, _ = _prev(df)
    p = (h + lo + c) / 3
    rng = h - lo
    return pd.DataFrame({
        "pivot": p,
        "r1": p + 0.382 * rng, "s1": p - 0.382 * rng,
        "r2": p + 0.618 * rng, "s2": p - 0.618 * rng,
        "r3": p + 1.000 * rng, "s3": p - 1.000 * rng,
    })


def camarilla_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Camarilla -- H3/L3 are the classic intraday reversal band, H4/L4 the
    breakout trigger."""
    h, lo, c, _ = _prev(df)
    rng = h - lo
    return pd.DataFrame({
        "pivot": (h + lo + c) / 3,
        "h1": c + rng * 1.1 / 12, "l1": c - rng * 1.1 / 12,
        "h2": c + rng * 1.1 / 6, "l2": c - rng * 1.1 / 6,
        "h3": c + rng * 1.1 / 4, "l3": c - rng * 1.1 / 4,
        "h4": c + rng * 1.1 / 2, "l4": c - rng * 1.1 / 2,
        "h5": c + rng * 1.1 / 2 * 1.168, "l5": c - rng * 1.1 / 2 * 1.168,
    })


def woodie_pivots(df: pd.DataFrame) -> pd.DataFrame:
    h, lo, c, o = _prev(df)
    p = (h + lo + 2 * o) / 4
    return pd.DataFrame({
        "pivot": p,
        "r1": 2 * p - lo, "s1": 2 * p - h,
        "r2": p + (h - lo), "s2": p - (h - lo),
    })


def cpr(df: pd.DataFrame) -> pd.DataFrame:
    """Central Pivot Range plus its width classification.

    ``cpr_width_pct`` is the CPR width as a fraction of price; ``cpr_type`` is
    NARROW / NORMAL / WIDE relative to the trailing distribution.  Narrow CPR
    days are the ones worth taking breakout setups on.
    """
    d = ensure_ohlcv(df)
    h, lo, c = d["high"].shift(1), d["low"].shift(1), d["close"].shift(1)
    pivot = (h + lo + c) / 3
    bc = (h + lo) / 2
    tc = 2 * pivot - bc
    top = pd.concat([tc, bc], axis=1).max(axis=1)
    bottom = pd.concat([tc, bc], axis=1).min(axis=1)
    width = top - bottom
    width_pct = 100 * width / c.replace(0, np.nan)

    med = width_pct.rolling(20).median()
    cpr_type = pd.Series("NORMAL", index=d.index, dtype=object)
    cpr_type = cpr_type.where(~(width_pct < med * 0.6), "NARROW")
    cpr_type = cpr_type.where(~(width_pct > med * 1.6), "WIDE")
    cpr_type = cpr_type.where(med.notna(), None)

    prev_top, prev_bottom = top.shift(1), bottom.shift(1)
    return pd.DataFrame({
        "cpr_top": top, "cpr_pivot": pivot, "cpr_bottom": bottom,
        "cpr_width": width, "cpr_width_pct": width_pct, "cpr_type": cpr_type,
        # A CPR that sits entirely above/below yesterday's is a directional prior.
        "cpr_higher_value": bottom > prev_top,
        "cpr_lower_value": top < prev_bottom,
        "cpr_overlapping": (bottom <= prev_top) & (top >= prev_bottom),
    })


def nearest_levels(price: float, levels: dict[str, float], count: int = 3) -> dict:
    """Split a dict of named levels into the nearest supports and resistances."""
    valid = {k: float(v) for k, v in levels.items() if v is not None and np.isfinite(v)}
    above = sorted(((v, k) for k, v in valid.items() if v > price))[:count]
    below = sorted(((v, k) for k, v in valid.items() if v < price), reverse=True)[:count]
    return {
        "resistance": [{"name": k, "level": round(v, 2), "distance_pct": round(100 * (v - price) / price, 3)} for v, k in above],
        "support": [{"name": k, "level": round(v, 2), "distance_pct": round(100 * (v - price) / price, 3)} for v, k in below],
    }
