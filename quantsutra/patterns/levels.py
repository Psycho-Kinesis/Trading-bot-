"""Support/resistance discovery, trendlines and channels.

Levels are found by *clustering* swing pivots rather than by picking the single
highest high: a price that has been rejected four times matters far more than
one that was touched once, and the cluster's touch count is what feeds the
signal engine's confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..indicators._util import ensure_ohlcv
from ..indicators.volatility import atr
from .swings import Swing, swing_points, zigzag


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

__all__ = [
    "Level",
    "Trendline",
    "confluence_zones",
    "find_levels",
    "fit_trendlines",
    "level_interaction",
    "reinforce_with_round_numbers",
    "round_number_levels",
]


@dataclass
class Level:
    price: float
    kind: str            # "SUPPORT" | "RESISTANCE" | "PIVOT"
    touches: int
    first_index: int
    last_index: int
    strength: float      # 0..1
    source: str = "swing"

    def distance_pct(self, price: float) -> float:
        return 100.0 * (self.price - price) / price


@dataclass
class Trendline:
    slope: float          # price units per bar
    intercept: float      # price at bar 0
    kind: str             # "SUPPORT" | "RESISTANCE"
    touches: int
    start_index: int
    end_index: int
    r_squared: float

    def value_at(self, index: int) -> float:
        return self.slope * index + self.intercept


def find_levels(
    df: pd.DataFrame,
    tolerance_atr: float = 0.5,
    min_touches: int = 2,
    lookback: int | None = 250,
    pivot_width: int = 3,
) -> list[Level]:
    """Cluster swing pivots into horizontal levels.

    ``tolerance_atr`` sets how close two pivots must be (in ATR units) to count
    as the same level -- an ATR-relative tolerance keeps the same code working
    on NIFTY, BANKNIFTY and SENSEX without retuning.
    """
    d = ensure_ohlcv(df)
    if lookback:
        d = d.tail(lookback)
    if len(d) < pivot_width * 2 + 5:
        return []

    a = _last_atr(d, 14)
    if not np.isfinite(a) or a <= 0:
        a = float(d["close"].iloc[-1]) * 0.005
    tol = a * tolerance_atr

    pivots = swing_points(d, pivot_width, pivot_width)
    if not pivots:
        return []

    # Greedy single-link clustering over sorted pivot prices.
    ordered = sorted(pivots, key=lambda s: s.price)
    clusters: list[list[Swing]] = [[ordered[0]]]
    for s in ordered[1:]:
        if s.price - clusters[-1][-1].price <= tol:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    last_price = float(d["close"].iloc[-1])
    n = len(d)
    levels: list[Level] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        price = float(np.mean([s.price for s in cluster]))
        highs = sum(1 for s in cluster if s.is_high)
        lows = len(cluster) - highs
        if highs > lows:
            kind = "RESISTANCE"
        elif lows > highs:
            kind = "SUPPORT"
        else:
            kind = "PIVOT"
        # A level that also flipped roles (was resistance, now support) is
        # stronger than one that only ever held from a single side.
        flipped = highs > 0 and lows > 0
        recency = 1.0 - (n - max(s.index for s in cluster)) / max(n, 1)
        strength = float(np.clip(
            0.25 * min(len(cluster), 5) / 5 * 2 + 0.3 * recency + (0.2 if flipped else 0.0), 0, 1
        ))
        levels.append(Level(
            price=price,
            kind="SUPPORT" if price < last_price and kind != "RESISTANCE" else
                 ("RESISTANCE" if price > last_price and kind != "SUPPORT" else kind),
            touches=len(cluster),
            first_index=min(s.index for s in cluster),
            last_index=max(s.index for s in cluster),
            strength=strength,
        ))
    return sorted(levels, key=lambda lv: lv.price)


def round_number_levels(price: float, step: int | None = None, count: int = 3,
                        atr_value: float | None = None) -> list[Level]:
    """Psychological round numbers.

    These matter more on Indian indices than most markets because option
    strikes sit on them: NIFTY 25000 is simultaneously a round number, a strike
    with heavy open interest, and a pin candidate on expiry day.

    The spacing is volatility-aware.  A 100-point grid is meaningful on a quiet
    NIFTY with a 90-point ATR and meaningless on a 425-point ATR day, where
    price crosses four of them in a single session -- so the step is widened
    until it is at least ~0.6 ATR.
    """
    if step is None:
        step = 500 if price > 30000 else (100 if price > 10000 else 50)
        if atr_value and np.isfinite(atr_value) and atr_value > 0:
            while step < atr_value * 0.6:
                step *= 5 if str(step)[0] == "1" else 2
    base = round(price / step) * step
    out: list[Level] = []
    for k in range(-count, count + 1):
        lv = base + k * step
        if lv <= 0 or abs(lv - price) / price > 0.06:
            continue
        # Bigger round numbers (multiples of 5x/10x the step) carry more weight,
        # but an *untested* round number is context, not a wall.  Strengths are
        # kept deliberately below the threshold that lets a level veto a trade;
        # a round number earns that status only by coinciding with a tested
        # swing level, which `reinforce_with_round_numbers` detects.
        weight = 0.20
        if lv % (step * 10) == 0:
            weight = 0.45
        elif lv % (step * 5) == 0:
            weight = 0.32
        out.append(Level(
            price=float(lv),
            kind="RESISTANCE" if lv > price else ("SUPPORT" if lv < price else "PIVOT"),
            touches=0, first_index=-1, last_index=-1, strength=weight, source="round_number",
        ))
    return out


def reinforce_with_round_numbers(levels: list[Level], atr_value: float) -> list[Level]:
    """Promote swing levels that coincide with a psychological round number.

    A swing high at 24,987 and a swing high at 25,000 are not equally strong:
    the latter is also a heavily-traded option strike.  Coincidence is what
    makes a level matter, so it is rewarded here rather than assumed.
    """
    if not levels or not np.isfinite(atr_value) or atr_value <= 0:
        return levels
    swings = [lv for lv in levels if lv.source == "swing"]
    rounds = [lv for lv in levels if lv.source == "round_number"]
    if not swings or not rounds:
        return levels
    tol = atr_value * 0.35
    for lv in swings:
        match = min(rounds, key=lambda r: abs(r.price - lv.price))
        if abs(match.price - lv.price) <= tol:
            lv.strength = float(min(1.0, lv.strength + 0.25))
            lv.source = "swing+round"
    return levels


def fit_trendlines(
    df: pd.DataFrame, lookback: int = 120, min_touches: int = 3, tolerance_atr: float = 0.6
) -> list[Trendline]:
    """Fit rising-support and falling-resistance lines through swing pivots.

    Every pair of pivots defines a candidate line; a line is kept when at least
    ``min_touches`` pivots sit on it and no *close* violates it badly.
    """
    d = ensure_ohlcv(df).tail(lookback).reset_index(drop=True)
    if len(d) < 20:
        return []
    a = _last_atr(d, 14)
    if not np.isfinite(a) or a <= 0:
        return []
    tol = a * tolerance_atr

    pivots = zigzag(d, use_atr=True, atr_mult=1.5)
    if len(pivots) < 3:
        pivots = swing_points(d, 3, 3)

    out: list[Trendline] = []
    for kind, want_high in (("RESISTANCE", True), ("SUPPORT", False)):
        pts = [p for p in pivots if p.is_high == want_high]
        if len(pts) < min_touches:
            continue
        best: Trendline | None = None
        for i in range(len(pts) - 1):
            for j in range(i + 1, len(pts)):
                x1, y1 = pts[i].index, pts[i].price
                x2, y2 = pts[j].index, pts[j].price
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                touches = [p for p in pts if abs(p.price - (slope * p.index + intercept)) <= tol]
                if len(touches) < min_touches:
                    continue
                # Reject lines that price has decisively broken.
                xs = np.arange(x1, len(d))
                line = slope * xs + intercept
                series = d["close"].to_numpy(float)[x1:]
                violation = (series - line if want_high else line - series)
                if np.nanmax(violation) > tol * 2.5:
                    continue
                tx = np.array([p.index for p in touches], dtype=float)
                ty = np.array([p.price for p in touches], dtype=float)
                pred = slope * tx + intercept
                ss_res = float(((ty - pred) ** 2).sum())
                ss_tot = float(((ty - ty.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
                cand = Trendline(slope, intercept, kind, len(touches),
                                 min(p.index for p in touches), max(p.index for p in touches), r2)
                if best is None or (cand.touches, cand.r_squared) > (best.touches, best.r_squared):
                    best = cand
        if best:
            out.append(best)
    return out


def level_interaction(price: float, levels: list[Level], atr_value: float,
                      min_strength: float = 0.0) -> dict:
    """Where price sits relative to the level map.

    ``at_level`` being true is what makes a reversal candle worth acting on and
    a breakout signal worth waiting for.

    ``min_strength`` filters out weak levels.  This matters for veto logic: a
    round number 0.3 ATR away is not a reason to skip a trade, but a swing
    level with four touches at the same distance is.
    """
    levels = [lv for lv in levels if lv.strength >= min_strength]
    if not levels or not np.isfinite(atr_value) or atr_value <= 0:
        return {"at_level": False, "nearest_support": None, "nearest_resistance": None,
                "room_to_resistance_atr": None, "room_to_support_atr": None}

    supports = [lv for lv in levels if lv.price < price]
    resistances = [lv for lv in levels if lv.price > price]
    ns = max(supports, key=lambda lv: lv.price) if supports else None
    nr = min(resistances, key=lambda lv: lv.price) if resistances else None
    nearest = min(levels, key=lambda lv: abs(lv.price - price))
    return {
        "at_level": abs(nearest.price - price) <= atr_value * 0.35,
        "nearest_level": round(nearest.price, 2),
        "nearest_level_strength": round(nearest.strength, 2),
        "nearest_support": round(ns.price, 2) if ns else None,
        "nearest_support_touches": ns.touches if ns else None,
        "nearest_resistance": round(nr.price, 2) if nr else None,
        "nearest_resistance_touches": nr.touches if nr else None,
        "room_to_resistance_atr": round((nr.price - price) / atr_value, 2) if nr else None,
        "room_to_support_atr": round((price - ns.price) / atr_value, 2) if ns else None,
    }


def confluence_zones(levels: list[Level], atr_value: float, min_sources: int = 2) -> list[dict]:
    """Merge levels from different sources that agree on a price.

    A swing level that coincides with a round number and a pivot is a far
    better place to fade or to target than any of the three alone.
    """
    if not levels or atr_value <= 0:
        return []
    tol = atr_value * 0.4
    ordered = sorted(levels, key=lambda lv: lv.price)
    zones: list[list[Level]] = [[ordered[0]]]
    for lv in ordered[1:]:
        if lv.price - zones[-1][-1].price <= tol:
            zones[-1].append(lv)
        else:
            zones.append([lv])
    out = []
    for zone in zones:
        # A level promoted by reinforce_with_round_numbers carries a composite
        # source ("swing+round"); split it so the zone lists each distinct
        # source once instead of printing "round_number+swing+round".
        sources = {part for lv in zone for part in lv.source.split("+")}
        sources = {"round_number" if s == "round" else s for s in sources}
        if len(sources) < min_sources:
            continue
        out.append({
            "price": round(float(np.mean([lv.price for lv in zone])), 2),
            "low": round(min(lv.price for lv in zone), 2),
            "high": round(max(lv.price for lv in zone), 2),
            "sources": sorted(sources),
            "total_touches": sum(lv.touches for lv in zone),
            "strength": round(float(min(1.0, sum(lv.strength for lv in zone) / 2)), 2),
        })
    return sorted(out, key=lambda z: -z["strength"])
