"""Classical chart patterns built on confirmed ZigZag swings.

Each detector returns the pattern's neckline/trigger, a measured-move target
and a stop, so a detection is directly tradable rather than merely descriptive.
Patterns are reported as ``FORMING`` or ``CONFIRMED``: a head-and-shoulders is
only actionable once the neckline breaks, and conflating the two is how most
pattern scanners produce fantasy signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..indicators._util import ensure_ohlcv
from ..indicators.volatility import atr
from .swings import Swing, zigzag

__all__ = ["ChartPattern", "detect_chart_patterns"]


@dataclass
class ChartPattern:
    name: str
    bias: int                    # +1 bullish, -1 bearish, 0 neutral/either-way
    status: str                  # "FORMING" | "CONFIRMED"
    start_index: int
    end_index: int
    trigger: float               # break level that confirms the pattern
    target: float | None         # measured move
    stop: float | None
    confidence: float            # 0..1
    notes: str = ""
    points: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "bias": self.bias, "status": self.status,
            "trigger": round(self.trigger, 2),
            "target": round(self.target, 2) if self.target is not None else None,
            "stop": round(self.stop, 2) if self.stop is not None else None,
            "confidence": round(self.confidence, 2), "notes": self.notes,
        }


def _rel(a: float, b: float) -> float:
    """Relative difference between two prices."""
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom


def detect_chart_patterns(
    df: pd.DataFrame,
    zigzag_pct: float = 1.2,
    use_atr_zigzag: bool = True,
    max_patterns: int = 12,
) -> list[ChartPattern]:
    """Scan the frame for classical formations.

    Only patterns whose last pivot is recent (within the trailing third of the
    frame) are returned -- a head-and-shoulders from 200 bars ago is history,
    not a trade.
    """
    d = ensure_ohlcv(df)
    if len(d) < 40:
        return []
    pivots = zigzag(d, zigzag_pct, use_atr=use_atr_zigzag, atr_mult=2.0)
    if len(pivots) < 4:
        pivots = zigzag(d, max(0.5, zigzag_pct / 2), use_atr=False)
    if len(pivots) < 4:
        return []

    a = float(atr(d, 14).iloc[-1])
    if not np.isfinite(a) or a <= 0:
        a = float(d["close"].iloc[-1]) * 0.005
    close = d["close"].to_numpy(float)
    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    n = len(d)
    last = close[-1]
    recent_cut = int(n * 0.55)

    found: list[ChartPattern] = []

    def broke_above(level: float, since: int) -> bool:
        return bool(np.nanmax(close[since:]) > level) if since < n else False

    def broke_below(level: float, since: int) -> bool:
        return bool(np.nanmin(close[since:]) < level) if since < n else False

    # ---- five-pivot formations: head & shoulders ------------------------
    for k in range(len(pivots) - 4):
        p = pivots[k:k + 5]
        if p[-1].index < recent_cut:
            continue
        kinds = "".join("H" if s.is_high else "L" for s in p)

        if kinds == "HLHLH":       # left shoulder, head, right shoulder (tops)
            ls, t1, head, t2, rs = p
            if head.price > ls.price and head.price > rs.price and _rel(ls.price, rs.price) < 0.04:
                neck = (t1.price + t2.price) / 2
                if abs(t1.price - t2.price) < a * 3:
                    height = head.price - neck
                    confirmed = broke_below(neck, rs.index)
                    found.append(ChartPattern(
                        "head_and_shoulders", -1, "CONFIRMED" if confirmed else "FORMING",
                        ls.index, rs.index, neck, neck - height, head.price,
                        0.72 if confirmed else 0.45,
                        "classic top; measured move is the head-to-neckline height",
                        [(s.index, s.price) for s in p],
                    ))
        if kinds == "LHLHL":       # inverse head & shoulders (bottoms)
            ls, t1, head, t2, rs = p
            if head.price < ls.price and head.price < rs.price and _rel(ls.price, rs.price) < 0.04:
                neck = (t1.price + t2.price) / 2
                if abs(t1.price - t2.price) < a * 3:
                    height = neck - head.price
                    confirmed = broke_above(neck, rs.index)
                    found.append(ChartPattern(
                        "inverse_head_and_shoulders", 1, "CONFIRMED" if confirmed else "FORMING",
                        ls.index, rs.index, neck, neck + height, head.price,
                        0.72 if confirmed else 0.45,
                        "classic bottom; measured move is the head-to-neckline height",
                        [(s.index, s.price) for s in p],
                    ))

    # ---- four/five-pivot: double and triple tops/bottoms ----------------
    for k in range(len(pivots) - 2):
        p = pivots[k:k + 3]
        if p[-1].index < recent_cut:
            continue
        kinds = "".join("H" if s.is_high else "L" for s in p)
        if kinds == "HLH":
            h1, trough, h2 = p
            if _rel(h1.price, h2.price) < 0.02 and (h1.price - trough.price) > a * 2:
                height = (h1.price + h2.price) / 2 - trough.price
                confirmed = broke_below(trough.price, h2.index)
                found.append(ChartPattern(
                    "double_top", -1, "CONFIRMED" if confirmed else "FORMING",
                    h1.index, h2.index, trough.price, trough.price - height,
                    max(h1.price, h2.price), 0.65 if confirmed else 0.4,
                    "two rejections at the same level; trigger is the intervening low",
                    [(s.index, s.price) for s in p],
                ))
        if kinds == "LHL":
            l1, peak, l2 = p
            if _rel(l1.price, l2.price) < 0.02 and (peak.price - l1.price) > a * 2:
                height = peak.price - (l1.price + l2.price) / 2
                confirmed = broke_above(peak.price, l2.index)
                found.append(ChartPattern(
                    "double_bottom", 1, "CONFIRMED" if confirmed else "FORMING",
                    l1.index, l2.index, peak.price, peak.price + height,
                    min(l1.price, l2.price), 0.65 if confirmed else 0.4,
                    "two defences of the same level; trigger is the intervening high",
                    [(s.index, s.price) for s in p],
                ))

    for k in range(len(pivots) - 4):
        p = pivots[k:k + 5]
        if p[-1].index < recent_cut:
            continue
        kinds = "".join("H" if s.is_high else "L" for s in p)
        if kinds == "HLHLH":
            h1, t1, h2, t2, h3 = p
            if max(_rel(h1.price, h2.price), _rel(h2.price, h3.price)) < 0.02:
                neck = min(t1.price, t2.price)
                height = np.mean([h1.price, h2.price, h3.price]) - neck
                confirmed = broke_below(neck, h3.index)
                found.append(ChartPattern(
                    "triple_top", -1, "CONFIRMED" if confirmed else "FORMING",
                    h1.index, h3.index, neck, neck - height, max(h1.price, h2.price, h3.price),
                    0.7 if confirmed else 0.45, "three rejections at one level",
                    [(s.index, s.price) for s in p],
                ))
        if kinds == "LHLHL":
            l1, t1, l2, t2, l3 = p
            if max(_rel(l1.price, l2.price), _rel(l2.price, l3.price)) < 0.02:
                neck = max(t1.price, t2.price)
                height = neck - float(np.mean([l1.price, l2.price, l3.price]))
                confirmed = broke_above(neck, l3.index)
                found.append(ChartPattern(
                    "triple_bottom", 1, "CONFIRMED" if confirmed else "FORMING",
                    l1.index, l3.index, neck, neck + height, min(l1.price, l2.price, l3.price),
                    0.7 if confirmed else 0.45, "three defences of one level",
                    [(s.index, s.price) for s in p],
                ))

    # ---- converging structures: triangles and wedges -------------------
    for k in range(len(pivots) - 3):
        p = pivots[k:k + 4]
        if p[-1].index < recent_cut:
            continue
        highs = [s for s in p if s.is_high]
        lows = [s for s in p if not s.is_high]
        if len(highs) < 2 or len(lows) < 2:
            continue
        h1, h2 = highs[0], highs[-1]
        l1, l2 = lows[0], lows[-1]
        if h2.index == h1.index or l2.index == l1.index:
            continue
        hs = (h2.price - h1.price) / (h2.index - h1.index)
        ls = (l2.price - l1.price) / (l2.index - l1.index)
        span = p[-1].index - p[0].index
        if span < 8:
            continue
        start_width = abs(h1.price - l1.price)
        end_width = abs(h2.price - l2.price)
        converging = end_width < start_width * 0.75
        flat_h = abs(hs) * span < a * 0.8
        flat_l = abs(ls) * span < a * 0.8
        height = max(start_width, a * 2)
        end_i = p[-1].index

        if converging and flat_h and ls > 0:
            trig = float(np.mean([h1.price, h2.price]))
            found.append(ChartPattern(
                "ascending_triangle", 1, "CONFIRMED" if broke_above(trig, end_i) else "FORMING",
                p[0].index, end_i, trig, trig + height, l2.price,
                0.62 if broke_above(trig, end_i) else 0.42,
                "flat resistance with rising lows; buyers paying up", [(s.index, s.price) for s in p]))
        elif converging and flat_l and hs < 0:
            trig = float(np.mean([l1.price, l2.price]))
            found.append(ChartPattern(
                "descending_triangle", -1, "CONFIRMED" if broke_below(trig, end_i) else "FORMING",
                p[0].index, end_i, trig, trig - height, h2.price,
                0.62 if broke_below(trig, end_i) else 0.42,
                "flat support with falling highs; sellers hitting bids", [(s.index, s.price) for s in p]))
        elif converging and hs < 0 and ls > 0:
            found.append(ChartPattern(
                "symmetrical_triangle", 0,
                "CONFIRMED" if (broke_above(h2.price, end_i) or broke_below(l2.price, end_i)) else "FORMING",
                p[0].index, end_i, h2.price, None, l2.price, 0.45,
                "coiling range; trade the break, not the middle -- direction unknown",
                [(s.index, s.price) for s in p]))
        elif converging and hs > 0 and ls > 0 and ls > hs:
            found.append(ChartPattern(
                "rising_wedge", -1, "CONFIRMED" if broke_below(l2.price, end_i) else "FORMING",
                p[0].index, end_i, l2.price, l2.price - height, h2.price,
                0.6 if broke_below(l2.price, end_i) else 0.4,
                "rising but narrowing; momentum fading into the highs",
                [(s.index, s.price) for s in p]))
        elif converging and hs < 0 and ls < 0 and hs < ls:
            found.append(ChartPattern(
                "falling_wedge", 1, "CONFIRMED" if broke_above(h2.price, end_i) else "FORMING",
                p[0].index, end_i, h2.price, h2.price + height, l2.price,
                0.6 if broke_above(h2.price, end_i) else 0.4,
                "falling but narrowing; selling pressure exhausting",
                [(s.index, s.price) for s in p]))
        elif not converging and end_width > start_width * 1.3 and hs > 0 and ls < 0:
            found.append(ChartPattern(
                "broadening_formation", 0, "FORMING", p[0].index, end_i, h2.price, None, l2.price,
                0.35, "expanding range -- widening stops, poor risk/reward for breakouts",
                [(s.index, s.price) for s in p]))

    # ---- rectangles / ranges -------------------------------------------
    for k in range(len(pivots) - 3):
        p = pivots[k:k + 4]
        if p[-1].index < recent_cut:
            continue
        highs = [s.price for s in p if s.is_high]
        lows = [s.price for s in p if not s.is_high]
        if len(highs) >= 2 and len(lows) >= 2:
            if _rel(max(highs), min(highs)) < 0.015 and _rel(max(lows), min(lows)) < 0.015:
                top, bottom = float(np.mean(highs)), float(np.mean(lows))
                if top - bottom > a * 2:
                    end_i = p[-1].index
                    if broke_above(top, end_i):
                        found.append(ChartPattern(
                            "rectangle_breakout", 1, "CONFIRMED", p[0].index, end_i,
                            top, top + (top - bottom), bottom, 0.6,
                            "range resolved upward", [(s.index, s.price) for s in p]))
                    elif broke_below(bottom, end_i):
                        found.append(ChartPattern(
                            "rectangle_breakdown", -1, "CONFIRMED", p[0].index, end_i,
                            bottom, bottom - (top - bottom), top, 0.6,
                            "range resolved downward", [(s.index, s.price) for s in p]))
                    else:
                        found.append(ChartPattern(
                            "rectangle_range", 0, "FORMING", p[0].index, end_i, top, None, bottom,
                            0.4, f"balanced range {bottom:.0f}-{top:.0f}; fade the edges",
                            [(s.index, s.price) for s in p]))

    # ---- flags and pennants (continuation after an impulse) ------------
    if len(pivots) >= 3:
        for k in range(len(pivots) - 2):
            pole_start, pole_end = pivots[k], pivots[k + 1]
            if pole_end.index < recent_cut - 20:
                continue
            move = pole_end.price - pole_start.price
            bars = pole_end.index - pole_start.index
            if bars < 2 or abs(move) < a * 4:
                continue
            seg_start = pole_end.index
            seg = close[seg_start:]
            if len(seg) < 4 or len(seg) > 40:
                continue
            retrace = (np.nanmax(seg) - np.nanmin(seg)) / abs(move)
            drift = np.nanmax(np.abs(seg - pole_end.price)) / abs(move)
            if retrace < 0.55 and drift < 0.62:
                bull = move > 0
                consol_high = float(np.nanmax(high[seg_start:]))
                consol_low = float(np.nanmin(low[seg_start:]))
                trig = consol_high if bull else consol_low
                target = trig + move
                found.append(ChartPattern(
                    "bull_flag" if bull else "bear_flag", 1 if bull else -1,
                    "CONFIRMED" if (broke_above(trig, seg_start) if bull else broke_below(trig, seg_start)) else "FORMING",
                    pole_start.index, n - 1, trig, target,
                    consol_low if bull else consol_high, 0.55,
                    "shallow consolidation after an impulse; measured move projects the pole",
                    [(pole_start.index, pole_start.price), (pole_end.index, pole_end.price)]))
                break

    # ---- cup and handle -------------------------------------------------
    if len(pivots) >= 4:
        for k in range(len(pivots) - 3):
            p = pivots[k:k + 4]
            if p[-1].index < recent_cut:
                continue
            if "".join("H" if s.is_high else "L" for s in p) != "HLHL":
                continue
            rim1, bottom, rim2, handle = p
            depth = rim1.price - bottom.price
            if depth <= a * 3 or _rel(rim1.price, rim2.price) > 0.03:
                continue
            handle_depth = rim2.price - handle.price
            if 0 < handle_depth < depth * 0.5 and (bottom.index - rim1.index) >= 5:
                trig = max(rim1.price, rim2.price)
                found.append(ChartPattern(
                    "cup_and_handle", 1, "CONFIRMED" if broke_above(trig, handle.index) else "FORMING",
                    rim1.index, handle.index, trig, trig + depth, handle.price,
                    0.6 if broke_above(trig, handle.index) else 0.42,
                    "rounded base with a shallow handle; target projects the cup depth",
                    [(s.index, s.price) for s in p]))
                break

    # Deduplicate by name, keeping the most recent/strongest of each.
    best: dict[str, ChartPattern] = {}
    for pat in found:
        prev = best.get(pat.name)
        if prev is None or (pat.end_index, pat.confidence) > (prev.end_index, prev.confidence):
            best[pat.name] = pat
    ranked = sorted(best.values(), key=lambda pt: (-pt.confidence, -pt.end_index))
    return ranked[:max_patterns]
