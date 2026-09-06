"""Candlestick pattern recognition.

Two design decisions worth knowing about:

1. Every pattern is *context-scaled*.  A 30-point NIFTY body is a doji in a
   volatile week and a marubozu in a quiet one, so body/shadow tests are
   measured against a rolling average range rather than fixed point values.

2. Each detection carries a ``strength`` in [0, 1] and a ``context`` flag
   saying whether the pattern appeared at a meaningful location (after a
   pullback, at a band edge, at a swing).  A bullish engulfing in the middle of
   a range is noise; the same bar at the lower Bollinger band after a
   three-bar decline is a signal.  The signal engine only rewards the latter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..indicators._util import ensure_ohlcv

__all__ = ["CandleSignal", "detect_candles", "candle_anatomy", "PATTERN_BIAS"]

# Direction each pattern implies: +1 bullish, -1 bearish, 0 indecision.
PATTERN_BIAS: dict[str, int] = {
    "doji": 0, "dragonfly_doji": 1, "gravestone_doji": -1, "long_legged_doji": 0,
    "spinning_top": 0,
    "hammer": 1, "inverted_hammer": 1, "hanging_man": -1, "shooting_star": -1,
    "bullish_marubozu": 1, "bearish_marubozu": -1,
    "bullish_engulfing": 1, "bearish_engulfing": -1,
    "bullish_harami": 1, "bearish_harami": -1, "harami_cross": 0,
    "piercing_line": 1, "dark_cloud_cover": -1,
    "morning_star": 1, "evening_star": -1,
    "morning_doji_star": 1, "evening_doji_star": -1,
    "three_white_soldiers": 1, "three_black_crows": -1,
    "three_inside_up": 1, "three_inside_down": -1,
    "three_outside_up": 1, "three_outside_down": -1,
    "tweezer_bottom": 1, "tweezer_top": -1,
    "bullish_belt_hold": 1, "bearish_belt_hold": -1,
    "rising_three_methods": 1, "falling_three_methods": -1,
    "bullish_kicker": 1, "bearish_kicker": -1,
    "inside_bar": 0, "outside_bar": 0,
}


@dataclass
class CandleSignal:
    name: str
    bias: int                 # +1 / 0 / -1
    index: int
    timestamp: object
    strength: float           # 0..1, how textbook the formation is
    context_ok: bool          # did it appear somewhere that matters
    notes: str = ""
    meta: dict = field(default_factory=dict)


def candle_anatomy(df: pd.DataFrame, ref_length: int = 20) -> pd.DataFrame:
    """Per-bar geometry, normalised by a rolling average range."""
    d = ensure_ohlcv(df)
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    avg_range = (h - l).rolling(ref_length).mean().replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    return pd.DataFrame({
        "body": body,
        "body_pct": body / rng,
        "upper_shadow": upper,
        "lower_shadow": lower,
        "upper_pct": upper / rng,
        "lower_pct": lower / rng,
        "range": h - l,
        "rel_range": (h - l) / avg_range,
        "rel_body": body / avg_range,
        "bullish": c > o,
        "bearish": c < o,
        "mid": (o + c) / 2,
    })


def _trend_context(d: pd.DataFrame, i: int, lookback: int = 5) -> int:
    """Crude prior-move direction: +1 if the market rose into the bar, -1 if it
    fell, 0 if flat.  Reversal patterns require the opposing prior move."""
    if i < lookback + 1:
        return 0
    closes = d["close"].to_numpy(dtype=float)
    change = closes[i - 1] - closes[i - 1 - lookback]
    # Scale against the *average* range of the prior bars, not the current bar:
    # a wide reversal bar would otherwise mask the move that preceded it.
    ranges = (d["high"].to_numpy(float) - d["low"].to_numpy(float))[i - lookback: i]
    ref = float(np.nanmean(ranges)) if len(ranges) else 0.0
    if ref <= 0 or np.isnan(change):
        return 0
    if change > ref:
        return 1
    if change < -ref:
        return -1
    return 0


def detect_candles(
    df: pd.DataFrame,
    lookback: int = 1,
    ref_length: int = 20,
    require_context: bool = False,
) -> list[CandleSignal]:
    """Scan the last ``lookback`` bars and return every pattern found.

    ``require_context=True`` drops patterns that fired without the prior move
    they need to mean anything (a hammer in an uptrend, say).
    """
    d = ensure_ohlcv(df)
    a = candle_anatomy(d, ref_length)
    n = len(d)
    if n < 5:
        return []

    o = d["open"].to_numpy(float); h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float); c = d["close"].to_numpy(float)
    body_pct = a["body_pct"].to_numpy(float)
    upper_pct = a["upper_pct"].to_numpy(float)
    lower_pct = a["lower_pct"].to_numpy(float)
    rel_body = a["rel_body"].to_numpy(float)
    rel_range = a["rel_range"].to_numpy(float)
    bull = a["bullish"].to_numpy(bool)
    bear = a["bearish"].to_numpy(bool)

    out: list[CandleSignal] = []
    start = max(4, n - lookback)

    def add(name: str, i: int, strength: float, ctx: bool, notes: str = "", **meta) -> None:
        if require_context and not ctx:
            return
        out.append(CandleSignal(
            name=name, bias=PATTERN_BIAS.get(name, 0), index=i, timestamp=d.index[i],
            strength=float(np.clip(strength, 0.0, 1.0)), context_ok=bool(ctx),
            notes=notes, meta=meta,
        ))

    for i in range(start, n):
        if np.isnan(rel_body[i]) or np.isnan(body_pct[i]):
            continue
        prior = _trend_context(d, i)
        b, u, lo_, rb, rr = body_pct[i], upper_pct[i], lower_pct[i], rel_body[i], rel_range[i]

        # ---- single-bar -------------------------------------------------
        if b < 0.1 and rr > 0.4:
            if lo_ > 0.6 and u < 0.15:
                add("dragonfly_doji", i, 0.55 + lo_ * 0.4, prior == -1, "rejection of lows")
            elif u > 0.6 and lo_ < 0.15:
                add("gravestone_doji", i, 0.55 + u * 0.4, prior == 1, "rejection of highs")
            elif u > 0.3 and lo_ > 0.3:
                add("long_legged_doji", i, 0.5 + rr * 0.2, True, "two-sided rejection, indecision")
            else:
                add("doji", i, 0.4, True, "indecision")
        elif b < 0.35 and u > 0.2 and lo_ > 0.2:
            add("spinning_top", i, 0.35, True, "indecision")

        if 0.1 <= b <= 0.4 and lo_ >= 2.0 * b and u <= 0.15 and rr > 0.7:
            if prior == -1:
                add("hammer", i, 0.55 + min(lo_, 0.8) * 0.5, True, "long lower wick after a decline")
            elif prior == 1:
                add("hanging_man", i, 0.5 + min(lo_, 0.8) * 0.4, True, "long lower wick after a rally")

        if 0.1 <= b <= 0.4 and u >= 2.0 * b and lo_ <= 0.15 and rr > 0.7:
            if prior == 1:
                add("shooting_star", i, 0.55 + min(u, 0.8) * 0.5, True, "long upper wick after a rally")
            elif prior == -1:
                add("inverted_hammer", i, 0.45 + min(u, 0.8) * 0.4, True, "long upper wick after a decline")

        if b > 0.9 and rr > 1.2:
            add("bullish_marubozu" if bull[i] else "bearish_marubozu", i,
                0.6 + min(rr, 2.5) * 0.15, True, "full-body conviction bar")

        # Belt hold: opens at the extreme and closes near the other end.
        if bull[i] and b > 0.7 and lo_ < 0.05 and rr > 1.0 and prior == -1:
            add("bullish_belt_hold", i, 0.55, True, "opened on the low and never looked back")
        if bear[i] and b > 0.7 and u < 0.05 and rr > 1.0 and prior == 1:
            add("bearish_belt_hold", i, 0.55, True, "opened on the high and sold off")

        # ---- two-bar ----------------------------------------------------
        p = i - 1
        if not np.isnan(rel_body[p]):
            body_i = abs(c[i] - o[i]); body_p = abs(c[p] - o[p])

            if bull[i] and bear[p] and c[i] >= o[p] and o[i] <= c[p] and body_i > body_p:
                add("bullish_engulfing", i, 0.55 + min(body_i / max(body_p, 1e-9) / 4, 0.4),
                    prior == -1, "buyers engulfed the prior bar", ratio=round(body_i / max(body_p, 1e-9), 2))
            if bear[i] and bull[p] and c[i] <= o[p] and o[i] >= c[p] and body_i > body_p:
                add("bearish_engulfing", i, 0.55 + min(body_i / max(body_p, 1e-9) / 4, 0.4),
                    prior == 1, "sellers engulfed the prior bar", ratio=round(body_i / max(body_p, 1e-9), 2))

            if body_i < body_p * 0.6 and max(o[i], c[i]) < max(o[p], c[p]) and min(o[i], c[i]) > min(o[p], c[p]):
                if bear[p] and bull[i]:
                    add("bullish_harami", i, 0.45, prior == -1, "selling pressure contracted")
                elif bull[p] and bear[i]:
                    add("bearish_harami", i, 0.45, prior == 1, "buying pressure contracted")
                if body_pct[i] < 0.1:
                    add("harami_cross", i, 0.5, True, "inside doji, momentum stalled")

            # Piercing / dark cloud need a gap open against the prior close.
            mid_p = (o[p] + c[p]) / 2
            if bear[p] and bull[i] and o[i] < l[p] and c[i] > mid_p and c[i] < o[p]:
                add("piercing_line", i, 0.55 + (c[i] - mid_p) / max(o[p] - c[p], 1e-9) * 0.3,
                    prior == -1, "gapped down and recovered past the midpoint")
            if bull[p] and bear[i] and o[i] > h[p] and c[i] < mid_p and c[i] > o[p]:
                add("dark_cloud_cover", i, 0.55 + (mid_p - c[i]) / max(c[p] - o[p], 1e-9) * 0.3,
                    prior == 1, "gapped up and closed below the midpoint")

            tol = 0.1 * np.nanmean([a["range"].to_numpy(float)[p], a["range"].to_numpy(float)[i]])
            if tol > 0:
                if abs(l[i] - l[p]) <= tol and prior == -1 and bull[i]:
                    add("tweezer_bottom", i, 0.45, True, "matched lows rejected twice")
                if abs(h[i] - h[p]) <= tol and prior == 1 and bear[i]:
                    add("tweezer_top", i, 0.45, True, "matched highs rejected twice")

            # Kicker: a gap in the opposite direction with no overlap at all.
            if bear[p] and bull[i] and o[i] > o[p] and l[i] > h[p]:
                add("bullish_kicker", i, 0.75, True, "full gap reversal, no overlap")
            if bull[p] and bear[i] and o[i] < o[p] and h[i] < l[p]:
                add("bearish_kicker", i, 0.75, True, "full gap reversal, no overlap")

            if h[i] <= h[p] and l[i] >= l[p]:
                add("inside_bar", i, 0.3, True, "compression, awaiting expansion")
            if h[i] > h[p] and l[i] < l[p]:
                add("outside_bar", i, 0.35, True, "both sides taken out")

        # ---- three-bar --------------------------------------------------
        q = i - 2
        if q >= 0 and not np.isnan(rel_body[q]):
            mid_q = (o[q] + c[q]) / 2
            small_middle = abs(c[p] - o[p]) < abs(c[q] - o[q]) * 0.5

            if bear[q] and small_middle and bull[i] and c[i] > mid_q and max(o[p], c[p]) < c[q]:
                name = "morning_doji_star" if body_pct[p] < 0.1 else "morning_star"
                add(name, i, 0.65 + (c[i] - mid_q) / max(o[q] - c[q], 1e-9) * 0.25,
                    prior == -1, "three-bar bottoming sequence")
            if bull[q] and small_middle and bear[i] and c[i] < mid_q and min(o[p], c[p]) > c[q]:
                name = "evening_doji_star" if body_pct[p] < 0.1 else "evening_star"
                add(name, i, 0.65 + (mid_q - c[i]) / max(c[q] - o[q], 1e-9) * 0.25,
                    prior == 1, "three-bar topping sequence")

            if bull[q] and bull[p] and bull[i] and c[i] > c[p] > c[q] and \
               o[p] > o[q] and o[i] > o[p] and min(body_pct[q], body_pct[p], body_pct[i]) > 0.55 and \
               max(upper_pct[q], upper_pct[p], upper_pct[i]) < 0.35:
                add("three_white_soldiers", i, 0.8, True, "three strong consecutive advances")
            if bear[q] and bear[p] and bear[i] and c[i] < c[p] < c[q] and \
               o[p] < o[q] and o[i] < o[p] and min(body_pct[q], body_pct[p], body_pct[i]) > 0.55 and \
               max(lower_pct[q], lower_pct[p], lower_pct[i]) < 0.35:
                add("three_black_crows", i, 0.8, True, "three strong consecutive declines")

            # Three inside/outside = harami / engulfing plus confirmation.
            inside_p = max(o[p], c[p]) < max(o[q], c[q]) and min(o[p], c[p]) > min(o[q], c[q])
            if inside_p and bear[q] and bull[p] and bull[i] and c[i] > h[q]:
                add("three_inside_up", i, 0.7, prior == -1, "harami confirmed by a breakout close")
            if inside_p and bull[q] and bear[p] and bear[i] and c[i] < l[q]:
                add("three_inside_down", i, 0.7, prior == 1, "harami confirmed by a breakdown close")
            if bear[q] and bull[p] and c[p] > o[q] and o[p] < c[q] and bull[i] and c[i] > c[p]:
                add("three_outside_up", i, 0.7, prior == -1, "engulfing confirmed the next bar")
            if bull[q] and bear[p] and c[p] < o[q] and o[p] > c[q] and bear[i] and c[i] < c[p]:
                add("three_outside_down", i, 0.7, prior == 1, "engulfing confirmed the next bar")

        # ---- five-bar continuation --------------------------------------
        if i >= 4:
            first, last = i - 4, i
            mids = range(i - 3, i)
            if bull[first] and bull[last] and c[last] > c[first] and rel_body[first] > 1.0 and \
               all(bear[m] or body_pct[m] < 0.4 for m in mids) and \
               all(h[m] <= h[first] and l[m] >= l[first] for m in mids):
                add("rising_three_methods", i, 0.65, True, "shallow pause inside a strong up bar")
            if bear[first] and bear[last] and c[last] < c[first] and rel_body[first] > 1.0 and \
               all(bull[m] or body_pct[m] < 0.4 for m in mids) and \
               all(h[m] <= h[first] and l[m] >= l[first] for m in mids):
                add("falling_three_methods", i, 0.65, True, "shallow pause inside a strong down bar")

    return out
