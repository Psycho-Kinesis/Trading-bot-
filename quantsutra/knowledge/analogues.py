"""Historical analogue search.

Finds past windows in a price series whose *shape* resembles the recent one,
using a scale-invariant comparison so a 2008 pattern at Nifty 4,000 can match
a 2024 one at Nifty 24,000.

Read the output carefully.  This is pattern similarity, not prediction:

* Financial series are noisy enough that visually striking matches occur by
  chance -- a random walk will always produce some.
* The forward returns reported are what happened *after those specific
  matches*, in a sample usually numbering single digits.
* Markets are non-stationary. The structural conditions of 2008 do not exist
  now, however similar the chart looks.

The forward-return dispersion is reported alongside the mean precisely so the
uncertainty is visible rather than implied.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from ..indicators._util import ensure_ohlcv
from .events import MARKET_EVENTS

__all__ = ["find_analogues", "shape_distance", "nearest_historical_events"]


def _normalise(window: np.ndarray) -> np.ndarray:
    """Scale to zero mean and unit variance so only shape is compared."""
    centred = window - window.mean()
    sd = centred.std()
    return centred / sd if sd > 0 else centred


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two z-normalised paths, per point."""
    if len(a) != len(b):
        raise ValueError("windows must be the same length")
    return float(np.sqrt(((_normalise(a) - _normalise(b)) ** 2).mean()))


def find_analogues(
    df: pd.DataFrame, window: int = 40, forward: int = 20, top_n: int = 5,
    exclude_recent: int = 60, min_separation: int | None = None,
) -> dict:
    """Find past windows most similar in shape to the most recent one.

    ``min_separation`` stops the result being five overlapping copies of the
    same episode -- without it, adjacent windows always look alike.
    """
    d = ensure_ohlcv(df)
    close = d["close"].to_numpy(dtype=float)
    n = len(close)
    min_separation = min_separation if min_separation is not None else window // 2

    if n < window + forward + exclude_recent + 20:
        return {
            "matches": [], "sample_size": 0,
            "note": (f"Need at least {window + forward + exclude_recent + 20} bars to "
                     f"search for analogues; have {n}."),
        }

    target = close[-window:]
    candidates: list[tuple[float, int]] = []
    last_start = n - window - forward - exclude_recent
    for start in range(0, last_start):
        segment = close[start: start + window]
        if np.isnan(segment).any():
            continue
        candidates.append((shape_distance(target, segment), start))

    candidates.sort()
    chosen: list[tuple[float, int]] = []
    for distance, start in candidates:
        if all(abs(start - other) >= min_separation for _, other in chosen):
            chosen.append((distance, start))
        if len(chosen) >= top_n:
            break

    matches = []
    forward_returns = []
    for distance, start in chosen:
        end = start + window
        entry = close[end - 1]
        exit_price = close[min(end + forward - 1, n - 1)]
        fwd = 100 * (exit_price - entry) / entry
        forward_returns.append(fwd)
        segment = close[start:end]
        matches.append({
            "start": str(d.index[start]), "end": str(d.index[end - 1]),
            "distance": round(distance, 4),
            "similarity": round(max(0.0, 100 * (1 - distance / 2)), 1),
            "window_return_pct": round(100 * (segment[-1] - segment[0]) / segment[0], 2),
            "forward_return_pct": round(fwd, 2),
            "forward_bars": forward,
            "nearby_events": [e.name for e in nearest_historical_events(d.index[start], 200)],
        })

    if not forward_returns:
        return {"matches": [], "sample_size": 0, "note": "no non-overlapping analogues found"}

    arr = np.array(forward_returns)
    positive = int((arr > 0).sum())
    return {
        "matches": matches,
        "sample_size": len(arr),
        "forward_mean_pct": round(float(arr.mean()), 2),
        "forward_median_pct": round(float(np.median(arr)), 2),
        "forward_std_pct": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 2),
        "forward_range_pct": [round(float(arr.min()), 2), round(float(arr.max()), 2)],
        "positive_share": round(positive / len(arr), 2),
        "note": (
            f"Based on {len(arr)} historical analogue(s). With a sample this small the "
            f"mean forward return is not a forecast -- the observed range was "
            f"{arr.min():.1f}% to {arr.max():.1f}%, and chart similarity does not imply "
            f"similar causes. Treat this as context, never as a signal."
        ),
    }


def nearest_historical_events(timestamp, window_days: int = 90) -> list:
    """Curated events within ``window_days`` of a timestamp."""
    try:
        date = pd.Timestamp(timestamp).date()
    except (TypeError, ValueError):
        return []
    return [e for e in MARKET_EVENTS if abs((e.date - date).days) <= window_days]
