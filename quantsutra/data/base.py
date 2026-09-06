"""Data feed protocol and shared helpers.

Every feed returns the same shape -- a DatetimeIndex in IST with columns
``open, high, low, close, volume`` -- so the analytics layer never has to know
where the bars came from.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..constants import IST

__all__ = ["DataFeed", "normalise_frame", "validate_frame", "FeedError", "resample_ohlcv"]


class FeedError(RuntimeError):
    """Raised when a feed cannot return usable data."""


@runtime_checkable
class DataFeed(Protocol):
    name: str

    def history(self, symbol: str, interval: str = "1d", lookback: int = 500,
                start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
        ...

    def quote(self, symbol: str) -> dict:
        ...


def normalise_frame(df: pd.DataFrame, tz_localize: bool = True) -> pd.DataFrame:
    """Lower-case columns, sort, drop duplicate timestamps, localise to IST."""
    if df is None or df.empty:
        raise FeedError("feed returned an empty frame")
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    rename = {"adj_close": "adj_close", "last": "close", "ltp": "close", "vol": "volume"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    if not isinstance(out.index, pd.DatetimeIndex):
        # Named date columns first, then the unnamed leading column that
        # `DataFrame.to_csv` writes for the index -- without this, a CSV
        # round-trip silently reinterprets a RangeIndex as epoch nanoseconds
        # and every timestamp lands in 1970.
        candidates = ["date", "datetime", "timestamp", "time", "unnamed:_0", "index"]
        for candidate in candidates:
            if candidate in out.columns:
                parsed = pd.to_datetime(out[candidate], errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.9:
                    out = out.set_index(parsed).drop(columns=[candidate])
                    out.index.name = "timestamp"
                    break
        else:
            parsed = pd.to_datetime(out.index, errors="coerce")
            if isinstance(out.index, pd.RangeIndex) or parsed.isna().all():
                raise FeedError(
                    "frame has no usable timestamp: no date/datetime/timestamp column "
                    f"and the index is not parseable as dates (columns: {list(out.columns)})"
                )
            out.index = parsed

    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()

    if tz_localize and isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize(IST, ambiguous="NaT", nonexistent="shift_forward")
        else:
            out.index = out.index.tz_convert(IST)
        out = out[~out.index.isna()]

    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    if len(keep) < 4:
        raise FeedError(f"feed frame is missing OHLC columns; got {list(out.columns)}")
    out = out[keep]
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["open", "high", "low", "close"])


def validate_frame(df: pd.DataFrame, symbol: str = "") -> list[str]:
    """Return a list of data-quality problems.

    Bad data produces confident nonsense, so this runs before any analysis and
    the CLI prints whatever it finds.  Silent bad bars are worse than no bars.
    """
    issues: list[str] = []
    if df.empty:
        return [f"{symbol}: frame is empty"]

    if (df["high"] < df["low"]).any():
        issues.append(f"{symbol}: {int((df['high'] < df['low']).sum())} bar(s) have high < low")
    bad_range = ((df["close"] > df["high"]) | (df["close"] < df["low"])
                 | (df["open"] > df["high"]) | (df["open"] < df["low"]))
    if bad_range.any():
        issues.append(f"{symbol}: {int(bad_range.sum())} bar(s) have open/close outside the high-low range")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append(f"{symbol}: non-positive prices present")

    ret = df["close"].pct_change().abs()
    extreme = ret > 0.20
    if extreme.any():
        when = ", ".join(str(t.date()) for t in df.index[extreme][:3])
        issues.append(f"{symbol}: {int(extreme.sum())} bar(s) move more than 20% "
                      f"({when}) -- verify these are real and not split/feed artefacts")

    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 5:
        gaps = df.index.to_series().diff().dt.days
        big = gaps > 10
        if big.any():
            issues.append(f"{symbol}: {int(big.sum())} gap(s) longer than 10 days in the "
                          f"series -- the history may be incomplete")

    if "volume" in df and df["volume"].notna().any():
        zero_vol = (df["volume"].fillna(0) == 0).mean()
        if zero_vol > 0.2:
            issues.append(f"{symbol}: {zero_vol:.0%} of bars have zero volume -- index spot "
                          f"feeds often do; volume-based signals will be skipped")
    else:
        issues.append(f"{symbol}: no volume data; volume rules will be skipped")

    stale = (df["close"].diff() == 0).rolling(5).sum() >= 5
    if stale.any():
        issues.append(f"{symbol}: {int(stale.sum())} run(s) of 5+ unchanged closes -- "
                      f"possible stale or padded data")
    return issues


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample bars to a coarser interval (e.g. 5m -> 15m, 1d -> 1wk)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="right", closed="right").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])
