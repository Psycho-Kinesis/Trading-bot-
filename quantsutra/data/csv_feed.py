"""Local file feed.

The recommended source for anything serious.  Export history from your broker
(Zerodha, Angel One, Fyers, Upstox all provide it), drop the files in a
directory, and every part of the package works against data you control --
no rate limits, no undocumented endpoints, no silent schema changes.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from .base import FeedError, normalise_frame, validate_frame

__all__ = ["CsvFeed"]


class CsvFeed:
    """Reads ``<directory>/<SYMBOL>_<interval>.csv`` (or ``.parquet``).

    Column names are matched case-insensitively and the usual broker export
    variants (``Date``, ``timestamp``, ``Close Price``, ``LTP``) are accepted.
    """

    name = "csv"

    def __init__(self, directory: str | Path = "data", validate: bool = True):
        self.directory = Path(directory)
        self.validate = validate
        self.last_issues: list[str] = []

    def _find(self, symbol: str, interval: str) -> Path:
        stem_options = [
            f"{symbol.upper()}_{interval}", f"{symbol.lower()}_{interval}",
            f"{symbol.upper()}-{interval}", symbol.upper(), symbol.lower(),
        ]
        for stem in stem_options:
            for suffix in (".csv", ".parquet", ".CSV"):
                path = self.directory / f"{stem}{suffix}"
                if path.exists():
                    return path
        available = sorted(p.name for p in self.directory.glob("*")) if self.directory.exists() else []
        raise FeedError(
            f"No file for {symbol} {interval} in {self.directory}. Looked for "
            f"{stem_options[0]}.csv among others. Files present: {available[:12] or 'none'}"
        )

    def history(self, symbol: str, interval: str = "1d", lookback: int = 500,
                start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
        path = self._find(symbol, interval)
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frame = normalise_frame(frame)

        if start is not None:
            frame = frame[frame.index.date >= start]
        if end is not None:
            frame = frame[frame.index.date <= end]

        self.last_issues = validate_frame(frame, symbol) if self.validate else []
        return frame.tail(lookback) if lookback else frame

    def quote(self, symbol: str) -> dict:
        frame = self.history(symbol, "1d", lookback=2)
        last = frame.iloc[-1]
        prev = frame.iloc[-2] if len(frame) > 1 else last
        return {
            "symbol": symbol.upper(), "price": float(last["close"]),
            "previous_close": float(prev["close"]),
            "change": float(last["close"] - prev["close"]),
            "change_pct": float(100 * (last["close"] - prev["close"]) / prev["close"]),
            "day_high": float(last["high"]), "day_low": float(last["low"]),
            "as_of": str(frame.index[-1]),
            "source": f"local file {self._find(symbol, '1d').name} (not live)",
        }

    def save(self, frame: pd.DataFrame, symbol: str, interval: str = "1d") -> Path:
        """Persist a frame in the layout :meth:`history` expects."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{symbol.upper()}_{interval}.csv"
        frame.to_csv(path)
        return path
