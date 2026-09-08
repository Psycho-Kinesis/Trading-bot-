"""On-disk caching for feed responses.

Two reasons this matters beyond speed: it keeps you under NSE's rate limits,
and it means a backtest can be re-run against exactly the bytes the first run
saw, which is the difference between a reproducible result and an anecdote.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

__all__ = ["FeedCache", "cached_history"]


def _read_cached_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index, format="mixed")
    return frame


class FeedCache:
    def __init__(self, directory: str | Path = "data_cache", ttl_seconds: int = 3600):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds

    def _key(self, *parts) -> str:
        raw = "|".join(str(p) for p in parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _path(self, key: str, suffix: str) -> Path:
        return self.directory / f"{key}{suffix}"

    def get_frame(self, *parts) -> pd.DataFrame | None:
        key = self._key(*parts)
        # Parquet preserves dtypes and the timezone; CSV is the fallback when
        # pyarrow is not installed, which is common enough that silently
        # caching nothing would be a poor default.
        for suffix, reader in ((".parquet", pd.read_parquet), (".csv", _read_cached_csv)):
            path = self._path(key, suffix)
            if not path.exists():
                continue
            if self.ttl and (time.time() - path.stat().st_mtime) > self.ttl:
                continue
            try:
                return reader(path)
            except (OSError, ValueError, ImportError):
                continue
        return None

    def put_frame(self, frame: pd.DataFrame, *parts) -> None:
        key = self._key(*parts)
        try:
            frame.to_parquet(self._path(key, ".parquet"))
            return
        except (OSError, ValueError, ImportError, AttributeError):
            pass
        # Caching is an optimisation, never a hard dependency.
        with contextlib.suppress(OSError):
            frame.to_csv(self._path(key, ".csv"), index_label="timestamp")

    def get_json(self, *parts) -> dict | None:
        path = self._path(self._key(*parts), ".json")
        if not path.exists():
            return None
        if self.ttl and (time.time() - path.stat().st_mtime) > self.ttl:
            return None
        try:
            with path.open() as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def put_json(self, payload: dict, *parts) -> None:
        try:
            with self._path(self._key(*parts), ".json").open("w") as fh:
                json.dump(payload, fh)
        except (OSError, TypeError):
            pass

    def clear(self) -> int:
        count = 0
        for path in self.directory.glob("*"):
            try:
                path.unlink()
                count += 1
            except OSError:
                pass
        return count


def cached_history(feed, symbol: str, interval: str = "1d", lookback: int = 500,
                   cache: FeedCache | None = None, ttl: int = 3600) -> pd.DataFrame:
    """Fetch history through a cache, falling back to a stale copy on failure.

    If the feed is down, a stale cached frame is better than nothing -- but the
    caller is told, via the frame's ``attrs``, that it is stale.
    """
    cache = cache or FeedCache(ttl_seconds=ttl)
    key = (feed.name, symbol.upper(), interval, lookback)
    hit = cache.get_frame(*key)
    if hit is not None and not hit.empty:
        hit.attrs["cache"] = "hit"
        return hit
    try:
        frame = feed.history(symbol, interval, lookback)
        cache.put_frame(frame, *key)
        frame.attrs["cache"] = "miss"
        return frame
    except Exception:
        stale = FeedCache(cache.directory, ttl_seconds=0).get_frame(*key)
        if stale is not None and not stale.empty:
            stale.attrs["cache"] = "stale"
            stale.attrs["warning"] = "feed unavailable; serving stale cached data"
            return stale
        raise
