"""Data feeds: Yahoo, NSE, local files, synthetic, plus caching and validation."""

from .base import DataFeed, FeedError, normalise_frame, resample_ohlcv, validate_frame
from .cache import FeedCache, cached_history
from .csv_feed import CsvFeed
from .nse import NseFeed
from .synthetic import (generate_index_series, generate_intraday_series,
                        generate_vix_series)
from .yahoo import YahooFeed

__all__ = [
    "DataFeed", "FeedError", "normalise_frame", "validate_frame", "resample_ohlcv",
    "YahooFeed", "NseFeed", "CsvFeed", "FeedCache", "cached_history",
    "generate_index_series", "generate_intraday_series", "generate_vix_series",
]


def get_feed(name: str = "yahoo", **kwargs):
    """Factory: ``get_feed("yahoo")``, ``get_feed("csv", directory="data")``."""
    feeds = {"yahoo": YahooFeed, "nse": NseFeed, "csv": CsvFeed}
    key = name.lower()
    if key not in feeds:
        raise ValueError(f"unknown feed {name!r}; available: {sorted(feeds)}")
    return feeds[key](**kwargs)
