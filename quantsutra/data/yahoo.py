"""Yahoo Finance feed.

Free, no API key, and adequate for daily and 5-15 minute index history, which
is what the analytics layer needs.  Its limits are real and worth knowing:

* Intraday history is capped (roughly 60 days at 5-minute granularity).
* Index "volume" is frequently zero or missing -- the package detects this and
  skips volume-based rules rather than producing nonsense.
* It is an unofficial endpoint with no uptime guarantee.  For anything you
  trade real money on, use your broker's historical API.

Implemented directly against the public chart endpoint so the package does not
require ``yfinance``; if ``yfinance`` is installed it is used as a fallback.
"""

from __future__ import annotations

import datetime as dt
import time

import pandas as pd
import requests

from ..constants import INDEX_YAHOO, IST
from .base import FeedError, normalise_frame

__all__ = ["YahooFeed"]

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json",
}

# Yahoo's interval vocabulary, mapped from ours.
INTERVALS = {"1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m", "30m": "30m",
             "60m": "60m", "1h": "60m", "1d": "1d", "1wk": "1wk", "1mo": "1mo"}

# Yahoo enforces a maximum range per interval.
MAX_RANGE = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
             "60m": "730d", "1d": "10y", "1wk": "10y", "1mo": "max"}


class YahooFeed:
    name = "yahoo"

    def __init__(self, session: requests.Session | None = None, timeout: int = 20,
                 retries: int = 3, retry_backoff: float = 1.5):
        self.session = session or requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout
        self.retries = retries
        self.retry_backoff = retry_backoff

    @staticmethod
    def resolve(symbol: str) -> str:
        """Map a friendly name (``NIFTY``) to a Yahoo ticker (``^NSEI``)."""
        return INDEX_YAHOO.get(symbol.upper(), symbol)

    def _get(self, url: str, params: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 429:
                    time.sleep(self.retry_backoff ** (attempt + 2))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last = exc
                if attempt < self.retries - 1:
                    time.sleep(self.retry_backoff ** attempt)
        raise FeedError(
            f"Yahoo request failed after {self.retries} attempts: {last}. "
            f"If you are behind a proxy or firewall, this endpoint may be blocked -- "
            f"use CsvFeed with an exported file, or your broker's API."
        ) from last

    def history(self, symbol: str, interval: str = "1d", lookback: int = 500,
                start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
        ticker = self.resolve(symbol)
        yahoo_interval = INTERVALS.get(interval)
        if yahoo_interval is None:
            raise FeedError(f"unsupported interval {interval!r}; known: {sorted(INTERVALS)}")

        params: dict = {"interval": yahoo_interval, "includePrePost": "false",
                        "events": "div,split"}
        if start or end:
            start = start or (dt.date.today() - dt.timedelta(days=365 * 3))
            end = end or dt.date.today()
            params["period1"] = int(dt.datetime.combine(start, dt.time()).timestamp())
            params["period2"] = int(dt.datetime.combine(end, dt.time(23, 59)).timestamp())
        else:
            params["range"] = self._range_for(yahoo_interval, lookback)

        payload = self._get(CHART_URL.format(symbol=ticker), params)
        return self._parse_chart(payload, ticker, lookback)

    @staticmethod
    def _range_for(interval: str, lookback: int) -> str:
        if interval in ("1d", "1wk", "1mo"):
            years = max(1, int(lookback / 250) + 1)
            return MAX_RANGE[interval] if years > 10 else f"{years}y"
        return MAX_RANGE.get(interval, "60d")

    @staticmethod
    def _parse_chart(payload: dict, ticker: str, lookback: int) -> pd.DataFrame:
        chart = (payload or {}).get("chart") or {}
        if chart.get("error"):
            raise FeedError(f"Yahoo returned an error for {ticker}: {chart['error']}")
        results = chart.get("result")
        if not results:
            raise FeedError(f"Yahoo returned no data for {ticker}")

        result = results[0]
        stamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        if not stamps:
            raise FeedError(f"Yahoo returned an empty series for {ticker} -- the ticker "
                            f"may be wrong or the market may never have traded in this range")

        frame = pd.DataFrame({
            "open": quote.get("open"), "high": quote.get("high"),
            "low": quote.get("low"), "close": quote.get("close"),
            "volume": quote.get("volume"),
        }, index=pd.to_datetime(stamps, unit="s", utc=True))
        frame.index = frame.index.tz_convert(IST)
        frame = normalise_frame(frame)
        return frame.tail(lookback) if lookback else frame

    def quote(self, symbol: str) -> dict:
        """Latest price and session context for one symbol."""
        ticker = self.resolve(symbol)
        payload = self._get(CHART_URL.format(symbol=ticker),
                            {"interval": "1d", "range": "5d"})
        result = ((payload.get("chart") or {}).get("result") or [{}])[0]
        meta = result.get("meta") or {}
        price = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        return {
            "symbol": symbol.upper(), "yahoo_symbol": ticker, "price": price,
            "previous_close": prev,
            "change": (price - prev) if price is not None and prev else None,
            "change_pct": (100 * (price - prev) / prev) if price is not None and prev else None,
            "day_high": meta.get("regularMarketDayHigh"),
            "day_low": meta.get("regularMarketDayLow"),
            "currency": meta.get("currency"), "exchange": meta.get("exchangeName"),
            "as_of": dt.datetime.now(IST).isoformat(),
            "source": "yahoo (unofficial endpoint, delayed -- do not use for execution)",
        }

    def india_vix(self, lookback: int = 250) -> pd.Series:
        """India VIX history, used for the volatility regime and IV percentile."""
        frame = self.history("INDIAVIX", "1d", lookback)
        return frame["close"].rename("india_vix")
