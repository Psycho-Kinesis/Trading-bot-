"""NSE India public endpoints.

This is where live option-chain data comes from -- open interest, IV and
strike-level prices for NIFTY and BANKNIFTY.

Three practical warnings:

* **NSE requires a cookie handshake.**  Requests without cookies from a prior
  homepage visit are rejected.  The session bootstrap is handled here, and it
  is re-run automatically when the cookies expire mid-session.
* **These are undocumented endpoints.**  They change without notice, they rate
  limit aggressively, and using them for high-frequency polling will get your
  IP blocked.  ``min_interval`` enforces a floor between requests.
* **The data is delayed and snapshot-based.**  Do not treat the option chain as
  a live quote for execution.

For anything with money on it, use a licensed broker API.
"""

from __future__ import annotations

import datetime as dt
import time

import requests

from ..calendar_in import lot_size, strike_step, today_ist, trading_days_between
from ..constants import IST
from ..options.chain import OptionChain, chain_from_records
from .base import FeedError

__all__ = ["NseFeed"]

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/option-chain",
}


class NseFeed:
    name = "nse"

    def __init__(self, timeout: int = 20, min_interval: float = 1.5, retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.timeout = timeout
        self.min_interval = min_interval
        self.retries = retries
        self._last_request = 0.0
        self._bootstrapped = False

    # -- session handling --------------------------------------------------
    def _bootstrap(self) -> None:
        """Visit the homepage to obtain the cookies the API endpoints require."""
        try:
            self.session.get(BASE, timeout=self.timeout)
            self.session.get(f"{BASE}/option-chain", timeout=self.timeout)
            self._bootstrapped = True
        except requests.RequestException as exc:
            raise FeedError(
                f"Could not reach nseindia.com to establish a session: {exc}. "
                f"NSE blocks datacentre IPs and many VPNs; if you are on a cloud host "
                f"this will not work. Use a broker API or a local machine."
            ) from exc

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request = time.time()

    def _get_json(self, path: str, params: dict | None = None) -> dict:
        if not self._bootstrapped:
            self._bootstrap()
        last: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                response = self.session.get(f"{BASE}{path}", params=params, timeout=self.timeout)
                if response.status_code in (401, 403):
                    # Cookies expired -- re-bootstrap once and try again.
                    self._bootstrapped = False
                    self._bootstrap()
                    continue
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
        raise FeedError(f"NSE request to {path} failed after {self.retries} attempts: {last}")

    # -- option chain ------------------------------------------------------
    def option_chain_raw(self, symbol: str = "NIFTY") -> dict:
        return self._get_json("/api/option-chain-indices", {"symbol": symbol.upper()})

    def option_chain(self, symbol: str = "NIFTY", expiry: dt.date | None = None) -> OptionChain:
        """Fetch and parse one expiry's option chain into an :class:`OptionChain`."""
        payload = self.option_chain_raw(symbol)
        return self.parse_option_chain(payload, symbol, expiry)

    @staticmethod
    def parse_option_chain(payload: dict, symbol: str = "NIFTY",
                           expiry: dt.date | None = None) -> OptionChain:
        """Parse NSE's option-chain JSON.

        Separated from the HTTP call so the parser can be tested offline and so
        a cached payload can be replayed.
        """
        records = (payload or {}).get("records") or {}
        rows = records.get("data") or []
        if not rows:
            raise FeedError("NSE option-chain payload contained no rows")

        spot = records.get("underlyingValue")
        if spot is None:
            raise FeedError("NSE option-chain payload has no underlyingValue")
        spot = float(spot)

        expiries = []
        for raw in records.get("expiryDates") or []:
            try:
                # NSE gives a bare date ("09-Sep-2025"); the naive intermediate
                # is discarded by .date() and never used as an instant.
                expiries.append(dt.datetime.strptime(raw, "%d-%b-%Y").date())  # noqa: DTZ007
            except ValueError:
                continue
        if expiry is None:
            expiry = min(expiries) if expiries else today_ist()

        want = expiry.strftime("%d-%b-%Y")
        flat: list[dict] = []
        for row in rows:
            if row.get("expiryDate") != want:
                continue
            strike = row.get("strikePrice")
            for side, key in (("CE", "CE"), ("PE", "PE")):
                leg = row.get(key)
                if not leg:
                    continue
                flat.append({
                    "strike": strike, "type": side,
                    "ltp": leg.get("lastPrice"),
                    "oi": leg.get("openInterest"),
                    "oi_change": leg.get("changeinOpenInterest"),
                    "volume": leg.get("totalTradedVolume"),
                    "iv": leg.get("impliedVolatility"),
                })
        if not flat:
            raise FeedError(
                f"No rows for expiry {want}. Available expiries: "
                f"{[e.isoformat() for e in expiries[:6]]}"
            )

        today = today_ist()
        chain = chain_from_records(
            symbol=symbol.upper(), spot=spot, expiry=expiry, records=flat,
            dte_trading=max(trading_days_between(today, expiry), 0),
            lot_size=_safe_lot(symbol), strike_step=_safe_step(symbol),
        )
        # NSE reports IV as 0 for illiquid strikes; treat those as missing so
        # the pricing layer recomputes them rather than trusting a zero.
        chain.data.loc[chain.data["ce_iv"] <= 0, "ce_iv"] = float("nan")
        chain.data.loc[chain.data["pe_iv"] <= 0, "pe_iv"] = float("nan")
        stamp = records.get("timestamp")
        chain.timestamp = stamp
        if stamp:
            chain.warnings.append(
                f"Chain snapshot timestamp from NSE: {stamp}. This is delayed data -- "
                f"do not use it as an execution quote."
            )
        return chain

    def expiries(self, symbol: str = "NIFTY") -> list[dt.date]:
        payload = self.option_chain_raw(symbol)
        out = []
        for raw in ((payload.get("records") or {}).get("expiryDates") or []):
            try:
                out.append(dt.datetime.strptime(raw, "%d-%b-%Y").date())  # noqa: DTZ007
            except ValueError:
                continue
        return sorted(out)

    # -- quotes and indices -------------------------------------------------
    def index_quote(self, symbol: str = "NIFTY 50") -> dict:
        payload = self._get_json("/api/allIndices")
        for row in payload.get("data", []):
            if row.get("index", "").upper() == symbol.upper():
                return {
                    "symbol": row.get("index"), "price": row.get("last"),
                    "change": row.get("variation"), "change_pct": row.get("percentChange"),
                    "open": row.get("open"), "high": row.get("high"), "low": row.get("low"),
                    "previous_close": row.get("previousClose"),
                    "year_high": row.get("yearHigh"), "year_low": row.get("yearLow"),
                    "pe": row.get("pe"), "pb": row.get("pb"),
                    "dividend_yield": row.get("dy"),
                    "as_of": dt.datetime.now(IST).isoformat(),
                }
        raise FeedError(f"index {symbol!r} not found in NSE allIndices response")

    def india_vix(self) -> float:
        return float(self.index_quote("INDIA VIX")["price"])

    def fno_ban_list(self) -> list[str]:
        """Securities in the F&O ban period -- no fresh positions allowed."""
        try:
            payload = self._get_json("/api/marketStatus")
            return payload.get("banList", []) or []
        except FeedError:
            return []


def _safe_lot(symbol: str) -> int:
    try:
        return lot_size(symbol)
    except (KeyError, ValueError):
        return 1


def _safe_step(symbol: str) -> int:
    try:
        return strike_step(symbol)
    except KeyError:
        return 50
