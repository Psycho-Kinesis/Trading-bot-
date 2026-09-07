"""Live scanning loop.

Polls a data feed on a schedule, runs the full analysis, journals every result
and notifies on changes.  It places no orders and never will -- see the note in
the package README on why an uncalibrated signal engine wired to live execution
is a fast way to lose money.

The loop is session-aware: it does nothing outside market hours, skips the
first fifteen minutes where spreads are widest and direction least reliable,
and stops at the close.
"""

from __future__ import annotations

import datetime as dt
import time
import warnings
from dataclasses import dataclass, field

from ..analysis import AnalysisConfig, analyze
from ..calendar_in import (is_market_open, is_trading_day, minutes_into_session,
                           session_close, session_phase)
from ..constants import IST
from .journal import Journal
from .notify import Notifier

__all__ = ["ScannerConfig", "LiveScanner"]


@dataclass
class ScannerConfig:
    symbols: tuple[str, ...] = ("NIFTY",)
    interval_seconds: int = 300
    timeframe: str = "15m"
    lookback: int = 400
    skip_first_minutes: int = 15
    stop_at_close: bool = True
    fetch_chain: bool = True
    journal_path: str = "quantsutra_journal.sqlite"
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    max_iterations: int | None = None      # for testing; None = run all session


class LiveScanner:
    def __init__(self, config: ScannerConfig | None = None,
                 feed=None, notifier: Notifier | None = None):
        from ..data import YahooFeed

        self.config = config or ScannerConfig()
        self.feed = feed or YahooFeed()
        self.notifier = notifier or Notifier()
        self.journal = Journal(self.config.journal_path)
        self.iterations = 0

    # -- one pass -----------------------------------------------------------
    def scan_once(self, now: dt.datetime | None = None) -> list[dict]:
        """Analyse every configured symbol once and return the results."""
        now = now or dt.datetime.now(IST)
        results = []
        for symbol in self.config.symbols:
            try:
                results.append(self._scan_symbol(symbol, now))
            except Exception as exc:      # one bad symbol must not stop the loop
                results.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
        self.iterations += 1
        return results

    def _scan_symbol(self, symbol: str, now: dt.datetime) -> dict:
        from ..data.base import resample_ohlcv

        bars = self.feed.history(symbol, self.config.timeframe, self.config.lookback)
        higher, higher_name = None, "1wk"
        if self.config.timeframe in ("5m", "15m", "30m"):
            higher, higher_name = resample_ohlcv(bars, "1h"), "1h"
        elif self.config.timeframe in ("1h", "60m"):
            higher, higher_name = resample_ohlcv(bars, "1D"), "1d"
        elif self.config.timeframe == "1d":
            higher, higher_name = resample_ohlcv(bars, "1W"), "1wk"

        chain, vix = None, None
        if self.config.fetch_chain:
            try:
                from ..data import NseFeed
                nse = NseFeed()
                chain = nse.option_chain(symbol)
                vix = nse.india_vix()
            except Exception:
                chain = None       # analysis degrades gracefully without it

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = analyze(symbol, bars, self.config.timeframe, higher_tf=higher,
                             higher_tf_name=higher_name, chain=chain, india_vix=vix,
                             now=now, config=self.config.analysis)

        signal_id = self.journal.log_signal(result.get("signal", {}), payload=result)
        result["journal_id"] = signal_id
        result["notification"] = self.notifier.send(result, now=now)
        return result

    # -- the loop -----------------------------------------------------------
    def run(self, now: dt.datetime | None = None) -> None:
        config = self.config
        while True:
            current = now or dt.datetime.now(IST)

            if not is_trading_day(current.date()):
                print(f"[{current:%H:%M}] Market closed today. Nothing to do.")
                return
            if config.stop_at_close and current > session_close(current.date()):
                print(f"[{current:%H:%M}] Session closed. Stopping.")
                return
            if not is_market_open(current):
                print(f"[{current:%H:%M}] Outside market hours "
                      f"({session_phase(current)}); waiting.")
                time.sleep(min(config.interval_seconds, 60))
                if now:
                    return
                continue

            elapsed = minutes_into_session(current)
            if elapsed < config.skip_first_minutes:
                print(f"[{current:%H:%M}] {elapsed} minutes into the session -- waiting "
                      f"out the opening auction drift, where spreads are widest.")
                time.sleep(min(config.interval_seconds, 60))
                if now:
                    return
                continue

            for result in self.scan_once(current):
                if "error" in result:
                    print(f"[{current:%H:%M}] {result['symbol']}: {result['error']}")
                    continue
                signal = result.get("signal", {})
                marker = "*" if signal.get("actionable") else " "
                print(f"[{current:%H:%M}] {marker} {signal.get('symbol'):10} "
                      f"{signal.get('action'):18} conf {signal.get('confidence'):5.1f}  "
                      f"{(result.get('recommendation') or {}).get('summary', '')[:60]}")

            if config.max_iterations and self.iterations >= config.max_iterations:
                return
            if now:
                return
            time.sleep(config.interval_seconds)

    def close(self) -> None:
        self.journal.close()
