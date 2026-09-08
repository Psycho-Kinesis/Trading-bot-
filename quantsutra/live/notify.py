"""Notification sinks.

Signals are only useful if they reach you, but a bot that pings on every bar
trains you to ignore it.  :class:`Notifier` therefore deduplicates: it will not
resend the same direction for the same symbol until the view actually changes
or a cooldown elapses.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, field

import requests

from ..calendar_in import now_ist

__all__ = ["ConsoleSink", "Notifier", "TelegramSink", "format_signal"]


def format_signal(result: dict, verbose: bool = False) -> str:
    """Plain-text rendering suitable for a phone screen."""
    signal = result.get("signal") or {}
    recommendation = result.get("recommendation") or {}
    lines = [
        f"{signal.get('symbol')} {signal.get('timeframe')} @ {signal.get('spot')}",
        f"{signal.get('action')}  (confidence {signal.get('confidence')})",
    ]
    if signal.get("entry"):
        lines.append(f"entry {signal['entry']:.0f} | stop {signal['stop_loss']:.0f} | "
                     f"T1 {signal['targets'][0]:.0f} | RR {signal.get('risk_reward')}")
    contract = result.get("contract") or {}
    if contract.get("strikes"):
        legs = ", ".join(f"{role} {strike:.0f}" for role, strike in contract["strikes"].items())
        lines.append(f"legs: {legs}")
    position = result.get("position") or {}
    if position.get("lots"):
        lines.append(f"size: {position['lots']} lot(s), risk "
                     f"Rs.{position.get('risk_amount', 0):,.0f}")
    if recommendation.get("summary"):
        lines.append(recommendation["summary"])
    if signal.get("vetoes"):
        lines.append("BLOCKED: " + signal["vetoes"][0])
    if verbose:
        lines.append("")
        lines.extend(signal.get("reasons", [])[:4])
    lines.append("")
    lines.append("Analysis only, not advice. Verify before acting.")
    return "\n".join(lines)


class ConsoleSink:
    name = "console"

    def send(self, text: str, title: str = "") -> bool:
        print(f"\n=== {title or 'signal'} ===\n{text}\n")
        return True


@dataclass
class TelegramSink:
    """Sends to a Telegram chat.

    Credentials come from ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` by
    default so they never end up in a config file or a repository.
    """

    token: str | None = None
    chat_id: str | None = None
    timeout: int = 10
    name: str = "telegram"

    def __post_init__(self):
        self.token = self.token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = self.chat_id or os.environ.get("TELEGRAM_CHAT_ID")

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str, title: str = "") -> bool:
        if not self.configured:
            return False
        body = f"*{title}*\n```\n{text}\n```" if title else text
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": body, "parse_mode": "Markdown"},
                timeout=self.timeout,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False


@dataclass
class Notifier:
    sinks: list = field(default_factory=lambda: [ConsoleSink()])
    cooldown_minutes: int = 45
    notify_no_trade: bool = False
    _last: dict = field(default_factory=dict)

    def should_send(self, symbol: str, action: str, now: dt.datetime | None = None) -> bool:
        """Suppress repeats of an unchanged view.

        A signal engine run every five minutes will produce the same answer
        most of the time. Sending it every time is how a useful alert becomes
        background noise you stop reading.
        """
        now = now or now_ist()
        if action == "NO_TRADE" and not self.notify_no_trade:
            return False
        previous = self._last.get(symbol)
        if previous is None:
            return True
        last_action, last_time = previous
        if last_action != action:
            return True
        return (now - last_time).total_seconds() / 60 >= self.cooldown_minutes

    def send(self, result: dict, verbose: bool = False,
             now: dt.datetime | None = None) -> dict:
        signal = result.get("signal") or {}
        symbol = signal.get("symbol", "?")
        action = signal.get("action", "NO_TRADE")
        if not self.should_send(symbol, action, now):
            return {"sent": False, "reason": "suppressed (unchanged view within cooldown)"}

        text = format_signal(result, verbose)
        title = f"{symbol} {action}"
        delivered = [sink.name for sink in self.sinks if sink.send(text, title)]
        self._last[symbol] = (action, now or now_ist())
        return {"sent": bool(delivered), "sinks": delivered}
