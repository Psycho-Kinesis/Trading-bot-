"""Indian market calendar: sessions, holidays, and derivative expiry resolution.

Expiry maths is the single most error-prone part of an Indian F&O bot.  The
weekday an index expires on has changed repeatedly, weekly contracts were
withdrawn for most indices in Nov-2024, and any expiry that lands on a holiday
rolls *backwards* to the previous trading day.  All of that is handled here and
driven by ``config/instruments.yaml`` / ``config/holidays.yaml`` so the rules
used are the rules that applied on the date being asked about.
"""

from __future__ import annotations

import bisect
import datetime as dt
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .constants import IST, MARKET_CLOSE, MARKET_OPEN, PRE_OPEN_END, PRE_OPEN_START

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

__all__ = [
    "ExpiryContext",
    "days_to_expiry",
    "expiry_chain",
    "expiry_context",
    "has_weekly",
    "holiday_data_status",
    "holidays",
    "instrument_specs",
    "is_expiry_day",
    "is_market_open",
    "is_trading_day",
    "lot_size",
    "minutes_into_session",
    "monthly_expiry",
    "monthly_expiry_on_or_after",
    "next_expiry",
    "next_trading_day",
    "now_ist",
    "previous_trading_day",
    "session_close",
    "session_open",
    "session_phase",
    "spec",
    "strike_step",
    "today_ist",
    "trading_days",
    "trading_days_between",
    "weekly_expiry_on_or_after",
]


def now_ist() -> dt.datetime:
    """Current instant in IST.

    Always use this rather than ``datetime.now()``. A host running UTC -- which
    is every cloud VPS by default -- is 5h30m behind India, so between 00:00 and
    05:30 IST a naive ``date.today()`` returns *yesterday's* Indian date. That
    is not cosmetic: it made ``is_expiry_day`` report True the day after expiry,
    which drives the expiry veto, the gamma-risk classification and the
    risk manager's block on selling premium into expiry.
    """
    return dt.datetime.now(IST)


def today_ist() -> dt.date:
    """Today's date in India, regardless of the host's timezone."""
    return now_ist().date()


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():  # pragma: no cover - packaging safety net
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def instrument_specs() -> dict:
    return _load("instruments.yaml")


@lru_cache(maxsize=1)
def _holiday_file() -> dict:
    return _load("holidays.yaml")


@lru_cache(maxsize=1)
def holidays() -> dict[dt.date, str]:
    """All configured holidays as ``{date: name}``."""
    out: dict[dt.date, str] = {}
    for block in _holiday_file().values():
        for row in block.get("dates", []):
            out[dt.date.fromisoformat(row["date"])] = row["name"]
    return out


@lru_cache(maxsize=1)
def _covered_years() -> tuple[int, ...]:
    return tuple(sorted(int(y) for y in _holiday_file()))


def holiday_data_status(year: int) -> str:
    """``"confirmed"``, ``"provisional"`` or ``"missing"`` for a given year."""
    block = _holiday_file().get(year)
    if block is None:
        return "missing"
    return "provisional" if block.get("provisional") else "confirmed"


def _warn_if_uncovered(date: dt.date) -> None:
    status = holiday_data_status(date.year)
    if status == "missing":
        warnings.warn(
            f"No holiday list configured for {date.year}; only weekends will be "
            f"treated as non-trading days. Update config/holidays.yaml.",
            stacklevel=3,
        )
    elif status == "provisional":
        warnings.warn(
            f"Holiday list for {date.year} is provisional. Verify against the "
            f"exchange circular before trading expiry-sensitive strategies.",
            stacklevel=3,
        )


def is_weekend(date: dt.date) -> bool:
    return date.weekday() >= 5


def is_trading_day(date: dt.date, warn: bool = False) -> bool:
    if warn:
        _warn_if_uncovered(date)
    return not is_weekend(date) and date not in holidays()


def previous_trading_day(date: dt.date) -> dt.date:
    d = date - dt.timedelta(days=1)
    for _ in range(30):
        if is_trading_day(d):
            return d
        d -= dt.timedelta(days=1)
    raise RuntimeError(f"no trading day found in the 30 days before {date}")


def next_trading_day(date: dt.date) -> dt.date:
    d = date + dt.timedelta(days=1)
    for _ in range(30):
        if is_trading_day(d):
            return d
        d += dt.timedelta(days=1)
    raise RuntimeError(f"no trading day found in the 30 days after {date}")


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    days, d = [], start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def trading_days_between(start: dt.date, end: dt.date) -> int:
    """Trading days strictly after ``start`` up to and including ``end``.

    This is the correct denominator for option time-to-expiry in a market with
    ~15 holidays a year -- calendar days overstate theta decay windows.
    """
    if end <= start:
        return 0
    return len(trading_days(start + dt.timedelta(days=1), end))


# --------------------------------------------------------------------------
# Session helpers
# --------------------------------------------------------------------------

def _at(date: dt.date, hm: tuple[int, int]) -> dt.datetime:
    return dt.datetime(date.year, date.month, date.day, hm[0], hm[1], tzinfo=IST)


def session_open(date: dt.date) -> dt.datetime:
    return _at(date, MARKET_OPEN)


def session_close(date: dt.date) -> dt.datetime:
    return _at(date, MARKET_CLOSE)


def is_market_open(now: dt.datetime | None = None) -> bool:
    now = now.astimezone(IST) if now else now_ist()
    if not is_trading_day(now.date()):
        return False
    return session_open(now.date()) <= now <= session_close(now.date())


def minutes_into_session(now: dt.datetime | None = None) -> int:
    """Minutes elapsed since 09:15, clamped to the 375-minute session."""
    now = now.astimezone(IST) if now else now_ist()
    delta = (now - session_open(now.date())).total_seconds() / 60
    return int(max(0, min(375, delta)))


def session_phase(now: dt.datetime | None = None) -> str:
    """Named phase of the session -- several playbooks are phase-gated."""
    now = now.astimezone(IST) if now else now_ist()
    if not is_trading_day(now.date()):
        return "HOLIDAY"
    if _at(now.date(), PRE_OPEN_START) <= now < _at(now.date(), PRE_OPEN_END):
        return "PRE_OPEN"
    if now < session_open(now.date()):
        return "CLOSED"
    if now > session_close(now.date()):
        return "POST_CLOSE"
    mins = minutes_into_session(now)
    if mins <= 15:
        return "OPENING_AUCTION_DRIFT"   # 09:15-09:30, widest spreads
    if mins <= 60:
        return "MORNING_TREND"           # 09:30-10:15, where most ORB setups resolve
    if mins <= 195:
        return "MIDDAY"                  # 10:15-12:30, lowest volume, chop risk
    if mins <= 315:
        return "AFTERNOON_TREND"         # 12:30-14:30
    return "CLOSING_HOUR"                # 14:30-15:30, expiry-day gamma zone


# --------------------------------------------------------------------------
# Effective-dated spec lookup
# --------------------------------------------------------------------------

def _effective(rows: list[dict], on: dt.date, key: str):
    """Pick the value from an effective-dated rule list that applies on ``on``."""
    dates = [dt.date.fromisoformat(r["from"]) for r in rows]
    idx = bisect.bisect_right(dates, on) - 1
    if idx < 0:
        return None
    return rows[idx].get(key)


def spec(symbol: str) -> dict:
    specs = instrument_specs()["instruments"]
    key = symbol.upper()
    if key not in specs:
        raise KeyError(f"unknown instrument {symbol!r}; known: {sorted(specs)}")
    return specs[key]


def lot_size(symbol: str, on: dt.date | None = None) -> int:
    on = on or today_ist()
    value = _effective(spec(symbol)["lot_size"], on, "value")
    if value is None:
        raise ValueError(f"no lot size configured for {symbol} on {on}")
    return int(value)


def strike_step(symbol: str) -> int:
    return int(spec(symbol)["strike_step"])


def has_weekly(symbol: str, on: dt.date | None = None) -> bool:
    on = on or today_ist()
    return _effective(spec(symbol)["weekly_expiry"], on, "weekday") is not None


def _roll_back_for_holiday(date: dt.date) -> dt.date:
    """Exchange convention: an expiry landing on a holiday moves to the
    *previous* trading day, never forward."""
    while not is_trading_day(date):
        date -= dt.timedelta(days=1)
    return date


def weekly_expiry_on_or_after(symbol: str, date: dt.date) -> dt.date | None:
    """Next weekly expiry at or after ``date``; ``None`` if no weekly contract."""
    weekday = _effective(spec(symbol)["weekly_expiry"], date, "weekday")
    if weekday is None:
        return None
    _warn_if_uncovered(date)
    ahead = (weekday - date.weekday()) % 7
    candidate = date + dt.timedelta(days=ahead)
    rolled = _roll_back_for_holiday(candidate)
    if rolled < date:
        # This week's expiry already passed (rolled back before today) -- take
        # the following week's.
        candidate += dt.timedelta(days=7)
        rolled = _roll_back_for_holiday(candidate)
    return rolled


def monthly_expiry(symbol: str, year: int, month: int) -> dt.date:
    """The monthly (series) expiry for a given calendar month."""
    weekday = _effective(
        spec(symbol)["monthly_expiry"], dt.date(year, month, 15), "weekday"
    )
    if weekday is None:
        raise ValueError(f"no monthly expiry rule for {symbol} in {year}-{month:02d}")
    # Walk back from the last day of the month to the last matching weekday.
    if month == 12:
        last = dt.date(year, 12, 31)
    else:
        last = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    back = (last.weekday() - weekday) % 7
    return _roll_back_for_holiday(last - dt.timedelta(days=back))


def monthly_expiry_on_or_after(symbol: str, date: dt.date) -> dt.date:
    exp = monthly_expiry(symbol, date.year, date.month)
    if exp < date:
        y, m = (date.year + 1, 1) if date.month == 12 else (date.year, date.month + 1)
        exp = monthly_expiry(symbol, y, m)
    return exp


def next_expiry(symbol: str, date: dt.date | None = None) -> dt.date:
    """The nearest tradable expiry -- weekly if the index has one, else monthly."""
    date = date or today_ist()
    wk = weekly_expiry_on_or_after(symbol, date)
    mo = monthly_expiry_on_or_after(symbol, date)
    return min(wk, mo) if wk else mo


def expiry_chain(symbol: str, date: dt.date | None = None, count: int = 4) -> list[dt.date]:
    """The next ``count`` distinct expiries, ascending."""
    date = date or today_ist()
    out: list[dt.date] = []
    cursor = date
    while len(out) < count:
        exp = next_expiry(symbol, cursor)
        if exp not in out:
            out.append(exp)
        cursor = exp + dt.timedelta(days=1)
    return out


def is_expiry_day(symbol: str, date: dt.date | None = None) -> bool:
    date = date or today_ist()
    return is_trading_day(date) and next_expiry(symbol, date) == date


def days_to_expiry(symbol: str, date: dt.date | None = None, calendar: bool = False) -> int:
    """Days until the nearest expiry -- trading days by default."""
    date = date or today_ist()
    exp = next_expiry(symbol, date)
    if calendar:
        return (exp - date).days
    return trading_days_between(date, exp)


@dataclass(frozen=True)
class ExpiryContext:
    """Everything the options layer needs to know about where we are in the cycle."""

    symbol: str
    date: dt.date
    expiry: dt.date
    dte_trading: int
    dte_calendar: int
    is_expiry_day: bool
    is_monthly: bool
    lot_size: int
    strike_step: int

    @property
    def gamma_risk(self) -> str:
        """Qualitative pin/gamma risk -- drives whether short premium is sane."""
        if self.is_expiry_day:
            return "EXTREME"
        if self.dte_trading <= 1:
            return "HIGH"
        if self.dte_trading <= 3:
            return "ELEVATED"
        return "NORMAL"


def expiry_context(symbol: str, date: dt.date | None = None) -> ExpiryContext:
    date = date or today_ist()
    exp = next_expiry(symbol, date)
    return ExpiryContext(
        symbol=symbol.upper(),
        date=date,
        expiry=exp,
        dte_trading=trading_days_between(date, exp),
        dte_calendar=(exp - date).days,
        is_expiry_day=exp == date,
        is_monthly=exp == monthly_expiry_on_or_after(symbol, date),
        lot_size=lot_size(symbol, date),
        strike_step=strike_step(symbol),
    )
