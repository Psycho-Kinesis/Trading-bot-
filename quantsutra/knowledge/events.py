"""Historical Indian market events and recurring event patterns.

Two distinct things live here.

**A curated event history** -- the crashes, rallies and policy shocks that
shaped the Indian market, with what actually happened and what the episode
teaches.  These are compiled from public record; magnitudes are approximate
and should be checked against price data before being cited as fact.

**Recurring calendar events** -- the Union Budget, RBI policy, monthly expiry,
quarterly results, US Fed decisions and general elections.  These are what
matter operationally: implied volatility rises into a scheduled event and
collapses immediately after it, so a long option bought the day before an
event can lose money on a correct directional call.  That is the single most
useful thing this module contributes to the signal engine.

Nothing here predicts anything.  The analogue finder reports *similarity*,
which is not the same as a forecast -- the sample of comparable episodes is
tiny and the base rate of "this time it is different" is high.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

__all__ = [
    "MARKET_EVENTS",
    "RECURRING_EVENTS",
    "MarketEvent",
    "event_risk",
    "events_between",
    "lessons_for_regime",
    "upcoming_events",
]


@dataclass(frozen=True)
class MarketEvent:
    date: dt.date
    name: str
    category: str            # CRASH | RALLY | POLICY | GLOBAL | SCAM | ELECTION | PANDEMIC
    approx_move_pct: float | None
    duration_days: int | None
    description: str
    lesson: str
    vix_peak: float | None = None

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(), "name": self.name, "category": self.category,
            "approx_move_pct": self.approx_move_pct, "duration_days": self.duration_days,
            "description": self.description, "lesson": self.lesson,
            "vix_peak": self.vix_peak,
        }


def _e(date, name, category, move, duration, description, lesson, vix=None):
    return MarketEvent(dt.date.fromisoformat(date), name, category, move, duration,
                       description, lesson, vix)


# Magnitudes are approximate and drawn from public record. Verify before citing.
MARKET_EVENTS: list[MarketEvent] = [
    _e("1992-04-23", "Harshad Mehta securities scam", "SCAM", -54.0, 365,
       "The Sensex had roughly tripled in the year to April 1992 on funds diverted "
       "from the inter-bank securities market. When the scam surfaced the index lost "
       "over half its value across the following year.",
       "A vertical rally with no earnings behind it is a positioning bubble. The "
       "unwind is always faster than the build, and 'it keeps going up' is not a thesis."),
    _e("2001-03-02", "Ketan Parekh scam / dot-com unwind", "SCAM", -35.0, 300,
       "Pay-order financed operator positions in a handful of technology names "
       "collapsed together with the global dot-com bust.",
       "Concentration risk is invisible until liquidity leaves. Narrow leadership -- a "
       "few names carrying the index -- is a warning, not a strength."),
    _e("2004-05-17", "General election result shock", "ELECTION", -15.5, 1,
       "The NDA's unexpected defeat triggered the largest single-day fall to that "
       "point; trading was halted twice as circuit breakers tripped.",
       "Election results are the highest-gap-risk event in the Indian calendar. "
       "Options priced for a 3% move do not protect against a 15% one, and a stop "
       "order does not execute at its trigger price through a halt."),
    _e("2006-05-18", "Global liquidity scare", "CRASH", -11.0, 30,
       "A sharp global risk-off move hit an over-leveraged Indian market.",
       "Leverage built in a calm tape gets liquidated in a fast one, regardless of "
       "domestic fundamentals."),
    _e("2008-01-21", "Global financial crisis begins", "GLOBAL", -60.0, 380,
       "From the January 2008 peak the Nifty fell roughly 60% into October 2008 as "
       "global credit markets seized.",
       "Bear markets fall in stages with violent counter-rallies between them. Each "
       "bounce feels like the bottom. Position sizing, not entry timing, is what "
       "determines whether you are still solvent at the actual bottom."),
    _e("2009-05-18", "Post-election limit-up", "ELECTION", 17.3, 1,
       "The UPA's return with a workable majority triggered a 17% single-day gain; "
       "the upper circuit halted trade for the session.",
       "Gap risk is symmetric. A short option position can be destroyed by good news "
       "just as fast as bad, and an upper circuit means you cannot exit either."),
    _e("2013-08-28", "Taper tantrum / rupee crisis", "GLOBAL", -12.0, 90,
       "The rupee fell past 68/USD as the Fed signalled tapering; rate-sensitive "
       "sectors led the decline.",
       "Indian equities are a leveraged bet on global dollar liquidity. Watch the "
       "rupee and US yields, not only domestic data."),
    _e("2016-11-08", "Demonetisation", "POLICY", -6.0, 60,
       "The withdrawal of Rs.500 and Rs.1000 notes was announced without warning "
       "after market hours; consumption and financial names fell hard.",
       "Unscheduled policy risk cannot be hedged in advance because it is not on any "
       "calendar. It is an argument for permanent position limits, not for prediction."),
    _e("2018-02-01", "LTCG tax reintroduced", "POLICY", -6.0, 40,
       "The Union Budget reintroduced long-term capital gains tax on equities.",
       "Budget day carries genuine single-day risk. IV is bid into it and collapses "
       "the moment the speech ends, which is why long options bought on budget "
       "morning often lose even when the direction is right."),
    _e("2020-03-23", "COVID-19 crash", "PANDEMIC", -38.0, 33,
       "The Nifty fell roughly 38% in about five weeks; India VIX printed above 80, "
       "an all-time high, and multiple lower circuits were hit.",
       "Volatility regimes shift in days, not months. Position sizes set for a "
       "12-VIX market are catastrophic at 80 VIX -- size must be a function of "
       "current volatility, not of a long-run average.", 83.6),
    _e("2020-03-24", "Post-COVID recovery begins", "RALLY", 150.0, 550,
       "From the March 2020 low the index roughly two-and-a-half times over the "
       "following eighteen months on unprecedented global liquidity.",
       "The most violent rallies begin at maximum pessimism, when nobody wants to "
       "buy. Being out of the market to 'wait for clarity' is itself a large bet."),
    _e("2021-10-19", "Post-liquidity peak", "RALLY", 0.0, 1,
       "The market topped after the liquidity-driven rally, then traded sideways to "
       "lower for roughly a year as global policy tightened.",
       "A top is a process, not a day. Structure -- lower highs and lower lows -- "
       "identifies it far earlier than any oscillator."),
    _e("2024-06-04", "General election result", "ELECTION", -5.9, 1,
       "Exit polls on 1 June pointed to a large majority and the index gapped up "
       "sharply on 3 June; the actual result on 4 June fell well short and the index "
       "fell about 6% intraday, giving back the entire move and more.",
       "Consensus positioning is the risk, not the event. When everyone is on one "
       "side, the surprise only has to be small to be violent -- and it moved both "
       "ways within two sessions."),
    _e("2024-08-05", "Yen carry-trade unwind", "GLOBAL", -3.0, 3,
       "A global unwind of yen-funded carry positions hit Asian equities; India "
       "gapped down with the region before recovering within days.",
       "Global funding shocks transmit to India overnight, through the gap, where "
       "no intraday stop can protect you."),
    _e("2024-10-01", "FII outflow correction", "CRASH", -11.0, 60,
       "Sustained foreign selling drove a roughly 11% correction from the September "
       "2024 peak, with domestic inflows absorbing much of it.",
       "Domestic institutional buying can cushion a decline but rarely reverses it. "
       "Watch the FII/DII net figures as a flow indicator, not as a timing signal."),
]

# Recurring events with an IV signature.  ``iv_impact`` is the qualitative
# effect on implied volatility going in; the crush afterwards is the mirror.
RECURRING_EVENTS: list[dict] = [
    {"name": "Union Budget", "when": "1 February (annual)", "iv_impact": "HIGH",
     "typical_move_pct": 1.5,
     "note": "IV rises for a week beforehand and collapses within minutes of the "
             "speech ending. Buying options on budget morning is paying peak premium "
             "for the crush."},
    {"name": "RBI Monetary Policy", "when": "roughly every two months", "iv_impact": "MEDIUM",
     "typical_move_pct": 0.8,
     "note": "Banking names move far more than the index; BANKNIFTY IV rises more "
             "than NIFTY IV into the decision."},
    {"name": "Monthly F&O expiry", "when": "last Tuesday (NSE) / Thursday (BSE)",
     "iv_impact": "MEDIUM", "typical_move_pct": 0.7,
     "note": "Rollover flows and pinning distort the last hour. Direction on expiry "
             "afternoon is often positioning, not information."},
    {"name": "Weekly expiry", "when": "Tuesday (NIFTY) / Thursday (SENSEX)",
     "iv_impact": "LOW", "typical_move_pct": 0.5,
     "note": "Theta dominates from about 13:30. Long options held into the close "
             "usually go to zero even when the direction was right."},
    {"name": "Quarterly results season", "when": "Jan, Apr, Jul, Oct", "iv_impact": "MEDIUM",
     "typical_move_pct": 1.0,
     "note": "Index IV rises as heavyweight results cluster; single-stock IV rises "
             "far more."},
    {"name": "US Federal Reserve FOMC", "when": "8 times a year", "iv_impact": "MEDIUM",
     "typical_move_pct": 0.9,
     "note": "The decision lands after Indian hours, so the reaction arrives as an "
             "overnight gap. Intraday stops offer no protection."},
    {"name": "US CPI release", "when": "monthly", "iv_impact": "MEDIUM",
     "typical_move_pct": 0.7,
     "note": "Also after Indian hours. Same gap-risk logic as FOMC."},
    {"name": "General election results", "when": "every 5 years", "iv_impact": "EXTREME",
     "typical_move_pct": 6.0,
     "note": "The highest single-day risk in the Indian calendar. Both 2004 and 2024 "
             "produced moves several times what options were pricing."},
    {"name": "State election results", "when": "varies", "iv_impact": "MEDIUM",
     "typical_move_pct": 1.2,
     "note": "Treated as a read-through to national politics; impact varies with how "
             "surprising the result is."},
    {"name": "Muhurat trading", "when": "Diwali (one hour)", "iv_impact": "LOW",
     "typical_move_pct": 0.3,
     "note": "Ceremonial session with thin volume. Not a session to trade size in."},
]


def events_between(start: dt.date, end: dt.date) -> list[MarketEvent]:
    return [e for e in MARKET_EVENTS if start <= e.date <= end]


def upcoming_events(on: dt.date | None = None, horizon_days: int = 30) -> list[dict]:
    """Scheduled recurring events in the near window.

    Dates for movable events (RBI policy, results) are approximate; this
    flags the *category* of risk, not an exact calendar.
    """
    from ..calendar_in import today_ist

    on = on or today_ist()
    out: list[dict] = []

    budget = dt.date(on.year, 2, 1)
    if budget < on:
        budget = dt.date(on.year + 1, 2, 1)
    if (budget - on).days <= horizon_days:
        out.append({"event": "Union Budget", "date": budget.isoformat(),
                    "days_away": (budget - on).days, "iv_impact": "HIGH",
                    "note": "IV inflates into the speech and crushes immediately after."})

    # Results seasons cluster in the month after each quarter end.
    for month in (1, 4, 7, 10):
        season = dt.date(on.year, month, 10)
        if season < on:
            season = dt.date(on.year + 1, month, 10)
        if 0 <= (season - on).days <= horizon_days:
            out.append({"event": "Quarterly results season", "date": season.isoformat(),
                        "days_away": (season - on).days, "iv_impact": "MEDIUM",
                        "note": "Heavyweight results cluster; index IV drifts up."})
            break

    return sorted(out, key=lambda x: x["days_away"])


def event_risk(on: dt.date | None = None, symbol: str = "NIFTY") -> dict:
    """Event-driven risk assessment for a given date.

    Used by the signal engine to warn before an option is bought into a known
    IV crush.
    """
    from ..calendar_in import expiry_context, is_trading_day, today_ist

    on = on or today_ist()

    flags: list[str] = []
    level = "NORMAL"

    if not is_trading_day(on):
        return {"level": "MARKET_CLOSED", "flags": ["Not a trading day."],
                "upcoming": upcoming_events(on)}

    try:
        exp = expiry_context(symbol, on)
        if exp.is_expiry_day:
            level = "HIGH"
            flags.append(
                "Expiry day. Premium decays to intrinsic through the session and gamma "
                "is extreme in the last hour -- long options usually expire worthless "
                "and short options can lose far more than the credit collected."
            )
        elif exp.dte_trading <= 1:
            level = "ELEVATED"
            flags.append(f"{exp.dte_trading} trading day to expiry: theta is at its "
                         f"steepest and the option needs the move immediately.")
        if exp.is_monthly and exp.dte_trading <= 2:
            flags.append("Monthly expiry approaching: rollover flows distort the last "
                         "two sessions.")
    except (KeyError, ValueError):
        pass

    if on.month == 2 and on.day <= 3:
        level = "HIGH"
        flags.append("Union Budget window. IV is elevated going in and collapses after "
                     "the speech -- a long option can lose on a correct call.")

    for event in upcoming_events(on, horizon_days=7):
        flags.append(f"{event['event']} in {event['days_away']} day(s) "
                     f"({event['iv_impact']} IV impact): {event['note']}")
        if event["iv_impact"] in ("HIGH", "EXTREME") and level == "NORMAL":
            level = "ELEVATED"

    if not flags:
        flags.append("No scheduled event risk identified for this date. Unscheduled "
                     "risk -- policy announcements, global shocks -- is never zero.")

    return {"level": level, "flags": flags, "upcoming": upcoming_events(on, 45)}


def lessons_for_regime(regime: str, volatility_state: str = "NORMAL") -> list[str]:
    """Historical lessons relevant to the current regime.

    Deliberately blunt: these are the failure modes that repeat.
    """
    out: list[str] = []
    regime = (regime or "").upper()
    vol = (volatility_state or "").upper()

    if "STRONG_UPTREND" in regime:
        out.append("2021 and 2007 both ended with narrow leadership and vertical "
                   "price action. Strong trends are for participating in, but the "
                   "later the stage, the smaller the position should be.")
    if "DOWNTREND" in regime:
        out.append("2008 fell in stages with sharp counter-rallies between them. "
                   "Each bounce in a downtrend looks like the bottom; most are not.")
    if "RANGE" in regime:
        out.append("Ranges pay option sellers and punish buyers. Most of a range's "
                   "life is spent going nowhere -- the cost is theta, and it is "
                   "charged daily.")
    if "CHOP" in regime:
        out.append("High volatility without direction is the worst environment for "
                   "both trend-following and premium selling. Standing aside is a "
                   "position.")
    if vol == "EXTREME":
        out.append("March 2020 saw India VIX above 80. Position sizes calibrated to a "
                   "12-VIX market are catastrophic there -- size must scale with "
                   "current volatility, not a long-run average.")
    if vol == "LOW":
        out.append("Low volatility compresses option premiums and encourages "
                   "over-sizing to 'make it worth it'. That is how a quiet market "
                   "sets up a large loss when volatility returns.")
    return out
