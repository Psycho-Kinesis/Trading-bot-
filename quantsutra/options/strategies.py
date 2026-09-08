"""Option strategy construction and payoff analysis.

A directional view maps to several different structures, and which one is
right depends on implied volatility and time to expiry, not on the view alone.
The two mistakes this module exists to prevent:

* Buying naked weekly options with 1-2 days to expiry when IV is high.  Theta
  and the post-event IV crush can beat you even when the direction is right --
  this is the single most common way Indian retail option buyers lose.

* Selling naked options into a trend, or on expiry day, where gamma is
  effectively unbounded.  Every short-premium template here is defined with
  wings (defined risk) unless the caller explicitly overrides it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import Action
from .pricing import DEFAULT_DIVIDEND_YIELD, DEFAULT_RATE, greeks, intrinsic_value

__all__ = [
    "STRUCTURE_NOTES",
    "Leg",
    "Strategy",
    "build_strategy",
    "choose_structure",
    "payoff_curve",
    "strategy_metrics",
]


@dataclass
class Leg:
    option_type: str        # "CE" | "PE"
    strike: float
    quantity: int           # +1 long, -1 short (in lots)
    premium: float
    iv: float | None = None

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    def payoff_at(self, spot: float) -> float:
        """Per-unit payoff at expiry, net of the premium paid/received."""
        value = intrinsic_value(spot, self.strike, self.option_type)
        return self.quantity * (value - self.premium)

    def to_dict(self) -> dict:
        return {
            "action": "BUY" if self.is_long else "SELL",
            "option_type": self.option_type, "strike": round(self.strike, 1),
            "lots": abs(self.quantity), "premium": round(self.premium, 2),
            "iv": round(self.iv, 2) if self.iv is not None and np.isfinite(self.iv) else None,
        }


@dataclass
class Strategy:
    name: str
    action: Action
    legs: list[Leg]
    lot_size: int
    spot: float
    expiry: object = None
    notes: list[str] = field(default_factory=list)

    @property
    def net_premium(self) -> float:
        """Per-unit net premium.  Negative = net debit (you pay)."""
        return -sum(leg.quantity * leg.premium for leg in self.legs)

    @property
    def is_debit(self) -> bool:
        return self.net_premium < 0

    @property
    def cost(self) -> float:
        """Cash outlay (debit) or credit received, in rupees, for the position."""
        return self.net_premium * self.lot_size

    def payoff_at(self, spot: float) -> float:
        """Total rupee P&L at expiry for the whole position."""
        return sum(leg.payoff_at(spot) for leg in self.legs) * self.lot_size

    def net_greeks(self, t: float, vol_override: float | None = None,
                   rate: float = DEFAULT_RATE, q: float = DEFAULT_DIVIDEND_YIELD) -> dict:
        """Position greeks, scaled by lot size.

        Net delta in *index points* is the number that matters for risk: a net
        delta of +150 on NIFTY means the position gains ~Rs.150 per lot per
        index point.
        """
        totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
        for leg in self.legs:
            vol = vol_override if vol_override is not None else (leg.iv or 15.0) / 100
            g = greeks(self.spot, leg.strike, t, vol, leg.option_type, rate, q)
            totals["delta"] += leg.quantity * g.delta
            totals["gamma"] += leg.quantity * g.gamma
            totals["theta"] += leg.quantity * g.theta
            totals["vega"] += leg.quantity * g.vega
        return {
            "delta": round(totals["delta"] * self.lot_size, 2),
            "gamma": round(totals["gamma"] * self.lot_size, 4),
            "theta_per_day": round(totals["theta"] * self.lot_size, 2),
            "vega_per_vol_pt": round(totals["vega"] * self.lot_size, 2),
            "delta_per_lot": round(totals["delta"], 4),
        }

    def to_dict(self) -> dict:
        m = strategy_metrics(self)
        return {
            "name": self.name, "action": self.action.value,
            "legs": [leg.to_dict() for leg in self.legs],
            "lot_size": self.lot_size,
            "net_premium_per_unit": round(self.net_premium, 2),
            "cash_flow": round(self.cost, 2),
            "type": "DEBIT" if self.is_debit else "CREDIT",
            **m, "notes": self.notes,
        }


def payoff_curve(strategy: Strategy, low: float | None = None, high: float | None = None,
                 points: int = 201) -> tuple[np.ndarray, np.ndarray]:
    """Expiry payoff over a spot range (default +/-8% around spot)."""
    low = low if low is not None else strategy.spot * 0.92
    high = high if high is not None else strategy.spot * 1.08
    xs = np.linspace(low, high, points)
    ys = np.array([strategy.payoff_at(float(x)) for x in xs])
    return xs, ys


def strategy_metrics(strategy: Strategy) -> dict:
    """Max profit, max loss, breakevens and risk/reward at expiry.

    Unbounded legs are reported as ``None`` rather than as a large number, so a
    naked short cannot be mistaken for a capped-risk position by anything
    downstream.
    """
    xs, ys = payoff_curve(strategy, strategy.spot * 0.75, strategy.spot * 1.25, 1001)

    # Detect unbounded tails by checking the slope at each end of the range.
    left_slope = ys[1] - ys[0]
    right_slope = ys[-1] - ys[-2]
    unbounded_loss = (left_slope > 1e-6 and ys[0] < 0) or (right_slope < -1e-6 and ys[-1] < 0)
    unbounded_profit = (left_slope < -1e-6 and ys[0] > 0) or (right_slope > 1e-6 and ys[-1] > 0)

    max_profit = None if unbounded_profit else float(np.max(ys))
    max_loss = None if unbounded_loss else float(np.min(ys))

    breakevens = []
    for i in range(len(xs) - 1):
        if ys[i] == 0:
            breakevens.append(float(xs[i]))
        elif ys[i] * ys[i + 1] < 0:
            # linear interpolation between the bracketing points
            frac = abs(ys[i]) / (abs(ys[i]) + abs(ys[i + 1]))
            breakevens.append(float(xs[i] + frac * (xs[i + 1] - xs[i])))

    rr = None
    if max_profit is not None and max_loss is not None and max_loss < 0:
        rr = round(max_profit / abs(max_loss), 2)

    return {
        "max_profit": round(max_profit, 2) if max_profit is not None else "UNLIMITED",
        "max_loss": round(max_loss, 2) if max_loss is not None else "UNLIMITED",
        "breakevens": [round(b, 2) for b in breakevens],
        "reward_risk": rr,
        "risk_defined": max_loss is not None,
    }


def build_strategy(
    action: Action, spot: float, strikes: dict[str, float], premiums: dict[str, float],
    lot_size: int, ivs: dict[str, float] | None = None, expiry=None, lots: int = 1,
) -> Strategy:
    """Assemble a named structure from strikes and premiums.

    ``strikes``/``premiums``/``ivs`` are keyed by role: ``long_ce``, ``short_ce``,
    ``long_pe``, ``short_pe``.
    """
    ivs = ivs or {}
    legs: list[Leg] = []
    notes: list[str] = []

    def add(role: str, option_type: str, qty: int) -> None:
        if role in strikes and role in premiums:
            legs.append(Leg(option_type, float(strikes[role]), qty * lots,
                            float(premiums[role]), ivs.get(role)))

    if action == Action.BUY_CALL:
        add("long_ce", "CE", 1)
        notes.append("Naked long call: max loss is the premium, but theta works "
                     "against you every day and an IV drop can lose money even if "
                     "the direction is right.")
    elif action == Action.BUY_PUT:
        add("long_pe", "PE", 1)
        notes.append("Naked long put: same theta/IV caveats as a long call.")
    elif action == Action.SELL_CALL:
        add("short_ce", "CE", -1)
        notes.append("NAKED SHORT CALL -- loss is theoretically unlimited and margin "
                     "is high. Prefer a bear call spread unless you have a specific "
                     "reason not to.")
    elif action == Action.SELL_PUT:
        add("short_pe", "PE", -1)
        notes.append("NAKED SHORT PUT -- loss runs to the strike. Prefer a bull put "
                     "spread for defined risk.")
    elif action == Action.BULL_CALL_SPREAD:
        add("long_ce", "CE", 1)
        add("short_ce", "CE", -1)
        notes.append("Debit spread: cheaper than a naked call and far less "
                     "IV-sensitive, at the cost of a capped upside.")
    elif action == Action.BEAR_PUT_SPREAD:
        add("long_pe", "PE", 1)
        add("short_pe", "PE", -1)
        notes.append("Debit spread: defined risk, capped reward, reduced theta bleed.")
    elif action == Action.BULL_PUT_SPREAD:
        add("short_pe", "PE", -1)
        add("long_pe", "PE", 1)
        notes.append("Credit spread: profits from time decay and from price staying "
                     "above the short strike. Defined risk.")
    elif action == Action.BEAR_CALL_SPREAD:
        add("short_ce", "CE", -1)
        add("long_ce", "CE", 1)
        notes.append("Credit spread: profits from time decay and from price staying "
                     "below the short strike. Defined risk.")
    elif action == Action.IRON_CONDOR:
        add("short_pe", "PE", -1)
        add("long_pe", "PE", 1)
        add("short_ce", "CE", -1)
        add("long_ce", "CE", 1)
        notes.append("Range-bound credit structure. Needs an actual range: it is the "
                     "worst possible position in a trending or gapping market.")
    elif action == Action.LONG_STRADDLE:
        add("long_ce", "CE", 1)
        add("long_pe", "PE", 1)
        notes.append("Long volatility: profits from a large move either way. Needs "
                     "the move to exceed the combined premium, so only worth it when "
                     "IV is low relative to the expected move.")
    elif action == Action.SHORT_STRADDLE:
        add("short_ce", "CE", -1)
        add("short_pe", "PE", -1)
        notes.append("SHORT STRADDLE -- unlimited risk on both sides. Only with strict "
                     "stops, never on expiry day, never through an event.")
    else:
        raise ValueError(f"no structure defined for {action}")

    if not legs:
        raise ValueError(f"{action.value} is missing required strikes/premiums: "
                         f"got {sorted(strikes)}")

    name = action.value.replace("_", " ").title()
    return Strategy(name=name, action=action, legs=legs, lot_size=lot_size,
                    spot=spot, expiry=expiry, notes=notes)


STRUCTURE_NOTES = {
    "high_iv_directional": "IV is rich: a debit spread or a credit spread in the "
                           "direction of the view beats a naked long option, which "
                           "would bleed on an IV mean-reversion.",
    "low_iv_directional": "IV is cheap: a naked long option gives the cleanest "
                          "exposure to the move and the IV downside is limited.",
    "expiry_day": "Expiry day: gamma dominates and premium decays intraday. Buy only "
                  "with a tight time stop; do not sell naked.",
}


def choose_structure(
    direction: int, iv_percentile: float | None, dte_trading: int,
    trend_strength: float, risk_defined_only: bool = True,
) -> tuple[Action, list[str]]:
    """Map a directional view plus the volatility/time context to a structure.

    This is where the "should I just buy a call?" question actually gets
    answered.  ``direction`` is +1 / -1 / 0.
    """
    notes: list[str] = []
    iv_pct = iv_percentile if iv_percentile is not None and np.isfinite(iv_percentile) else 50.0
    iv_rich = iv_pct >= 65
    iv_cheap = iv_pct <= 35

    if direction == 0:
        if iv_rich and dte_trading >= 2:
            notes.append("No directional edge but IV is rich and there is time: a "
                         "defined-risk range structure is the only sensible expression.")
            return Action.IRON_CONDOR, notes
        if iv_cheap and dte_trading >= 2:
            notes.append("No direction but IV is cheap: a long straddle pays if the "
                         "range expands. Size it small -- it loses money in a quiet tape.")
            return Action.LONG_STRADDLE, notes
        notes.append("No directional edge and no volatility edge. Standing aside is "
                     "the correct trade.")
        return Action.NO_TRADE, notes

    bullish = direction > 0

    if dte_trading <= 0:
        notes.append(STRUCTURE_NOTES["expiry_day"])
        notes.append("Expiry-day buying is a scalp, not a position: use a hard time "
                     "stop and expect the premium to go to zero if you are wrong.")
        return (Action.BUY_CALL if bullish else Action.BUY_PUT), notes

    if iv_rich:
        notes.append(STRUCTURE_NOTES["high_iv_directional"])
        if trend_strength >= 0.6:
            # Strong trend + rich IV: a debit spread keeps directional exposure
            # while selling some of the expensive premium back.
            return (Action.BULL_CALL_SPREAD if bullish else Action.BEAR_PUT_SPREAD), notes
        notes.append("Trend is not strong enough to pay for a debit: sell the "
                     "opposing side instead and let time work for you.")
        return (Action.BULL_PUT_SPREAD if bullish else Action.BEAR_CALL_SPREAD), notes

    if iv_cheap:
        notes.append(STRUCTURE_NOTES["low_iv_directional"])
        return (Action.BUY_CALL if bullish else Action.BUY_PUT), notes

    # Middling IV -- let trend strength and time decide.
    if trend_strength >= 0.65 and dte_trading >= 2:
        notes.append("Strong trend with average IV: a naked long option gives the "
                     "most convexity if the move continues.")
        return (Action.BUY_CALL if bullish else Action.BUY_PUT), notes
    notes.append("Average IV and a moderate trend: a debit spread caps the reward but "
                 "cuts the theta bleed roughly in half.")
    return (Action.BULL_CALL_SPREAD if bullish else Action.BEAR_PUT_SPREAD), notes
