"""Strike and expiry selection.

Turning "I am bullish on NIFTY" into "buy 1 lot of the 25100 CE expiring
Tuesday at ~Rs.160" is where most of the practical damage is done.  The rules
encoded here:

* **Liquidity first.** A strike with no open interest cannot be exited at a
  fair price.  Strikes are filtered on OI and, where available, bid-ask spread
  before anything else is considered.

* **Delta, not distance.** Strikes are chosen by target delta so the same
  instruction behaves consistently across expiries and volatility levels.

* **Time buffer.** Buying an option with 0-1 days left is a scalp; the selector
  will roll to the next expiry when the intended holding period does not fit
  inside the remaining time, and says so.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..calendar_in import expiry_chain, expiry_context
from ..constants import Action
from .chain import OptionChain
from .pricing import delta_to_strike, greeks, moneyness, time_to_expiry

__all__ = ["StrikeChoice", "select_strike", "select_expiry", "recommend_contract",
           "MIN_OI_DEFAULT"]

# A strike below this OI is usually not worth trading on a weekly index option;
# tune per instrument -- BANKNIFTY monthlies and SENSEX are thinner than NIFTY.
MIN_OI_DEFAULT = 25_000


@dataclass
class StrikeChoice:
    strike: float
    option_type: str
    premium: float
    delta: float
    iv: float | None
    oi: int | None
    moneyness: str
    liquidity_ok: bool
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "strike": self.strike, "option_type": self.option_type,
            "premium": round(self.premium, 2) if np.isfinite(self.premium) else None,
            "delta": round(self.delta, 3), "iv": self.iv, "oi": self.oi,
            "moneyness": self.moneyness, "liquidity_ok": self.liquidity_ok,
            "reasons": self.reasons,
        }


def select_expiry(
    symbol: str, on: dt.date, intended_hold_days: int = 1, min_dte: int = 1
) -> dict:
    """Pick the expiry to trade, rolling forward when time is too short.

    The rule of thumb encoded here: you want at least ``intended_hold_days + 1``
    trading days of life left, so that being right slowly does not still lose.
    """
    chain = expiry_chain(symbol, on, count=4)
    ctx = expiry_context(symbol, on)
    need = max(min_dte, intended_hold_days + 1)

    from ..calendar_in import trading_days_between

    chosen, rolled, reason = chain[0], False, ""
    for exp in chain:
        dte = trading_days_between(on, exp)
        if dte >= need:
            chosen = exp
            rolled = exp != chain[0]
            break
    else:
        chosen = chain[-1]
        rolled = True

    dte = trading_days_between(on, chosen)
    if rolled:
        reason = (f"Nearest expiry has only {ctx.dte_trading} trading day(s) left, "
                  f"which is too short for a {intended_hold_days}-day hold. Rolled to "
                  f"{chosen} ({dte} trading days) to avoid paying pure gamma.")
    elif ctx.is_expiry_day:
        reason = ("Today IS expiry: premium decays to intrinsic by 15:30 and gamma is "
                  "extreme. Only intraday scalps with a hard time stop belong here.")
    else:
        reason = f"Nearest expiry {chosen} has {dte} trading day(s) -- adequate for the hold."

    return {
        "expiry": chosen, "dte_trading": dte,
        "dte_calendar": (chosen - on).days,
        "is_expiry_day": chosen == on, "rolled_forward": rolled,
        "gamma_risk": ctx.gamma_risk, "reason": reason,
        "alternatives": [{"expiry": e, "dte": trading_days_between(on, e)} for e in chain],
    }


def select_strike(
    chain: OptionChain, option_type: str, target_delta: float = 0.40,
    min_oi: int = MIN_OI_DEFAULT, max_spread_pct: float = 0.05,
    prefer_atm_on_expiry: bool = True,
) -> StrikeChoice | None:
    """Pick the best strike of a given type from a live chain.

    ``target_delta`` conventions:

    * 0.55-0.70 -- deep-ish ITM, behaves most like the index, least theta decay
      per rupee.  Best when you want directional exposure and are willing to
      pay for it.
    * 0.40-0.50 -- near ATM. The default: the best balance of premium, gamma
      and liquidity.
    * 0.20-0.30 -- OTM. Cheap and convex, but the probability of expiring
      worthless is high and this is where most retail premium is lost.
    """
    option_type = option_type.upper()
    prefix = "ce" if option_type == "CE" else "pe"
    d = chain.data
    if d.empty:
        return None

    t = chain.t
    if prefer_atm_on_expiry and chain.dte_trading <= 0:
        target_delta = max(target_delta, 0.50)

    candidates = []
    for strike, row in d.iterrows():
        strike = float(strike)
        ltp = row.get(f"{prefix}_ltp")
        oi = row.get(f"{prefix}_oi")
        iv = row.get(f"{prefix}_iv")
        if ltp is None or not np.isfinite(ltp) or ltp <= 0:
            continue
        vol = float(iv) / 100 if iv is not None and np.isfinite(iv) and iv > 0 else 0.15
        g = greeks(chain.spot, strike, t, vol, option_type, chain.rate, chain.dividend_yield)
        oi_val = int(oi) if oi is not None and np.isfinite(oi) else None
        candidates.append({
            "strike": strike, "ltp": float(ltp), "delta": g.delta, "iv": iv,
            "oi": oi_val, "liquid": oi_val is None or oi_val >= min_oi,
        })

    if not candidates:
        return None

    liquid = [c for c in candidates if c["liquid"]]
    pool = liquid or candidates
    reasons: list[str] = []
    if not liquid:
        reasons.append(f"No strike met the {min_oi:,} OI floor -- the whole chain looks "
                       f"illiquid. Reduce size or trade a different expiry.")

    best = min(pool, key=lambda c: abs(abs(c["delta"]) - target_delta))
    reasons.append(f"Chose {best['strike']:.0f} {option_type} for delta "
                   f"{abs(best['delta']):.2f} against a {target_delta:.2f} target.")

    if best["oi"] is not None:
        reasons.append(f"Open interest {best['oi']:,} "
                       f"({'adequate' if best['liquid'] else 'THIN -- exits may slip'}).")

    prem_pct = 100 * best["ltp"] / chain.spot
    if prem_pct > 1.5:
        reasons.append(f"Premium is {prem_pct:.2f}% of spot -- expensive; the index must "
                       f"move materially just to break even.")
    if chain.dte_trading <= 1 and abs(best["delta"]) < 0.35:
        reasons.append("OTM option with <=1 day to expiry: the probability of it "
                       "expiring worthless is high. This is a lottery ticket, size it as one.")

    return StrikeChoice(
        strike=best["strike"], option_type=option_type, premium=best["ltp"],
        delta=float(best["delta"]),
        iv=round(float(best["iv"]), 2) if best["iv"] is not None and np.isfinite(best["iv"]) else None,
        oi=best["oi"], moneyness=moneyness(chain.spot, best["strike"], option_type),
        liquidity_ok=bool(best["liquid"]), reasons=reasons,
    )


def recommend_contract(
    action: Action, chain: OptionChain, target_delta: float = 0.40,
    spread_width_strikes: int = 4, min_oi: int = MIN_OI_DEFAULT,
) -> dict:
    """Resolve an :class:`Action` into concrete strikes and premiums.

    Returns the ``strikes``/``premiums``/``ivs`` dicts that
    :func:`quantsutra.options.strategies.build_strategy` expects, plus the
    reasoning behind each leg.
    """
    step = chain.strike_step
    width = spread_width_strikes * step
    strikes: dict[str, float] = {}
    premiums: dict[str, float] = {}
    ivs: dict[str, float] = {}
    reasons: list[str] = []
    missing: list[str] = []

    def leg_from(role: str, option_type: str, choice: StrikeChoice | None) -> None:
        if choice is None:
            missing.append(role)
            return
        strikes[role] = choice.strike
        premiums[role] = choice.premium
        if choice.iv is not None:
            ivs[role] = choice.iv
        reasons.extend(choice.reasons)

    def at_strike(strike: float, option_type: str, role: str) -> None:
        prefix = "ce" if option_type == "CE" else "pe"
        strike = float(round(strike / step) * step)
        if strike not in chain.data.index:
            missing.append(role)
            return
        ltp = chain.data.at[strike, f"{prefix}_ltp"]
        if ltp is None or not np.isfinite(ltp) or ltp <= 0:
            missing.append(role)
            return
        strikes[role] = strike
        premiums[role] = float(ltp)
        iv = chain.data.at[strike, f"{prefix}_iv"]
        if iv is not None and np.isfinite(iv):
            ivs[role] = float(iv)

    if action in (Action.BUY_CALL, Action.SELL_CALL):
        role = "long_ce" if action == Action.BUY_CALL else "short_ce"
        leg_from(role, "CE", select_strike(chain, "CE", target_delta, min_oi))
    elif action in (Action.BUY_PUT, Action.SELL_PUT):
        role = "long_pe" if action == Action.BUY_PUT else "short_pe"
        leg_from(role, "PE", select_strike(chain, "PE", target_delta, min_oi))
    elif action == Action.BULL_CALL_SPREAD:
        long_leg = select_strike(chain, "CE", target_delta, min_oi)
        leg_from("long_ce", "CE", long_leg)
        if long_leg:
            at_strike(long_leg.strike + width, "CE", "short_ce")
            reasons.append(f"Sold the {width:.0f}-point-higher call to fund the debit; "
                           f"this caps the upside at that strike.")
    elif action == Action.BEAR_PUT_SPREAD:
        long_leg = select_strike(chain, "PE", target_delta, min_oi)
        leg_from("long_pe", "PE", long_leg)
        if long_leg:
            at_strike(long_leg.strike - width, "PE", "short_pe")
            reasons.append(f"Sold the {width:.0f}-point-lower put to fund the debit.")
    elif action == Action.BULL_PUT_SPREAD:
        short_leg = select_strike(chain, "PE", min(target_delta, 0.30), min_oi)
        leg_from("short_pe", "PE", short_leg)
        if short_leg:
            at_strike(short_leg.strike - width, "PE", "long_pe")
            reasons.append(f"Bought the {width:.0f}-point-lower put as protection -- this "
                           f"is what makes the risk defined.")
    elif action == Action.BEAR_CALL_SPREAD:
        short_leg = select_strike(chain, "CE", min(target_delta, 0.30), min_oi)
        leg_from("short_ce", "CE", short_leg)
        if short_leg:
            at_strike(short_leg.strike + width, "CE", "long_ce")
            reasons.append(f"Bought the {width:.0f}-point-higher call as protection.")
    elif action == Action.IRON_CONDOR:
        short_pe = select_strike(chain, "PE", 0.20, min_oi)
        short_ce = select_strike(chain, "CE", 0.20, min_oi)
        leg_from("short_pe", "PE", short_pe)
        leg_from("short_ce", "CE", short_ce)
        if short_pe:
            at_strike(short_pe.strike - width, "PE", "long_pe")
        if short_ce:
            at_strike(short_ce.strike + width, "CE", "long_ce")
        reasons.append("Condor short strikes set at ~20 delta each: roughly an 80% "
                       "chance each side expires worthless, before costs.")
    elif action in (Action.LONG_STRADDLE, Action.SHORT_STRADDLE):
        atm = chain.atm_strike
        long = action == Action.LONG_STRADDLE
        at_strike(atm, "CE", "long_ce" if long else "short_ce")
        at_strike(atm, "PE", "long_pe" if long else "short_pe")
        reasons.append(f"Straddle struck at the ATM {atm:.0f}.")
    elif action == Action.NO_TRADE:
        return {"action": action.value, "strikes": {}, "premiums": {}, "ivs": {},
                "reasons": ["No trade recommended."], "complete": True}
    else:
        raise ValueError(f"unsupported action {action}")

    if missing:
        reasons.append(f"Could not resolve leg(s) {missing}: those strikes are absent "
                       f"or unpriced in the chain. Do not place a partial structure.")

    return {
        "action": action.value, "strikes": strikes, "premiums": premiums, "ivs": ivs,
        "reasons": reasons, "complete": not missing, "missing_legs": missing,
    }
