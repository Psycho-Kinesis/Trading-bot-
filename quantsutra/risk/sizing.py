"""Position sizing.

Sizing decides survival; entries decide little by comparison.  Two things this
module refuses to do:

* Size an options position off the index stop distance alone.  A 200-point
  NIFTY stop does not cost 200 x lot size on a long option -- it costs the
  premium change, which depends on delta.  Sizing off the index move without
  the delta translation systematically *undersizes* premium risk.

* Return a fractional lot.  Indian index derivatives trade in fixed lots, so
  the answer is an integer number of lots or zero, and "zero" is a legitimate
  and frequent answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["SizingResult", "fixed_fractional", "size_option_position",
           "kelly_fraction", "volatility_target_size"]


@dataclass
class SizingResult:
    lots: int
    units: int
    risk_amount: float          # rupees actually at risk if the stop is hit
    risk_pct_of_capital: float
    max_loss: float             # worst case for this structure
    notional: float
    margin_estimate: float | None
    reasons: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "lots": self.lots, "units": self.units,
            "risk_amount": round(self.risk_amount, 2),
            "risk_pct_of_capital": round(self.risk_pct_of_capital, 3),
            "max_loss": round(self.max_loss, 2),
            "notional": round(self.notional, 2),
            "margin_estimate": round(self.margin_estimate, 2) if self.margin_estimate else None,
            "reasons": self.reasons, "warnings": self.warnings,
        }


def fixed_fractional(capital: float, risk_pct: float, risk_per_unit: float,
                     lot_size: int, max_lots: int = 100) -> int:
    """Classic fixed-fractional sizing, rounded down to whole lots."""
    if capital <= 0 or risk_pct <= 0 or risk_per_unit <= 0 or lot_size <= 0:
        return 0
    budget = capital * risk_pct
    risk_per_lot = risk_per_unit * lot_size
    return int(min(max_lots, math.floor(budget / risk_per_lot))) if risk_per_lot > 0 else 0


def size_option_position(
    capital: float,
    risk_pct: float,
    premium: float,
    lot_size: int,
    delta: float | None = None,
    index_stop_distance: float | None = None,
    max_premium_pct: float = 0.25,
    stop_is_premium_pct: float | None = 0.40,
    max_lots: int = 100,
    direction: str = "LONG",
) -> SizingResult:
    """Size a long option position.

    Risk per unit is taken as the *smaller* implied risk of two views, so the
    position is never larger than either view allows:

    1. **Delta-translated index stop.**  If the index moves ``index_stop_distance``
       against you, the option loses roughly ``delta x distance`` per unit
       (ignoring gamma, which makes it worse, and theta, which adds to it).
    2. **Premium stop.**  A hard "cut it at -40% of premium" rule, which is how
       most option buyers actually exit.

    ``max_premium_pct`` additionally caps total premium outlay as a share of
    capital, because on a long option the entire premium is at risk if the
    index gaps through the stop overnight -- and Indian indices gap regularly.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    if capital <= 0 or premium <= 0 or lot_size <= 0:
        return SizingResult(0, 0, 0, 0, 0, 0, None,
                            ["Invalid inputs: capital, premium and lot size must be positive."],
                            [])

    candidates: list[tuple[float, str]] = []
    if delta is not None and index_stop_distance and abs(delta) > 0.01:
        per_unit = abs(delta) * abs(index_stop_distance)
        capped = per_unit > premium
        candidates.append((min(per_unit, premium), "delta-translated index stop"))
        if capped:
            reasons.append(
                f"A {abs(index_stop_distance):.0f}-point adverse index move implies about "
                f"Rs.{per_unit:.1f} per unit at delta {abs(delta):.2f}, which is more than "
                f"the Rs.{premium:.1f} premium -- the option would be near worthless well "
                f"before the index stop is reached, so the whole premium is the real risk."
            )
        else:
            reasons.append(
                f"A {abs(index_stop_distance):.0f}-point adverse index move costs about "
                f"Rs.{per_unit:.1f} per unit at delta {abs(delta):.2f} "
                f"(gamma and theta make the real loss larger)."
            )
    if stop_is_premium_pct:
        per_unit = premium * stop_is_premium_pct
        candidates.append((per_unit, f"{stop_is_premium_pct:.0%} premium stop"))
        reasons.append(f"A {stop_is_premium_pct:.0%} premium stop risks Rs.{per_unit:.1f} per unit.")

    if not candidates:
        candidates.append((premium, "full premium at risk"))
        warnings.append("No stop basis supplied: assuming the entire premium can be lost.")

    risk_per_unit, basis = max(candidates, key=lambda c: c[0])
    reasons.append(f"Sizing off the more conservative basis: {basis}.")

    lots = fixed_fractional(capital, risk_pct, risk_per_unit, lot_size, max_lots)

    # Cap total premium outlay regardless of the stop.
    premium_cap_lots = int(math.floor(capital * max_premium_pct / (premium * lot_size)))
    if premium_cap_lots < lots:
        warnings.append(
            f"Reduced from {lots} to {premium_cap_lots} lot(s) to keep total premium outlay "
            f"under {max_premium_pct:.0%} of capital -- an overnight gap can take the whole "
            f"premium, not just the stop distance."
        )
        lots = premium_cap_lots

    if lots < 1:
        needed = risk_per_unit * lot_size / max(risk_pct, 1e-9)
        return SizingResult(
            0, 0, 0.0, 0.0, 0.0, 0.0, None,
            reasons + [
                f"One lot risks Rs.{risk_per_unit * lot_size:,.0f}, which exceeds "
                f"{risk_pct:.1%} of Rs.{capital:,.0f}. Correct size is ZERO lots."
            ],
            warnings + [
                f"Taking this trade at 1 lot would risk "
                f"{100 * risk_per_unit * lot_size / capital:.1f}% of capital. "
                f"You would need about Rs.{needed:,.0f} of capital to take it at your "
                f"stated risk limit."
            ],
        )

    units = lots * lot_size
    risk_amount = risk_per_unit * units
    total_premium = premium * units
    return SizingResult(
        lots=lots, units=units, risk_amount=risk_amount,
        risk_pct_of_capital=risk_amount / capital,
        max_loss=total_premium if direction.upper() == "LONG" else float("inf"),
        notional=total_premium,
        margin_estimate=total_premium if direction.upper() == "LONG" else None,
        reasons=reasons + [
            f"{lots} lot(s) = {units} units. Premium outlay Rs.{total_premium:,.0f}, "
            f"risk at the stop Rs.{risk_amount:,.0f} "
            f"({100 * risk_amount / capital:.2f}% of capital)."
        ],
        warnings=warnings,
    )


def volatility_target_size(capital: float, target_annual_vol: float, instrument_vol: float,
                           price: float, lot_size: int, max_lots: int = 100,
                           leverage: float = 1.0) -> int:
    """Size so the position's own volatility matches a target.

    Useful for holding exposure steady across regimes: the same lot count is a
    much bigger bet when NIFTY is realising 22% than when it is realising 9%.

    ``leverage`` matters here.  One NIFTY futures lot carries roughly
    Rs.19 lakh of notional but needs only ~Rs.2 lakh of margin, so unleveraged
    notional targeting (``leverage=1``) will return zero lots for any retail
    account.  Pass the leverage your margin actually gives you -- and note that
    the volatility target then applies to the *notional*, which is the correct
    denominator for risk even though it is not the cash you put up.
    """
    if instrument_vol <= 0 or price <= 0 or lot_size <= 0 or leverage <= 0:
        return 0
    target_notional = capital * leverage * (target_annual_vol / instrument_vol)
    return int(min(max_lots, math.floor(target_notional / (price * lot_size))))


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   cap: float = 0.25) -> dict:
    """Kelly fraction, deliberately capped.

    Full Kelly maximises long-run growth *given exact knowledge* of the edge.
    Nobody has that.  Estimation error means full Kelly regularly produces
    drawdowns most people cannot hold through, so this returns quarter-Kelly
    by default and reports the full number alongside for reference.
    """
    if avg_loss <= 0 or not (0 < win_rate < 1):
        return {"kelly": 0.0, "capped": 0.0, "note": "insufficient or invalid statistics"}
    b = avg_win / avg_loss
    kelly = (win_rate * (b + 1) - 1) / b
    if kelly <= 0:
        return {"kelly": round(kelly, 4), "capped": 0.0,
                "note": "negative edge -- Kelly says do not take this trade at all"}
    capped = min(kelly * 0.25, cap)
    return {
        "kelly": round(kelly, 4), "quarter_kelly": round(kelly * 0.25, 4),
        "capped": round(capped, 4), "payoff_ratio": round(b, 2),
        "note": ("Full Kelly assumes your win rate and payoff estimates are exact. "
                 "They are not -- quarter-Kelly is the practical ceiling."),
    }
