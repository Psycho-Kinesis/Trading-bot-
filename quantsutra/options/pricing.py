"""Black-Scholes-Merton pricing, greeks and implied volatility.

Indian index options are European-exercise and cash-settled, so plain BSM is
the right model -- no early-exercise adjustment is needed.  Two India-specific
details are handled explicitly:

* **Dividend yield.** NIFTY/SENSEX are price indices, so index futures and
  options price off the forward, which is depressed by the dividend yield of
  the constituents (~1.2-1.5% for NIFTY).  Ignoring it biases put/call
  parity and therefore every greek that depends on moneyness.

* **Time in trading days.** With ~250 trading days and ~15 holidays a year,
  using calendar days overstates the time value of a weekly option going into
  a long weekend.  ``time_to_expiry`` accepts trading days and converts with a
  252-day year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "DEFAULT_DIVIDEND_YIELD",
    "DEFAULT_RATE",
    "Greeks",
    "OptionQuote",
    "bs_price",
    "delta_to_strike",
    "forward_price",
    "greeks",
    "implied_volatility",
    "intrinsic_value",
    "moneyness",
    "time_to_expiry",
]

# Indicative levels -- override per your own funding assumptions.
DEFAULT_RATE = 0.065           # ~India 3M T-bill / MIBOR area
DEFAULT_DIVIDEND_YIELD = 0.013  # ~NIFTY 50 trailing dividend yield
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    theta: float       # per calendar day
    vega: float        # per 1 volatility point (1% = 0.01)
    rho: float
    vanna: float = float("nan")
    charm: float = float("nan")
    vomma: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "price": round(float(self.price), 2), "delta": round(float(self.delta), 4),
            "gamma": round(float(self.gamma), 6), "theta": round(float(self.theta), 3),
            "vega": round(float(self.vega), 3), "rho": round(float(self.rho), 4),
            "vanna": round(float(self.vanna), 5) if np.isfinite(self.vanna) else None,
            "charm": round(float(self.charm), 5) if np.isfinite(self.charm) else None,
            "vomma": round(float(self.vomma), 4) if np.isfinite(self.vomma) else None,
        }


@dataclass
class OptionQuote:
    strike: float
    option_type: str          # "CE" | "PE"
    ltp: float
    expiry: object
    spot: float
    iv: float | None = None
    oi: int | None = None
    oi_change: int | None = None
    volume: int | None = None
    bid: float | None = None
    ask: float | None = None

    @property
    def spread_pct(self) -> float | None:
        """Bid-ask spread as a fraction of the mid.

        This is the single most under-modelled cost in retail option backtests:
        a far-OTM weekly can quote 2% wide, which is several times a typical
        edge per trade.
        """
        if self.bid is None or self.ask is None or self.ask <= 0:
            return None
        mid = (self.bid + self.ask) / 2
        return None if mid <= 0 else (self.ask - self.bid) / mid


def time_to_expiry(days: float, trading_days: bool = True) -> float:
    """Convert days to expiry into a year fraction."""
    days = max(float(days), 0.0)
    return days / (TRADING_DAYS_PER_YEAR if trading_days else 365.0)


def forward_price(spot: float, t: float, rate: float = DEFAULT_RATE,
                  dividend_yield: float = DEFAULT_DIVIDEND_YIELD) -> float:
    return spot * math.exp((rate - dividend_yield) * t)


def _d1_d2(spot: float, strike: float, t: float, vol: float, rate: float, q: float):
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return float("nan"), float("nan")
    vt = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate - q + 0.5 * vol * vol) * t) / vt
    return d1, d1 - vt


def intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    return max(0.0, spot - strike) if option_type.upper() == "CE" else max(0.0, strike - spot)


def bs_price(spot: float, strike: float, t: float, vol: float, option_type: str,
             rate: float = DEFAULT_RATE, dividend_yield: float = DEFAULT_DIVIDEND_YIELD) -> float:
    """Black-Scholes-Merton price for a European index option."""
    option_type = option_type.upper()
    if t <= 0 or vol <= 0:
        return intrinsic_value(spot, strike, option_type)
    d1, d2 = _d1_d2(spot, strike, t, vol, rate, dividend_yield)
    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-dividend_yield * t)
    if option_type == "CE":
        return float(spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2))
    return float(strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1))


def greeks(spot: float, strike: float, t: float, vol: float, option_type: str,
           rate: float = DEFAULT_RATE, dividend_yield: float = DEFAULT_DIVIDEND_YIELD) -> Greeks:
    """Full greek set, including the second-order greeks that matter on expiry day.

    Sign conventions: theta is negative for a long option and quoted per
    *calendar* day; vega is per 1 volatility point (a move from 14% to 15%).
    """
    option_type = option_type.upper()
    if t <= 0 or vol <= 0:
        intrinsic = intrinsic_value(spot, strike, option_type)
        if option_type == "CE":
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return Greeks(intrinsic, delta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    d1, d2 = _d1_d2(spot, strike, t, vol, rate, dividend_yield)
    disc_r = math.exp(-rate * t)
    disc_q = math.exp(-dividend_yield * t)
    pdf = norm.pdf(d1)
    sqrt_t = math.sqrt(t)

    price = bs_price(spot, strike, t, vol, option_type, rate, dividend_yield)
    gamma = float(disc_q * pdf / (spot * vol * sqrt_t))
    vega = float(spot * disc_q * pdf * sqrt_t / 100.0)

    if option_type == "CE":
        delta = disc_q * norm.cdf(d1)
        theta = (-spot * disc_q * pdf * vol / (2 * sqrt_t)
                 - rate * strike * disc_r * norm.cdf(d2)
                 + dividend_yield * spot * disc_q * norm.cdf(d1)) / 365.0
        rho = strike * t * disc_r * norm.cdf(d2) / 100.0
        charm = (-disc_q * (pdf * (2 * (rate - dividend_yield) * t - d2 * vol * sqrt_t)
                            / (2 * t * vol * sqrt_t) - dividend_yield * norm.cdf(d1))) / 365.0
    else:
        delta = -disc_q * norm.cdf(-d1)
        theta = (-spot * disc_q * pdf * vol / (2 * sqrt_t)
                 + rate * strike * disc_r * norm.cdf(-d2)
                 - dividend_yield * spot * disc_q * norm.cdf(-d1)) / 365.0
        rho = -strike * t * disc_r * norm.cdf(-d2) / 100.0
        charm = (-disc_q * (pdf * (2 * (rate - dividend_yield) * t - d2 * vol * sqrt_t)
                            / (2 * t * vol * sqrt_t) + dividend_yield * norm.cdf(-d1))) / 365.0

    vanna = float(-disc_q * pdf * d2 / vol / 100.0)
    vomma = float(vega * d1 * d2 / vol)
    return Greeks(float(price), float(delta), gamma, float(theta), vega,
                  float(rho), vanna, float(charm), vomma)


def implied_volatility(
    market_price: float, spot: float, strike: float, t: float, option_type: str,
    rate: float = DEFAULT_RATE, dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
    lo: float = 0.005, hi: float = 5.0,
) -> float:
    """Back out IV from a market price.

    Uses Brent on the bracketed root, which is robust where Newton is not --
    deep OTM weeklies have near-zero vega and Newton diverges on them.
    Returns NaN when the price is outside the no-arbitrage bounds, rather than
    silently returning a garbage number.
    """
    option_type = option_type.upper()
    if not np.isfinite(market_price) or market_price <= 0 or t <= 0:
        return float("nan")

    intrinsic = intrinsic_value(spot * math.exp(-dividend_yield * t),
                                strike * math.exp(-rate * t), option_type)
    if market_price < intrinsic - 1e-6:
        return float("nan")          # below intrinsic: stale or crossed quote
    upper_bound = spot if option_type == "CE" else strike
    if market_price >= upper_bound:
        return float("nan")

    def objective(vol: float) -> float:
        return bs_price(spot, strike, t, vol, option_type, rate, dividend_yield) - market_price

    try:
        f_lo, f_hi = objective(lo), objective(hi)
        if f_lo * f_hi > 0:
            return float("nan")
        return float(brentq(objective, lo, hi, xtol=1e-6, maxiter=100))
    except (ValueError, RuntimeError):
        return float("nan")


def moneyness(spot: float, strike: float, option_type: str) -> str:
    """ITM / ATM / OTM label; ATM is within 0.25% of spot."""
    band = spot * 0.0025
    if abs(spot - strike) <= band:
        return "ATM"
    if option_type.upper() == "CE":
        return "ITM" if spot > strike else "OTM"
    return "ITM" if spot < strike else "OTM"


def delta_to_strike(target_delta: float, spot: float, t: float, vol: float, option_type: str,
                    strike_step: int = 50, rate: float = DEFAULT_RATE,
                    dividend_yield: float = DEFAULT_DIVIDEND_YIELD) -> float:
    """Find the listed strike whose delta is closest to ``target_delta``.

    Delta-targeting is how professional desks pick strikes -- "sell the 20
    delta" is a consistent instruction across expiries and volatility levels,
    whereas "sell 300 points OTM" is not.
    """
    target = abs(target_delta)
    lo = spot * 0.7
    hi = spot * 1.3
    best_strike, best_err = spot, float("inf")
    k = math.floor(lo / strike_step) * strike_step
    while k <= hi:
        if k > 0:
            g = greeks(spot, k, t, vol, option_type, rate, dividend_yield)
            err = abs(abs(g.delta) - target)
            if err < best_err:
                best_strike, best_err = k, err
        k += strike_step
    return float(best_strike)
