"""Transaction cost modelling for Indian index derivatives.

Cost is not a rounding error in this market.  A single NIFTY option round trip
carries STT on the sell side, exchange transaction charges, SEBI turnover
fees, stamp duty on the buy side, GST on top of the fee components, and
brokerage -- plus the bid-ask spread, which is usually the largest component
of all and the one most often left out of retail backtests.

Every number is loaded from ``config/costs.yaml`` so it can be updated when a
circular lands, and so a backtest records what it assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

__all__ = ["CostBreakdown", "CostModel", "estimate_slippage", "load_cost_config"]

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "costs.yaml"


@lru_cache(maxsize=1)
def load_cost_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


@dataclass
class CostBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange_charge: float = 0.0
    sebi_fee: float = 0.0
    stamp_duty: float = 0.0
    ipft: float = 0.0
    clearing: float = 0.0
    gst: float = 0.0
    slippage: float = 0.0

    @property
    def statutory(self) -> float:
        """Everything except slippage -- what appears on the contract note."""
        return (self.brokerage + self.stt + self.exchange_charge + self.sebi_fee
                + self.stamp_duty + self.ipft + self.clearing + self.gst)

    @property
    def total(self) -> float:
        return self.statutory + self.slippage

    def to_dict(self) -> dict:
        return {
            "brokerage": round(self.brokerage, 2), "stt": round(self.stt, 2),
            "exchange_charge": round(self.exchange_charge, 2),
            "sebi_fee": round(self.sebi_fee, 2), "stamp_duty": round(self.stamp_duty, 2),
            "ipft": round(self.ipft, 2), "clearing": round(self.clearing, 2),
            "gst": round(self.gst, 2), "slippage": round(self.slippage, 2),
            "statutory": round(self.statutory, 2), "total": round(self.total, 2),
        }

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage, stt=self.stt + other.stt,
            exchange_charge=self.exchange_charge + other.exchange_charge,
            sebi_fee=self.sebi_fee + other.sebi_fee,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            ipft=self.ipft + other.ipft, clearing=self.clearing + other.clearing,
            gst=self.gst + other.gst, slippage=self.slippage + other.slippage,
        )


def estimate_slippage(premium: float, quantity: int, moneyness: str = "ATM",
                      is_expiry_day: bool = False, config: dict | None = None) -> float:
    """Half-spread cost for one side of an option trade, in rupees.

    Defaults are conservative guesses, not measurements.  If your feed carries
    bid/ask, pass the real spread instead -- this is where backtest optimism
    hides.
    """
    cfg = (config or load_cost_config())["slippage"]
    key = {"ATM": "option_atm_pct", "ITM": "option_atm_pct",
           "OTM": "option_otm_pct", "FAR_OTM": "option_far_otm_pct"}.get(
        moneyness.upper(), "option_otm_pct")
    pct = cfg[key]
    if is_expiry_day:
        pct *= cfg.get("expiry_day_multiplier", 1.0)
    return abs(premium) * quantity * pct


@dataclass
class CostModel:
    """Computes the all-in cost of an options or futures trade."""

    exchange: str = "NSE"
    config: dict = field(default_factory=load_cost_config)
    include_slippage: bool = True

    # -- options ----------------------------------------------------------
    def option_leg(
        self, premium: float, quantity: int, side: str, moneyness: str = "ATM",
        is_expiry_day: bool = False, exercised_intrinsic: float | None = None,
    ) -> CostBreakdown:
        """Cost of one option leg.

        ``side`` is ``"BUY"`` or ``"SELL"``.  ``quantity`` is the number of
        units (lots x lot size), not lots.  ``exercised_intrinsic`` triggers
        the higher exercise STT that applies when an ITM option is allowed to
        expire rather than being squared off -- which is why squaring off ITM
        options before 15:30 on expiry day is almost always cheaper.
        """
        cfg = self.config
        opt = cfg["index_options"]
        side = side.upper()
        turnover = abs(premium) * quantity
        b = CostBreakdown()

        br = cfg["brokerage"]
        b.brokerage = min(br["flat_per_order"], turnover * br["percent_of_turnover"])
        b.brokerage = min(b.brokerage, br["cap_per_order"])

        if side == "SELL":
            b.stt = turnover * opt["stt_sell_premium"]
        else:
            b.stamp_duty = turnover * opt["stamp_duty_buy"]

        if exercised_intrinsic:
            # Exercise STT is charged on intrinsic value, and is an order of
            # magnitude larger than the premium-based STT on a square-off.
            b.stt += abs(exercised_intrinsic) * quantity * opt["stt_exercise_intrinsic"]

        rate_key = "nse_transaction_charge" if self.exchange.upper() == "NSE" else "bse_transaction_charge"
        b.exchange_charge = turnover * opt[rate_key]
        b.sebi_fee = turnover * cfg["sebi_turnover_fee"]
        b.ipft = turnover * opt.get("ipft", 0.0)
        b.clearing = turnover * opt.get("clearing", 0.0)
        b.gst = (b.brokerage + b.exchange_charge + b.sebi_fee) * cfg["gst_rate"]

        if self.include_slippage:
            b.slippage = estimate_slippage(premium, quantity, moneyness, is_expiry_day, cfg)
        return b

    def option_round_trip(
        self, entry_premium: float, exit_premium: float, quantity: int,
        direction: str = "LONG", moneyness: str = "ATM", is_expiry_day: bool = False,
    ) -> CostBreakdown:
        """Both legs of an option position, entry and exit."""
        if direction.upper() == "LONG":
            entry = self.option_leg(entry_premium, quantity, "BUY", moneyness, is_expiry_day)
            exit_ = self.option_leg(exit_premium, quantity, "SELL", moneyness, is_expiry_day)
        else:
            entry = self.option_leg(entry_premium, quantity, "SELL", moneyness, is_expiry_day)
            exit_ = self.option_leg(exit_premium, quantity, "BUY", moneyness, is_expiry_day)
        return entry + exit_

    # -- futures ----------------------------------------------------------
    def futures_leg(self, price: float, quantity: int, side: str) -> CostBreakdown:
        cfg = self.config
        fut = cfg["index_futures"]
        turnover = abs(price) * quantity
        b = CostBreakdown()

        br = cfg["brokerage"]
        b.brokerage = min(br["flat_per_order"], turnover * br["percent_of_turnover"],
                          br["cap_per_order"])
        if side.upper() == "SELL":
            b.stt = turnover * fut["stt_sell"]
        else:
            b.stamp_duty = turnover * fut["stamp_duty_buy"]

        rate_key = "nse_transaction_charge" if self.exchange.upper() == "NSE" else "bse_transaction_charge"
        b.exchange_charge = turnover * fut[rate_key]
        b.sebi_fee = turnover * cfg["sebi_turnover_fee"]
        b.ipft = turnover * fut.get("ipft", 0.0)
        b.gst = (b.brokerage + b.exchange_charge + b.sebi_fee) * cfg["gst_rate"]

        if self.include_slippage:
            ticks = cfg["slippage"]["index_futures_ticks"]
            b.slippage = ticks * 0.05 * quantity
        return b

    def futures_round_trip(self, entry: float, exit_: float, quantity: int) -> CostBreakdown:
        return self.futures_leg(entry, quantity, "BUY") + self.futures_leg(exit_, quantity, "SELL")

    # -- helpers ----------------------------------------------------------
    def breakeven_move(self, premium: float, quantity: int, moneyness: str = "ATM",
                       is_expiry_day: bool = False) -> dict:
        """How far the option must move just to cover costs.

        This is the number worth internalising: on a cheap OTM weekly, costs
        alone can require a 3-5% move in the premium before the trade is at
        breakeven.
        """
        rt = self.option_round_trip(premium, premium, quantity, "LONG", moneyness, is_expiry_day)
        per_unit = rt.total / quantity if quantity else 0.0
        return {
            "total_cost": round(rt.total, 2),
            "cost_per_unit": round(per_unit, 3),
            "premium_move_needed": round(per_unit, 2),
            "premium_move_pct": round(100 * per_unit / premium, 2) if premium > 0 else None,
            "breakdown": rt.to_dict(),
        }
