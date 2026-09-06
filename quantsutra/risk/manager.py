"""Portfolio-level risk limits and the kill switch.

Individual trade sizing is not enough.  Most account blow-ups in Indian F&O
come from *sequences*: three losses in a row on expiry day, then doubling up
to "get it back".  The limits here are checked before every signal is allowed
through, and the daily-loss and consecutive-loss stops are deliberately hard.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

__all__ = ["RiskLimits", "RiskManager", "RiskCheck"]


@dataclass
class RiskLimits:
    max_risk_per_trade: float = 0.01        # fraction of capital
    max_daily_loss: float = 0.03            # hard stop for the day
    max_weekly_loss: float = 0.06
    max_open_positions: int = 3
    max_positions_same_direction: int = 2
    max_consecutive_losses: int = 3
    max_trades_per_day: int = 6
    max_capital_deployed: float = 0.40      # premium outlay ceiling
    block_new_after_daily_target: float | None = 0.06   # stop while ahead
    block_expiry_day_selling: bool = True
    block_first_minutes: int = 15           # avoid the opening auction drift
    block_last_minutes: int = 10            # avoid closing-auction illiquidity


@dataclass
class RiskCheck:
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    adjustments: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"allowed": self.allowed, "reasons": self.reasons,
                "adjustments": self.adjustments}


@dataclass
class RiskManager:
    capital: float
    limits: RiskLimits = field(default_factory=RiskLimits)

    day_pnl: float = 0.0
    week_pnl: float = 0.0
    open_positions: list = field(default_factory=list)
    trades_today: int = 0
    consecutive_losses: int = 0
    deployed_capital: float = 0.0
    halted_reason: str | None = None
    _current_day: dt.date | None = None

    # -- bookkeeping -------------------------------------------------------
    def roll_day(self, day: dt.date) -> None:
        """Reset daily counters.  Call once per trading day."""
        if self._current_day == day:
            return
        if self._current_day is not None and day.isocalendar()[1] != self._current_day.isocalendar()[1]:
            self.week_pnl = 0.0
        self._current_day = day
        self.day_pnl = 0.0
        self.trades_today = 0
        self.halted_reason = None

    def record_trade(self, pnl: float) -> None:
        self.day_pnl += pnl
        self.week_pnl += pnl
        self.trades_today += 1
        if pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = 0

    # -- the gate ----------------------------------------------------------
    def check(
        self, direction: str, risk_amount: float, premium_outlay: float = 0.0,
        now: dt.datetime | None = None, minutes_into_session: int | None = None,
        is_expiry_day: bool = False, is_short_premium: bool = False,
    ) -> RiskCheck:
        """Decide whether a new position may be opened."""
        lim = self.limits
        reasons: list[str] = []
        adjustments: dict = {}
        allowed = True

        if self.halted_reason:
            return RiskCheck(False, [f"Trading halted for the day: {self.halted_reason}"])

        loss_limit = -self.capital * lim.max_daily_loss
        if self.day_pnl <= loss_limit:
            self.halted_reason = (
                f"daily loss limit hit (Rs.{self.day_pnl:,.0f} against a "
                f"Rs.{loss_limit:,.0f} limit)")
            return RiskCheck(False, [
                f"Daily loss limit reached: Rs.{self.day_pnl:,.0f} vs limit "
                f"Rs.{loss_limit:,.0f}. No further trades today. Revenge trading after a "
                f"limit-down day is the most reliable way to turn a bad day into a bad month."
            ])

        weekly_limit = -self.capital * lim.max_weekly_loss
        if self.week_pnl <= weekly_limit:
            return RiskCheck(False, [
                f"Weekly loss limit reached (Rs.{self.week_pnl:,.0f} vs "
                f"Rs.{weekly_limit:,.0f}). Stop and review the week before continuing."
            ])

        if lim.block_new_after_daily_target:
            target = self.capital * lim.block_new_after_daily_target
            if self.day_pnl >= target:
                return RiskCheck(False, [
                    f"Daily profit target reached (Rs.{self.day_pnl:,.0f}). Stopping while "
                    f"ahead is a rule worth keeping -- most give-backs happen after a good "
                    f"morning."
                ])

        if self.consecutive_losses >= lim.max_consecutive_losses:
            return RiskCheck(False, [
                f"{self.consecutive_losses} consecutive losses. Either the regime has "
                f"changed or the setup has stopped working; stop and re-read the tape "
                f"rather than sizing up."
            ])

        if self.trades_today >= lim.max_trades_per_day:
            return RiskCheck(False, [
                f"Trade count limit reached ({self.trades_today}/{lim.max_trades_per_day}). "
                f"Overtrading converts a small edge into commission for your broker."
            ])

        if len(self.open_positions) >= lim.max_open_positions:
            return RiskCheck(False, [
                f"Already holding {len(self.open_positions)} positions "
                f"(limit {lim.max_open_positions}). Correlated index positions are one "
                f"bet, not several."
            ])

        same_dir = sum(1 for p in self.open_positions
                       if str(getattr(p, "direction", p.get("direction") if isinstance(p, dict) else "")) == direction)
        if same_dir >= lim.max_positions_same_direction:
            return RiskCheck(False, [
                f"Already {same_dir} open position(s) in the {direction} direction. "
                f"NIFTY, BANKNIFTY and SENSEX move together -- stacking them multiplies "
                f"the same risk rather than diversifying it."
            ])

        max_risk = self.capital * lim.max_risk_per_trade
        if risk_amount > max_risk:
            allowed = False
            reasons.append(
                f"Risk Rs.{risk_amount:,.0f} exceeds the per-trade limit of "
                f"Rs.{max_risk:,.0f} ({lim.max_risk_per_trade:.1%} of capital)."
            )
            adjustments["max_risk_amount"] = max_risk

        if premium_outlay:
            projected = self.deployed_capital + premium_outlay
            cap = self.capital * lim.max_capital_deployed
            if projected > cap:
                allowed = False
                reasons.append(
                    f"Premium outlay would take deployed capital to Rs.{projected:,.0f}, "
                    f"above the Rs.{cap:,.0f} ceiling ({lim.max_capital_deployed:.0%})."
                )
                adjustments["max_premium_outlay"] = max(0.0, cap - self.deployed_capital)

        if minutes_into_session is not None:
            if minutes_into_session < lim.block_first_minutes:
                allowed = False
                reasons.append(
                    f"Only {minutes_into_session} minutes into the session. The first "
                    f"{lim.block_first_minutes} minutes carry the widest spreads and the "
                    f"least reliable direction."
                )
            elif minutes_into_session > 375 - lim.block_last_minutes:
                allowed = False
                reasons.append(
                    f"Within {375 - minutes_into_session} minutes of the close -- "
                    f"liquidity thins and exits get expensive."
                )

        if is_expiry_day and is_short_premium and lim.block_expiry_day_selling:
            allowed = False
            reasons.append(
                "Selling premium on expiry day is blocked. Gamma is effectively unbounded "
                "in the final hours; a 100-point move against a naked short can cost more "
                "than a month of collected premium."
            )

        if allowed and not reasons:
            reasons.append(
                f"Within all limits: risk Rs.{risk_amount:,.0f} "
                f"({100 * risk_amount / self.capital:.2f}% of capital), "
                f"{len(self.open_positions)}/{lim.max_open_positions} positions open, "
                f"day P&L Rs.{self.day_pnl:,.0f}."
            )
        return RiskCheck(allowed, reasons, adjustments)

    def state(self) -> dict:
        return {
            "capital": round(self.capital, 2),
            "day_pnl": round(self.day_pnl, 2),
            "week_pnl": round(self.week_pnl, 2),
            "day_pnl_pct": round(100 * self.day_pnl / self.capital, 2) if self.capital else 0,
            "open_positions": len(self.open_positions),
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "deployed_capital": round(self.deployed_capital, 2),
            "halted": self.halted_reason,
            "daily_loss_limit": round(-self.capital * self.limits.max_daily_loss, 2),
            "room_to_daily_limit": round(
                self.day_pnl + self.capital * self.limits.max_daily_loss, 2),
        }
