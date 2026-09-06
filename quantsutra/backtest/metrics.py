"""Performance metrics.

Deliberately reports the numbers that are unflattering as well as the ones
that are not.  In particular:

* **Return per unit of drawdown**, not just CAGR.  A 40% return with a 35%
  drawdown is not a good strategy, it is a leveraged one.
* **Trade count and its implication for significance.**  Thirty trades cannot
  establish an edge; the confidence interval on the win rate is reported so
  that is visible rather than assumed away.
* **Cost drag** as an explicit line, because a strategy that is profitable
  gross and unprofitable net is the normal outcome for intraday options.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PerformanceReport", "compute_metrics", "drawdown_series",
           "win_rate_confidence_interval"]

TRADING_DAYS = 252


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak.replace(0, np.nan)


def win_rate_confidence_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for the win rate.

    With 30 trades a 60% win rate has a 95% interval of roughly 42%-75%.  That
    range includes "no edge", which is exactly the point: small samples cannot
    distinguish skill from luck, and this makes the ambiguity explicit.
    """
    if total <= 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class PerformanceReport:
    metrics: dict
    equity: pd.Series
    trades: pd.DataFrame

    def summary_lines(self) -> list[str]:
        m = self.metrics
        lines = [
            f"Trades          : {m['trades']} ({m['wins']}W / {m['losses']}L)",
            f"Win rate        : {m['win_rate']:.1%}  (95% CI {m['win_rate_ci'][0]:.1%}-{m['win_rate_ci'][1]:.1%})",
            f"Net P&L         : Rs.{m['net_pnl']:,.0f}   (gross Rs.{m['gross_pnl']:,.0f}, costs Rs.{m['total_costs']:,.0f})",
            f"Return          : {m['total_return']:.2%}   CAGR {m['cagr']:.2%}" if m.get("cagr") is not None else
            f"Return          : {m['total_return']:.2%}",
            f"Max drawdown    : {m['max_drawdown']:.2%}   (Rs.{m['max_drawdown_value']:,.0f})",
            f"Sharpe / Sortino: {m['sharpe']:.2f} / {m['sortino']:.2f}",
            f"Profit factor   : {m['profit_factor']}",
            f"Expectancy      : Rs.{m['expectancy']:,.0f} per trade  ({m['expectancy_r']:.2f} R)",
            f"Avg win / loss  : Rs.{m['avg_win']:,.0f} / Rs.{m['avg_loss']:,.0f}",
            f"Max consec loss : {m['max_consecutive_losses']}",
            (f"Cost drag       : {m['cost_drag_pct']:.1f}% of gross profit"
             if m['gross_pnl'] > 0 else
             f"Costs           : Rs.{m['total_costs']:,.0f} "
             f"(gross was already negative, so costs deepened the loss)"),
        ]
        if m.get("time_in_market_pct") is not None:
            lines.append(f"Time in market  : {m['time_in_market_pct']:.1f}% of bars")
        if m.get("return_over_max_dd") is not None:
            lines.append(f"Return / max DD : {m['return_over_max_dd']:.2f}")
        return lines

    def caveats(self) -> list[str]:
        """Statistical health warnings for this particular result."""
        m = self.metrics
        out = []
        if m["trades"] < 100:
            out.append(
                f"Only {m['trades']} trades. A sample this small cannot distinguish an edge "
                f"from luck -- the win-rate confidence interval spans "
                f"{m['win_rate_ci'][0]:.0%} to {m['win_rate_ci'][1]:.0%}."
            )
        if m["trades"] and m["win_rate_ci"][0] < 0.5 < m["win_rate_ci"][1]:
            out.append("The win-rate confidence interval straddles 50%: this result is "
                       "statistically consistent with having no directional edge at all.")
        if m["total_costs"] > 0 and m["gross_pnl"] > 0 and m["cost_drag_pct"] > 40:
            out.append(f"Costs consumed {m['cost_drag_pct']:.0f}% of gross profit. Small "
                       f"changes in slippage assumptions would flip this to a loss.")
        if m["gross_pnl"] > 0 and m["net_pnl"] <= 0:
            out.append("Profitable before costs, unprofitable after. This is the normal "
                       "outcome for high-frequency option strategies and the reason gross "
                       "backtests mislead.")
        if m["max_drawdown"] < -0.25:
            out.append(f"Maximum drawdown of {m['max_drawdown']:.0%}. Ask honestly whether "
                       f"you would have kept trading the system through that.")
        if m.get("best_trade_share") and m["best_trade_share"] > 0.4:
            out.append(f"The single best trade produced {m['best_trade_share']:.0%} of total "
                       f"profit. Remove it and the strategy may have no edge.")
        return out

    def to_dict(self) -> dict:
        return {"metrics": self.metrics, "caveats": self.caveats()}


def _time_in_market(trades: pd.DataFrame, equity: pd.Series) -> float | None:
    """Share of the backtest period spent holding a position.

    Low time in market is not a flaw -- a selective system should be flat most
    of the time -- but it changes how Sharpe and drawdown should be read.
    """
    if trades is None or trades.empty or "bars_held" not in trades or len(equity) < 2:
        return None
    bars_held = float(trades["bars_held"].fillna(0).sum())
    return round(100 * bars_held / len(equity), 1)


def compute_metrics(
    trades: pd.DataFrame, equity: pd.Series, initial_capital: float,
    periods_per_year: int = TRADING_DAYS, risk_free_rate: float = 0.0,
) -> PerformanceReport:
    """Build the full report from a trade log and an equity curve.

    ``trades`` needs ``pnl`` and optionally ``costs``, ``risk``, ``entry_time``,
    ``exit_time``.

    ``risk_free_rate`` defaults to zero on purpose.  A signal system is flat
    most of the time, and idle capital sits in cash earning roughly the
    risk-free rate -- so charging that rate against every flat day produces a
    large negative Sharpe for a profitable strategy.  Pass a non-zero rate only
    if your equity curve genuinely represents fully-deployed capital.
    """
    if trades is None or trades.empty:
        empty = {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "win_rate_ci": (0.0, 0.0), "net_pnl": 0.0, "gross_pnl": 0.0,
            "total_costs": 0.0, "total_return": 0.0, "cagr": None,
            "max_drawdown": 0.0, "max_drawdown_value": 0.0, "sharpe": 0.0,
            "sortino": 0.0, "profit_factor": None, "expectancy": 0.0,
            "expectancy_r": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_consecutive_losses": 0, "cost_drag_pct": 0.0,
            "best_trade_share": None, "time_in_market_pct": None,
            "return_over_max_dd": None,
        }
        return PerformanceReport(empty, equity, trades if trades is not None else pd.DataFrame())

    pnl = trades["pnl"].astype(float)
    costs = trades["costs"].astype(float) if "costs" in trades else pd.Series(0.0, index=trades.index)
    gross = float((pnl + costs).sum())
    net = float(pnl.sum())

    wins_mask = pnl > 0
    losses_mask = pnl < 0
    wins, losses = int(wins_mask.sum()), int(losses_mask.sum())
    total = len(pnl)
    win_rate = wins / total if total else 0.0

    gross_profit = float(pnl[wins_mask].sum())
    gross_loss = float(-pnl[losses_mask].sum())
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

    avg_win = float(pnl[wins_mask].mean()) if wins else 0.0
    avg_loss = float(pnl[losses_mask].mean()) if losses else 0.0
    expectancy = float(pnl.mean())

    expectancy_r = 0.0
    if "risk" in trades:
        risk = trades["risk"].astype(float).replace(0, np.nan)
        r_multiples = (pnl / risk).dropna()
        expectancy_r = float(r_multiples.mean()) if len(r_multiples) else 0.0

    # Longest losing streak.
    streak = max_streak = 0
    for value in pnl:
        if value < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    dd = drawdown_series(equity) if len(equity) else pd.Series([0.0])
    max_dd = float(dd.min()) if len(dd) else 0.0
    peak = equity.cummax() if len(equity) else pd.Series([initial_capital])
    max_dd_value = float((equity - peak).min()) if len(equity) else 0.0

    returns = equity.pct_change().dropna() if len(equity) > 1 else pd.Series(dtype=float)
    sharpe = sortino = 0.0
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        excess = returns - risk_free_rate / periods_per_year
        sharpe = float(excess.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year))
        downside = returns[returns < 0]
        if len(downside) > 1 and downside.std(ddof=1) > 0:
            sortino = float(excess.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))

    total_return = net / initial_capital if initial_capital else 0.0
    cagr = None
    if "exit_time" in trades and "entry_time" in trades and len(trades) > 1:
        try:
            span_days = (pd.Timestamp(trades["exit_time"].iloc[-1])
                         - pd.Timestamp(trades["entry_time"].iloc[0])).days
            if span_days > 30:
                years = span_days / 365.25
                ending = initial_capital + net
                if ending > 0 and initial_capital > 0:
                    cagr = float((ending / initial_capital) ** (1 / years) - 1)
        except (TypeError, ValueError):
            cagr = None

    best_share = None
    if gross_profit > 0 and wins:
        best_share = float(pnl.max() / gross_profit)

    metrics = {
        "trades": total, "wins": wins, "losses": losses, "win_rate": win_rate,
        "win_rate_ci": win_rate_confidence_interval(wins, total),
        "net_pnl": net, "gross_pnl": gross, "total_costs": float(costs.sum()),
        "total_return": total_return, "cagr": cagr,
        "max_drawdown": max_dd, "max_drawdown_value": max_dd_value,
        "sharpe": sharpe, "sortino": sortino, "profit_factor": profit_factor,
        "expectancy": expectancy, "expectancy_r": expectancy_r,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "max_consecutive_losses": max_streak,
        # Only meaningful against a positive gross; reported as None otherwise
        # so a losing strategy cannot appear to have zero cost drag.
        "cost_drag_pct": (100 * float(costs.sum()) / gross) if gross > 0 else 0.0,
        "costs_vs_gross_note": (None if gross > 0 else "gross P&L is negative"),
        "best_trade_share": best_share,
        "return_over_max_dd": (total_return / abs(max_dd)) if max_dd < 0 else None,
        "time_in_market_pct": _time_in_market(trades, equity),
    }
    return PerformanceReport(metrics, equity, trades)
