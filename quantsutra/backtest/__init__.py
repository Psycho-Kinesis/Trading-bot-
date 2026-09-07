"""Backtesting: event-driven simulation, metrics and walk-forward validation."""

from .engine import BacktestConfig, BacktestResult, Backtester, Trade
from .metrics import (PerformanceReport, buy_and_hold, compute_metrics,
                      drawdown_series, win_rate_confidence_interval)
from .walkforward import WalkForwardResult, rolling_windows, walk_forward

__all__ = ["Backtester", "BacktestConfig", "BacktestResult", "Trade",
           "compute_metrics", "PerformanceReport", "drawdown_series",
           "win_rate_confidence_interval", "buy_and_hold",
           "walk_forward", "WalkForwardResult", "rolling_windows"]
