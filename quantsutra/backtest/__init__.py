"""Backtesting: event-driven simulation, metrics and walk-forward validation."""

from .engine import BacktestConfig, Backtester, BacktestResult, Trade
from .metrics import (
                      PerformanceReport,
                      buy_and_hold,
                      compute_metrics,
                      drawdown_series,
                      win_rate_confidence_interval,
)
from .walkforward import WalkForwardResult, rolling_windows, walk_forward

__all__ = [
                      "BacktestConfig",
                      "BacktestResult",
                      "Backtester",
                      "PerformanceReport",
                      "Trade",
                      "WalkForwardResult",
                      "buy_and_hold",
                      "compute_metrics",
                      "drawdown_series",
                      "rolling_windows",
                      "walk_forward",
                      "win_rate_confidence_interval",
]
