"""Backtesting: event-driven simulation, metrics and walk-forward validation."""

from .engine import BacktestConfig, BacktestResult, Backtester, Trade
from .metrics import (PerformanceReport, compute_metrics, drawdown_series,
                      win_rate_confidence_interval)

__all__ = ["Backtester", "BacktestConfig", "BacktestResult", "Trade",
           "compute_metrics", "PerformanceReport", "drawdown_series",
           "win_rate_confidence_interval"]
