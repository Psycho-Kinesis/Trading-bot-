"""Rendering: terminal output for signals, regimes, chains and backtests."""

from .console import (
                      console,
                      render_backtest,
                      render_chain,
                      render_events,
                      render_regime,
                      render_signal,
)

__all__ = [
                      "console",
                      "render_backtest",
                      "render_chain",
                      "render_events",
                      "render_regime",
                      "render_signal",
]
