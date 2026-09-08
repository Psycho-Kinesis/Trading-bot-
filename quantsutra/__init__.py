"""quantsutra -- analysis and signal engine for Indian index derivatives.

Covers NIFTY, BANKNIFTY, SENSEX and their futures and options, with an
indicator library, pattern recognition, a regime classifier, a Black-Scholes
options layer, a confluence signal engine, Indian transaction costs, and a
backtester that models all of the above.

Quick start::

    from quantsutra import analyze
    from quantsutra.data import YahooFeed

    bars = YahooFeed().history("NIFTY", "1d", 500)
    result = analyze("NIFTY", bars)
    print(result["recommendation"]["summary"])

This is analysis software, not investment advice, and it does not predict
prices. See ``quantsutra.analysis.DISCLAIMER``.
"""

__version__ = "0.1.0"

from .analysis import DISCLAIMER, AnalysisConfig, analyze
from .constants import Action, Bias, Direction, Regime, Timeframe

__all__ = [
    "DISCLAIMER",
    "Action",
    "AnalysisConfig",
    "Bias",
    "Direction",
    "Regime",
    "Timeframe",
    "__version__",
    "analyze",
]
