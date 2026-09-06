"""Signal vocabulary: the context a rule reads and the verdict it returns."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..constants import Action, Bias, Direction, score_to_bias
from ..regime import RegimeState

__all__ = ["RuleResult", "MarketContext", "Signal", "Category", "CATEGORY_CAPS"]


class Category:
    """Rule families.  Aggregation is capped per family (see CATEGORY_CAPS)."""

    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    STRUCTURE = "STRUCTURE"
    PATTERN = "PATTERN"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    LEVELS = "LEVELS"
    OPTIONS = "OPTIONS"
    CONTEXT = "CONTEXT"


# Maximum share of the total score any one family can contribute.
#
# This is the most important number in the engine.  Without it, eight
# moving-average rules all saying "up" would produce a 95% confidence reading
# that is really *one* piece of evidence counted eight times.  Structure and
# trend get the largest budgets because they are the most reliably informative;
# options positioning gets a modest one because OI is noisy intraday.
CATEGORY_CAPS: dict[str, float] = {
    Category.TREND: 0.22,
    Category.STRUCTURE: 0.20,
    Category.MOMENTUM: 0.15,
    Category.PATTERN: 0.13,
    Category.LEVELS: 0.12,
    Category.VOLUME: 0.08,
    Category.OPTIONS: 0.10,
    Category.VOLATILITY: 0.06,
    Category.CONTEXT: 0.06,
}


@dataclass
class RuleResult:
    """One rule's verdict.

    ``score`` is in [-1, +1] (negative = bearish).  ``weight`` is the rule's
    importance within its category.  ``confidence`` is how sure the rule is
    that its own inputs were meaningful -- a rule with insufficient data
    returns a confidence of 0 and is dropped rather than voting "neutral",
    which would otherwise dilute genuine signals.
    """

    name: str
    category: str
    score: float
    weight: float = 1.0
    confidence: float = 1.0
    rationale: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def effective(self) -> float:
        return float(np.clip(self.score, -1, 1)) * self.weight * self.confidence

    @property
    def bias(self) -> Bias:
        return score_to_bias(self.score)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "category": self.category,
            "score": round(float(self.score), 3), "weight": round(self.weight, 2),
            "confidence": round(self.confidence, 2), "bias": self.bias.value,
            "rationale": self.rationale,
        }


@dataclass
class MarketContext:
    """Everything the rules can see.

    Assembled once per evaluation so that no rule recomputes an indicator, and
    so a rule cannot accidentally look at a bar that has not closed yet.
    """

    symbol: str
    timeframe: str
    now: dt.datetime
    ohlcv: pd.DataFrame                 # raw bars, last row is the most recent CLOSED bar
    features: pd.DataFrame              # indicators.compute_all output
    snapshot: dict                      # last row of features, flattened
    regime: RegimeState
    spot: float
    atr: float

    # Optional enrichments -- rules must degrade gracefully when these are None.
    higher_tf: dict | None = None       # snapshot of a higher timeframe
    chain_summary: dict | None = None   # options chain summary
    india_vix: float | None = None
    iv_percentile: float | None = None
    expiry: dict | None = None          # expiry_context as a dict
    levels: list = field(default_factory=list)
    candles: list = field(default_factory=list)
    chart_patterns: list = field(default_factory=list)
    structure: object = None
    session_phase: str = "UNKNOWN"
    events: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def get(self, key: str, default=None):
        value = self.snapshot.get(key, default)
        if value is None:
            return default
        if isinstance(value, float) and not np.isfinite(value):
            return default
        return value

    def has(self, *keys: str) -> bool:
        return all(self.get(k) is not None for k in keys)


@dataclass
class Signal:
    """The engine's final output for one symbol at one point in time."""

    symbol: str
    timestamp: dt.datetime
    timeframe: str
    direction: Direction
    action: Action
    confidence: float                    # 0-100
    raw_score: float                     # -1..+1 before bucketing
    spot: float

    entry: float | None = None
    stop_loss: float | None = None
    targets: list[float] = field(default_factory=list)
    risk_reward: float | None = None

    contract: dict | None = None         # resolved option leg(s)
    strategy: dict | None = None         # payoff/greeks of the structure
    position: dict | None = None         # sizing

    regime: dict | None = None
    rules: list[RuleResult] = field(default_factory=list)
    category_scores: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.action != Action.NO_TRADE and not self.vetoes

    def top_reasons(self, n: int = 6) -> list[str]:
        ranked = sorted(self.rules, key=lambda r: -abs(r.effective))
        out = []
        for r in ranked[:n]:
            if abs(r.effective) < 0.02 or not r.rationale:
                continue
            arrow = "^" if r.score > 0 else ("v" if r.score < 0 else "-")
            out.append(f"[{arrow}] {r.rationale}")
        return out

    def to_dict(self, include_rules: bool = True) -> dict:
        out = {
            "symbol": self.symbol, "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe, "direction": self.direction.value,
            "action": self.action.value, "confidence": round(self.confidence, 1),
            "raw_score": round(self.raw_score, 3), "spot": round(self.spot, 2),
            "entry": round(self.entry, 2) if self.entry else None,
            "stop_loss": round(self.stop_loss, 2) if self.stop_loss else None,
            "targets": [round(t, 2) for t in self.targets],
            "risk_reward": self.risk_reward,
            "contract": self.contract, "strategy": self.strategy,
            "position": self.position, "regime": self.regime,
            "category_scores": {k: round(v, 3) for k, v in self.category_scores.items()},
            "reasons": self.reasons, "warnings": self.warnings, "vetoes": self.vetoes,
            "actionable": self.is_actionable,
        }
        if include_rules:
            out["rules"] = [r.to_dict() for r in self.rules if r.confidence > 0]
        return out
