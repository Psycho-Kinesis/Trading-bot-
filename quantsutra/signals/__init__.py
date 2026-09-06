"""Signal generation: rules, confluence engine and named playbooks."""

from .base import CATEGORY_CAPS, Category, MarketContext, RuleResult, Signal
from .engine import EngineConfig, SignalEngine, build_context
from .playbooks import PLAYBOOKS, Playbook, PlaybookMatch, match_playbooks
from .rules import ALL_RULES, evaluate_rules

__all__ = [
    "Signal", "RuleResult", "MarketContext", "Category", "CATEGORY_CAPS",
    "SignalEngine", "EngineConfig", "build_context",
    "ALL_RULES", "evaluate_rules",
    "Playbook", "PlaybookMatch", "PLAYBOOKS", "match_playbooks",
]
