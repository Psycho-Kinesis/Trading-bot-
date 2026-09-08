"""Signal generation: rules, confluence engine and named playbooks."""

from .base import CATEGORY_CAPS, Category, MarketContext, RuleResult, Signal
from .engine import EngineConfig, SignalEngine, build_context
from .playbooks import PLAYBOOKS, Playbook, PlaybookMatch, match_playbooks
from .rules import ALL_RULES, evaluate_rules

__all__ = [
    "ALL_RULES",
    "CATEGORY_CAPS",
    "PLAYBOOKS",
    "Category",
    "EngineConfig",
    "MarketContext",
    "Playbook",
    "PlaybookMatch",
    "RuleResult",
    "Signal",
    "SignalEngine",
    "build_context",
    "evaluate_rules",
    "match_playbooks",
]
