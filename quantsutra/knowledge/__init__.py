"""Historical market knowledge: curated events, calendar risk, analogues."""

from .analogues import find_analogues, nearest_historical_events, shape_distance
from .events import (MARKET_EVENTS, RECURRING_EVENTS, MarketEvent, event_risk,
                     events_between, lessons_for_regime, upcoming_events)

__all__ = [
    "MarketEvent", "MARKET_EVENTS", "RECURRING_EVENTS", "events_between",
    "event_risk", "upcoming_events", "lessons_for_regime",
    "find_analogues", "shape_distance", "nearest_historical_events",
]
