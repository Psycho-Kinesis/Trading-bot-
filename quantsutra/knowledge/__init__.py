"""Historical market knowledge: curated events, calendar risk, analogues."""

from .analogues import find_analogues, nearest_historical_events, shape_distance
from .events import (
                     MARKET_EVENTS,
                     RECURRING_EVENTS,
                     MarketEvent,
                     event_risk,
                     events_between,
                     lessons_for_regime,
                     upcoming_events,
)

__all__ = [
                     "MARKET_EVENTS",
                     "RECURRING_EVENTS",
                     "MarketEvent",
                     "event_risk",
                     "events_between",
                     "find_analogues",
                     "lessons_for_regime",
                     "nearest_historical_events",
                     "shape_distance",
                     "upcoming_events",
]
