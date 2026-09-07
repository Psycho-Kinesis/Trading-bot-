"""Broker protocol.

Deliberately minimal, and deliberately not implemented against any real broker.
An uncalibrated signal engine wired to live order placement loses money faster
than manual trading, not slower -- so the only implementation shipped here is
:class:`~quantsutra.brokers.paper.PaperBroker`.

If you add a real adapter, implement it against this protocol so the paper
broker stays a drop-in substitute for testing.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["Order", "Position", "Fill", "Broker", "OrderType", "OrderStatus"]


class OrderType:
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"            # stop-loss limit
    SL_M = "SL-M"        # stop-loss market


class OrderStatus:
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    symbol: str
    side: str                     # BUY | SELL
    quantity: int                 # units, not lots
    order_type: str = OrderType.MARKET
    price: float | None = None    # for LIMIT
    trigger_price: float | None = None  # for SL / SL-M
    product: str = "NRML"         # NRML | MIS (intraday)
    tag: str = ""
    order_id: str | None = None
    status: str = OrderStatus.PENDING
    created_at: dt.datetime | None = None
    reject_reason: str = ""


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    charges: float = 0.0
    timestamp: dt.datetime | None = None


@dataclass
class Position:
    symbol: str
    quantity: int                 # signed: positive long, negative short
    average_price: float
    last_price: float = 0.0
    realised_pnl: float = 0.0
    charges_paid: float = 0.0
    opened_at: dt.datetime | None = None
    meta: dict = field(default_factory=dict)

    @property
    def unrealised_pnl(self) -> float:
        return (self.last_price - self.average_price) * self.quantity

    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@runtime_checkable
class Broker(Protocol):
    name: str

    def place_order(self, order: Order) -> Order: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def positions(self) -> list[Position]: ...
    def funds(self) -> dict: ...
