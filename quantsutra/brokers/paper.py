"""Paper broker.

Forward testing on live prices with imaginary money.  This is the step between
a backtest and real capital, and it is the one most often skipped -- a backtest
cannot show you that you will not follow the system, and paper trading can.

Fills are modelled honestly: market orders cross the assumed spread against
you, limit orders only fill when price actually trades through them, and every
fill is charged the full Indian cost stack.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field

from ..constants import IST
from ..risk.costs import CostModel
from .base import Fill, Order, OrderStatus, OrderType, Position

__all__ = ["PaperBroker"]


@dataclass
class PaperBroker:
    """An in-memory broker for forward testing."""

    starting_capital: float = 500_000.0
    exchange: str = "NSE"
    slippage_pct: float = 0.004       # half-spread crossed on a market order
    name: str = "paper"

    cash: float = field(init=False)
    _positions: dict = field(default_factory=dict, init=False)
    orders: dict = field(default_factory=dict, init=False)
    fills: list = field(default_factory=list, init=False)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1), init=False)
    _costs: CostModel = field(init=False)

    def __post_init__(self):
        self.cash = self.starting_capital
        self._costs = CostModel(exchange=self.exchange, include_slippage=False)

    # -- orders -------------------------------------------------------------
    def place_order(self, order: Order, market_price: float | None = None) -> Order:
        order.order_id = order.order_id or f"P{next(self._ids):06d}"
        order.created_at = order.created_at or dt.datetime.now(IST)
        self.orders[order.order_id] = order

        if order.quantity <= 0:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "quantity must be positive"
            return order

        if order.order_type == OrderType.MARKET:
            if market_price is None:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "a market order needs a market_price to fill against"
                return order
            self._fill(order, self._crossed(market_price, order.side))
        return order

    def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.status != OrderStatus.PENDING:
            return False
        order.status = OrderStatus.CANCELLED
        return True

    def _crossed(self, price: float, side: str) -> float:
        """Market orders pay the spread, always in the wrong direction."""
        adjustment = 1 + self.slippage_pct if side.upper() == "BUY" else 1 - self.slippage_pct
        return price * adjustment

    def on_price(self, symbol: str, price: float, high: float | None = None,
                 low: float | None = None) -> list[Fill]:
        """Advance the clock: mark positions and fill any triggered orders.

        ``high``/``low`` let a bar-driven loop fill limit and stop orders that
        traded during the bar rather than only at its close.
        """
        high = high if high is not None else price
        low = low if low is not None else price

        position = self._positions.get(symbol)
        if position:
            position.last_price = price

        filled: list[Fill] = []
        for order in list(self.orders.values()):
            if order.status != OrderStatus.PENDING or order.symbol != symbol:
                continue
            buy = order.side.upper() == "BUY"

            if order.order_type == OrderType.LIMIT and order.price is not None:
                if (buy and low <= order.price) or (not buy and high >= order.price):
                    filled.append(self._fill(order, order.price))
            elif order.order_type in (OrderType.SL, OrderType.SL_M) and order.trigger_price:
                triggered = (high >= order.trigger_price) if buy else (low <= order.trigger_price)
                if triggered:
                    # A stop becomes a market order once triggered, so it pays
                    # the spread -- this is why stops slip.
                    fill_price = (order.price if order.order_type == OrderType.SL and order.price
                                  else self._crossed(order.trigger_price, order.side))
                    filled.append(self._fill(order, fill_price))
        return filled

    def _fill(self, order: Order, price: float) -> Fill:
        charges = self._costs.option_leg(price, order.quantity, order.side).total
        signed = order.quantity if order.side.upper() == "BUY" else -order.quantity

        position = self._positions.get(order.symbol)
        if position is None:
            self._positions[order.symbol] = Position(
                symbol=order.symbol, quantity=signed, average_price=price,
                last_price=price, charges_paid=charges,
                opened_at=order.created_at or dt.datetime.now(IST),
            )
        else:
            if position.quantity * signed > 0:            # adding to the position
                total = position.quantity + signed
                position.average_price = (
                    (position.average_price * position.quantity + price * signed) / total
                )
                position.quantity = total
            else:                                          # reducing or reversing
                closing = min(abs(signed), abs(position.quantity))
                direction = 1 if position.quantity > 0 else -1
                position.realised_pnl += (price - position.average_price) * closing * direction
                position.quantity += signed
                if position.quantity == 0:
                    position.average_price = 0.0
                elif position.quantity * signed > 0:
                    position.average_price = price
            position.last_price = price
            position.charges_paid += charges

        self.cash -= signed * price + charges
        order.status = OrderStatus.FILLED

        fill = Fill(order.order_id, order.symbol, order.side, order.quantity, price,
                    charges, dt.datetime.now(IST))
        self.fills.append(fill)
        return fill

    # -- state ---------------------------------------------------------------
    def positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.quantity != 0]

    def funds(self) -> dict:
        open_positions = self.positions()
        unrealised = sum(p.unrealised_pnl for p in open_positions)
        realised = sum(p.realised_pnl for p in self._positions.values())
        charges = sum(p.charges_paid for p in self._positions.values())
        return {
            "starting_capital": round(self.starting_capital, 2),
            "cash": round(self.cash, 2),
            "realised_pnl": round(realised, 2),
            "unrealised_pnl": round(unrealised, 2),
            "charges_paid": round(charges, 2),
            "equity": round(self.cash + sum(p.last_price * p.quantity for p in open_positions), 2),
            "open_positions": len(open_positions),
            "fills": len(self.fills),
        }

    def flatten(self, prices: dict) -> list[Fill]:
        """Close everything at the given prices -- e.g. at the session close."""
        out = []
        for position in self.positions():
            price = prices.get(position.symbol, position.last_price)
            order = Order(symbol=position.symbol,
                          side="SELL" if position.is_long else "BUY",
                          quantity=abs(position.quantity), order_type=OrderType.MARKET)
            self.place_order(order, market_price=price)
            out.extend(self.fills[-1:])
        return out
