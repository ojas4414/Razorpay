"""The rules that decide whether an order goes through.

Extracted from order_service so the simulation applies exactly the same checks
the HTTP path does. The engine previously ran its own weaker version -- a bare
``agent.can_afford(price)`` against a randomly drawn price -- so an order the
service would have rejected could still succeed in the simulation.

Rejections are an enum whose *values* are the detail strings order_service has
always returned, so the HTTP contract is unchanged while the simulation gets a
machine-readable reason code. Those codes are what the recovery audit trail
records: "why did this session fail to convert" needs a symbol, not prose.
"""
from enum import Enum


class OrderRejection(str, Enum):
    """Why an order was refused. Values double as the HTTP detail string."""

    EMPTY_CART = "Cart is empty"
    PRODUCT_NOT_FOUND = "Product not found"
    OUT_OF_STOCK = "Product out of stock"
    INSUFFICIENT_BALANCE = "Insufficient balance"


def validate_order(cart: list, product: dict | None, balance: float) -> OrderRejection | None:
    """Return the reason this order cannot proceed, or None if it can.

    Checked in the order a shopper would hit them: an empty cart is a client
    mistake, a missing or unstocked product is a catalog problem, and only then
    does the wallet matter.
    """
    if not cart:
        return OrderRejection.EMPTY_CART
    if not product:
        return OrderRejection.PRODUCT_NOT_FOUND
    if product["stock"] <= 0:
        return OrderRejection.OUT_OF_STOCK
    if balance < product["price"]:
        return OrderRejection.INSUFFICIENT_BALANCE
    return None


def settle(balance: float, price: float) -> float:
    """Balance remaining after paying ``price``."""
    return round(balance - price, 2)
