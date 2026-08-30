"""Cart valuation.

Deliberately thin. It exists because the engine needs to know what a cart is
worth -- to charge for it, and later to discount it -- and that number has to be
derived the same way everywhere. Prices come from core.pricing, so a cart's
value is a pure function of the product IDs in it and is stable across runs.

Note this is *cart* semantics: the total of everything in the cart. It differs
from what order_service charges, which bills a single product per order and
treats the cart only as a non-empty precondition. That difference is the
service's existing behaviour and is left alone here; the simulation needs a
basket total because the recovery intervention discounts a basket.
"""
from core.pricing import price_of


def cart_total(cart: list[str]) -> float:
    """Total value of the products in ``cart``."""
    return round(sum(price_of(product_id) for product_id in cart), 2)
