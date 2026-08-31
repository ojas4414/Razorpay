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

# Extra costs added at checkout. Flat shipping plus proportional tax, which is
# the ordinary shape of them.
#
# These exist because "extra costs too high" is the most-cited real reason carts
# are abandoned, and because without them this model could not reproduce both
# halves of the problem at once. Budget-blind baskets landed at 2x what a
# shopper would pay, far past anything a discount could bridge. Budget-aware
# baskets landed under the line, so nobody abandoned over price at all. A fee
# revealed *after* the basket is assembled is what puts an affordable cart just
# over the threshold -- which is where a discount can carry it back.
# Calibrated jointly with budget-aware cart formation so that conversion lands
# in the 2-5% band and cart abandonment in 70-85%, while leaving the largest
# population a 15% discount can actually carry across the line (8.2% of flagged
# sessions). Heavier fees push abandonment up but move the bill so far past the
# reservation price that a discount can no longer bridge it: at 95 flat the
# crossable share halves to 3.8%, at 120 it falls to 2.4%.
SHIPPING_FLAT = 70.0
TAX_RATE = 0.08


def cart_total(cart: list[str]) -> float:
    """Total value of the products in ``cart``."""
    return round(sum(price_of(product_id) for product_id in cart), 2)


def checkout_total(subtotal: float) -> float:
    """What the shopper is finally asked to pay, fees included.

    Tax is proportional to the subtotal, so a discount reduces it; shipping is
    flat and is not discounted. That asymmetry is deliberate and it caps what an
    offer can do -- a discount cannot remove the fixed part of the bill.
    """
    if subtotal <= 0:
        return 0.0
    return round(subtotal * (1.0 + TAX_RATE) + SHIPPING_FLAT, 2)


def checkout_fees(subtotal: float) -> float:
    """The part of the bill that was not visible while shopping."""
    return round(checkout_total(subtotal) - subtotal, 2)
