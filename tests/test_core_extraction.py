"""The core/ extraction must not have changed any behaviour.

These are the checks that justified moving pricing and order validation out of
the microservices. They compare core/ against reference copies of the code as it
was written inline in backend/product_service and backend/order_service, so a
future edit to core/ that changes what a product costs, or which orders are
refused, fails here rather than silently moving every published number.
"""
import hashlib
import itertools
import json
import random

from core.orders import OrderRejection, settle, validate_order
from core.pricing import price_of


def _original_price(asin):
    """product_service._deterministic_price, exactly as it was written inline."""
    seed = int(hashlib.md5(asin.encode()).hexdigest(), 16) % (2 ** 32)
    return round(random.Random(seed).uniform(10, 500), 2)


def _original_validate(cart, product, balance):
    """order_service.place_order's checks, in their original order."""
    if not cart:
        return "Cart is empty"
    if not product:
        return "Product not found"
    if product["stock"] <= 0:
        return "Product out of stock"
    if balance < product["price"]:
        return "Insufficient balance"
    return None


def _vocab():
    with open("vocab.json") as handle:
        return list(json.load(handle).keys())


class TestPricing:
    def test_matches_the_original_implementation(self):
        for asin in _vocab():
            assert price_of(asin) == _original_price(asin), asin

    def test_prices_the_entire_vocabulary(self):
        """The catalog covered 224 of 2,985 products; price_of needs only an ID.

        This is what closed the "budget constraint never binds" gap: pricing
        derived from the catalog left 92.5% of the policy's products unpriced,
        so can_afford() was answering against a missing value.
        """
        vocab = _vocab()
        assert len(vocab) > 2000
        assert all(price_of(asin) > 0 for asin in vocab)

    def test_is_stable_across_global_reseeding(self):
        """Prices must not move when the simulation seeds the global RNG."""
        asin = _vocab()[0]
        before = price_of(asin)
        random.seed(12345)
        assert price_of(asin) == before
        random.seed(999)
        assert price_of(asin) == before

    def test_is_deterministic_per_id(self):
        asin = _vocab()[0]
        assert price_of(asin) == price_of(asin)


class TestOrderRules:
    def test_matches_the_original_implementation(self):
        carts = [[], ["A"], ["A", "B"]]
        products = [
            None,
            {"stock": 0, "price": 50.0},
            {"stock": 5, "price": 50.0},
            {"stock": 5, "price": 0.0},
        ]
        balances = [0.0, 49.99, 50.0, 1000.0]

        for cart, product, balance in itertools.product(carts, products, balances):
            expected = _original_validate(cart, product, balance)
            actual = validate_order(cart, product, balance)
            assert (None if actual is None else actual.value) == expected

    def test_rejection_values_are_the_http_detail_strings(self):
        """order_service returns these verbatim, so changing one is an API break."""
        assert OrderRejection.EMPTY_CART.value == "Cart is empty"
        assert OrderRejection.PRODUCT_NOT_FOUND.value == "Product not found"
        assert OrderRejection.OUT_OF_STOCK.value == "Product out of stock"
        assert OrderRejection.INSUFFICIENT_BALANCE.value == "Insufficient balance"

    def test_checks_apply_in_the_original_order(self):
        """An empty cart outranks every other problem, as it always did."""
        assert validate_order([], None, 0.0) is OrderRejection.EMPTY_CART

    def test_settle(self):
        assert settle(100.0, 30.0) == 70.0
        assert settle(49.99, 12.34) == round(49.99 - 12.34, 2)
