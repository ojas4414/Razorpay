"""Deterministic product pricing.

The price of a product is a pure function of its ID: an md5 of the ASIN seeds a
private RNG, which draws once from a fixed range. Two consequences matter.

Stable across processes and runs -- unlike ``hash()``, md5 does not vary with
PYTHONHASHSEED, so the product_service, the simulation, and a re-run three days
from now all agree on what a thing costs. Reproducibility of the experiment
depends on that.

Defined for *every* product ID, not just catalogued ones. product_service builds
its catalog from the 500 most-reviewed ASINs, but the policy vocabulary is 2,985
products; pricing only what the catalog holds left 92.5% of the vocabulary
unpriced. Because this function needs nothing but the ID, the simulation can
price the whole vocabulary without loading a catalog at all.
"""
import hashlib
import random

# The range the original product_service drew from. Kept as named constants so
# the simulation can reason about the distribution without hardcoding it again.
PRICE_MIN = 10.0
PRICE_MAX = 500.0


def price_of(product_id: str) -> float:
    """Return the stable price for ``product_id``, rounded to two decimals."""
    seed = int(hashlib.md5(product_id.encode()).hexdigest(), 16) % (2 ** 32)
    return round(random.Random(seed).uniform(PRICE_MIN, PRICE_MAX), 2)
