"""Building the product catalog from the raw review dump.

Separate from pricing because this is the one part that touches the disk.
pricing.price_of needs only a product ID, so the simulation imports that and
never pays for catalog loading; product_service needs a browsable list, so it
imports this.
"""
import json
from collections import Counter

from core.pricing import price_of

DEFAULT_CATEGORY = "beauty"
DEFAULT_STOCK = 100
DEFAULT_TOP_N = 500


def load_catalog(filepath: str, top_n: int = DEFAULT_TOP_N) -> list[dict]:
    """Return the ``top_n`` most-reviewed products as plain dicts.

    Ordered by review count, which is what makes the catalog a plausible
    storefront rather than an arbitrary slice. Returns an empty list when the
    dump is absent -- the raw dataset is gitignored, and a missing file should
    leave the service running with an empty catalog rather than refusing to boot.
    """
    counter: Counter = Counter()
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            for line in handle:
                asin = json.loads(line).get("parent_asin")
                if asin:
                    counter[asin] += 1
    except FileNotFoundError:
        return []

    return [
        {
            "product_id": asin,
            "name": asin,
            "category": DEFAULT_CATEGORY,
            "price": price_of(asin),
            "stock": DEFAULT_STOCK,
        }
        for asin, _ in counter.most_common(top_n)
    ]
