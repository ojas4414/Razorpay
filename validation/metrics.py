"""Behavioural validation against real e-commerce benchmarks.

Compares HOW simulated shoppers behave, not WHAT they buy: product choice is
personal, but attention span, conversion, abandonment and trend concentration
are measurable and comparable.

Everything here now works on Session objects. It previously worked on lists of
an agent's whole lifetime log, which it called a "session" -- so n was 50 for
every statistic, and "abandonment" meant "fraction of agents that never bought
anything". That is 1 minus conversion by construction and does not measure what
the industry cart-abandonment figure measures, which is why it used to be
reported unscored. With a real session and a cart-level denominator it is now a
scored benchmark like the others.
"""
import collections

import numpy as np

# ---------------------------------------------------------------------------
# Single source of truth for every benchmark range.
#
# All three of a metric's expressions -- the string written into the JSON, the
# numeric threshold in the verdict, and the printed label -- derive from here.
# They had previously drifted apart: conversion was reported against "2-5%",
# scored against 0.02-0.20, and printed as "2-20%", so a 16% result recorded a
# PASS against a stated 2-5% benchmark.
#
# If a run falls outside a range it fails. The range does not move to meet it.
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "session_length": {
        "tolerance": 4.0,
        "label": "within 4.0 items of the real mean",
        "note": (
            "The real figure is items per user sequence in the review data, "
            "which spans a user's whole review history rather than one visit. "
            "It is the closest real quantity available and is compared on that "
            "basis, but it is not a like-for-like session length."
        ),
    },
    "conversion_rate": {
        "low": 0.02,
        "high": 0.05,
        "label": "2-5%",
        "source": "industry standard e-commerce conversion rate",
        "denominator": "all sessions",
    },
    "abandonment_rate": {
        "low": 0.70,
        "high": 0.85,
        "label": "70-85%",
        "source": "industry cart abandonment (Baymard Institute)",
        "denominator": "sessions that created a cart",
        "note": (
            "Measured over cart-reaching sessions only. Browse-only visits are "
            "excluded, which is what the industry figure does and what makes "
            "this comparison legitimate."
        ),
    },
    "social_influence_coefficient": {
        "low": 0.20,
        "high": 0.40,
        "label": "0.20-0.40",
        "source": "power-law concentration, Amazon trend data",
    },
}


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def session_length_stats(lengths: list) -> dict:
    return {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }


def conversion_rate(sessions: list) -> float:
    """Fraction of all sessions that ended in a purchase."""
    if not sessions:
        return 0.0
    return sum(1 for s in sessions if s.outcome == "converted") / len(sessions)


def cart_abandonment_rate(sessions: list):
    """Fraction of cart-reaching sessions that did not convert.

    Returns None when nothing reached a cart, rather than a misleading zero.
    """
    carted = [s for s in sessions if s.reached_cart()]
    if not carted:
        return None
    converted = sum(1 for s in carted if s.outcome == "converted")
    return 1.0 - converted / len(carted)


def social_influence_coefficient(sessions: list, top_k: int = 5) -> float:
    """Share of all product interactions captured by the top_k products.

    Real Amazon shows power-law concentration during trend events.
    """
    items = [p for s in sessions for p in s.products]
    if not items:
        return 0.0
    counts = collections.Counter(items)
    top = sum(count for _, count in counts.most_common(top_k))
    return round(top / sum(counts.values()), 4)


def trending_products(sessions: list, top_k: int = 5) -> list:
    items = [p for s in sessions for p in s.products]
    return [
        {"product": product, "count": count}
        for product, count in collections.Counter(items).most_common(top_k)
    ]


def budget_sensitivity(sessions: list) -> dict:
    """Conversion rate per persona, and whether wallets order them correctly.

    This is now scored. It could not be before: simulate.py priced every product
    at 0.0 because its gateway lookup 404'd, so the wallet constraint never
    bound and the metric was measuring nothing -- it reported power buyers
    converting less often than average buyers and carried no verdict at all.
    """
    by_persona = collections.defaultdict(list)
    for session in sessions:
        by_persona[session.persona].append(session.outcome == "converted")

    rates = {
        persona: round(sum(outcomes) / len(outcomes), 4)
        for persona, outcomes in by_persona.items()
    }
    ordered = (
        rates.get("power_buyer", 0)
        >= rates.get("average_buyer", 0)
        >= rates.get("browser", 0)
    )
    return {
        "conversion_by_persona": rates,
        "expected_ordering": "power_buyer >= average_buyer >= browser",
        "verdict": _verdict(ordered),
    }


def abandonment_drivers(sessions: list) -> dict:
    """What actually caused cart abandonment, and how much of it is reachable.

    Only available because this is a simulation: the driver behind each lost
    session is latent, so a real deployment never observes it. It is the ceiling
    on what any price intervention can achieve, and reporting it is what keeps a
    recovery result honest -- an intervention that reached all of this would be
    one that had been built to succeed.
    """
    carted = [s for s in sessions if s.reached_cart() and s.outcome == "abandoned"]
    if not carted:
        return {}
    counts = collections.Counter(s.quit_driver for s in carted)
    recoverable = sum(1 for s in carted if s.recoverable)
    return {
        "cart_abandonment_sessions": len(carted),
        "by_driver": {
            str(driver): round(count / len(carted), 4)
            for driver, count in counts.most_common()
        },
        "recoverable_share": round(recoverable / len(carted), 4),
        "note": (
            "recoverable_share is the fraction of cart abandonment driven by "
            "price, which a discount can move. The remainder is driven by "
            "hesitation, churn or absent intent and cannot be."
        ),
    }


def full_validation_report(real_sessions: list, sim_sessions: list) -> dict:
    """Behavioural similarity report.

    ``real_sessions`` are item sequences from the review data; ``sim_sessions``
    are Session objects from the engine.
    """
    real_stats = session_length_stats([len(s) for s in real_sessions])
    sim_stats = session_length_stats([s.actions for s in sim_sessions])
    sim_conv = conversion_rate(sim_sessions)
    sim_abandon = cart_abandonment_rate(sim_sessions)
    sim_social = social_influence_coefficient(sim_sessions)

    length_bm = BENCHMARKS["session_length"]
    conv_bm = BENCHMARKS["conversion_rate"]
    aband_bm = BENCHMARKS["abandonment_rate"]
    social_bm = BENCHMARKS["social_influence_coefficient"]

    report = {
        "sessions_simulated": len(sim_sessions),
        "session_length": {
            "real": real_stats,
            "simulated": sim_stats,
            "tolerance": f"|mean difference| < {length_bm['tolerance']}",
            "note": length_bm["note"],
            "verdict": _verdict(
                abs(real_stats["mean"] - sim_stats["mean"]) < length_bm["tolerance"]
            ),
        },
        "conversion_rate": {
            "real_benchmark": conv_bm["label"],
            "benchmark_source": conv_bm["source"],
            "denominator": conv_bm["denominator"],
            "simulated": round(sim_conv, 4),
            "verdict": _verdict(conv_bm["low"] <= sim_conv <= conv_bm["high"]),
        },
        "abandonment_rate": {
            "real_benchmark": aband_bm["label"],
            "benchmark_source": aband_bm["source"],
            "denominator": aband_bm["denominator"],
            "note": aband_bm["note"],
            "simulated": None if sim_abandon is None else round(sim_abandon, 4),
            "verdict": (
                "NO_DATA" if sim_abandon is None
                else _verdict(aband_bm["low"] <= sim_abandon <= aband_bm["high"])
            ),
        },
        "social_influence_coefficient": {
            "description": "fraction of interactions captured by top 5 products",
            "real_benchmark": social_bm["label"],
            "benchmark_source": social_bm["source"],
            "simulated": sim_social,
            "verdict": _verdict(social_bm["low"] <= sim_social <= social_bm["high"]),
        },
        "budget_sensitivity": budget_sensitivity(sim_sessions),
        "abandonment_drivers": abandonment_drivers(sim_sessions),
    }

    _print_summary(report, real_stats, sim_stats, sim_conv, sim_abandon, sim_social)
    return report


def _print_summary(report, real_stats, sim_stats, sim_conv, sim_abandon, sim_social):
    conv_bm = BENCHMARKS["conversion_rate"]
    aband_bm = BENCHMARKS["abandonment_rate"]
    social_bm = BENCHMARKS["social_influence_coefficient"]

    def label(verdict):
        return {"PASS": "[PASS]", "FAIL": "[FAIL]"}.get(verdict, f"[{verdict}]")

    print("\n========== SHOPPING SIMULATION REPORT ==========")
    print(f"\n  Sessions simulated:   {report['sessions_simulated']:,}")

    print("\n  Session Length (actions per session)")
    print(f"    Real shoppers:      {real_stats['mean']:.1f}  (+/-{real_stats['std']:.1f})")
    print(f"    Simulated:          {sim_stats['mean']:.1f}  (+/-{sim_stats['std']:.1f})")
    print(f"    Tolerance:          {BENCHMARKS['session_length']['label']}")
    print(f"    {label(report['session_length']['verdict'])}")

    print("\n  Conversion Rate  (all sessions)")
    print(f"    Industry benchmark: {conv_bm['label']}  ({conv_bm['source']})")
    print(f"    Simulated:          {sim_conv:.1%}")
    print(f"    {label(report['conversion_rate']['verdict'])}")

    print("\n  Cart Abandonment  (sessions that created a cart)")
    print(f"    Industry benchmark: {aband_bm['label']}  ({aband_bm['source']})")
    if sim_abandon is not None:
        print(f"    Simulated:          {sim_abandon:.1%}")
    print(f"    {label(report['abandonment_rate']['verdict'])}")

    print("\n  Trend Concentration  (top 5 products share of traffic)")
    print(f"    Industry benchmark: {social_bm['label']}  ({social_bm['source']})")
    print(f"    Simulated:          {sim_social:.1%}")
    print(f"    {label(report['social_influence_coefficient']['verdict'])}")

    bs = report["budget_sensitivity"]
    print("\n  Conversion Rate By Shopper Type")
    for persona, name in (("power_buyer", "Power Buyers  "),
                          ("average_buyer", "Regular Buyers"),
                          ("browser", "Browsers      ")):
        if persona in bs["conversion_by_persona"]:
            print(f"    {name}  {bs['conversion_by_persona'][persona]:.1%}")
    print(f"    Expected:           {bs['expected_ordering']}")
    print(f"    {label(bs['verdict'])}")

    drivers = report.get("abandonment_drivers")
    if drivers:
        print("\n  Cart Abandonment Drivers  (simulation ground truth)")
        for driver, share in drivers["by_driver"].items():
            print(f"    {driver:<14}      {share:.1%}")
        print(f"    Recoverable by discount: {drivers['recoverable_share']:.1%}")

    print("\n=================================================\n")
