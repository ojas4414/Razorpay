"""Does the result survive the coefficients being wrong?

Every number this project publishes sits downstream of constants that were
chosen: how hard price pushes on the quit hazard, how sharply the reservation
price bites, what shipping costs. They were calibrated against published
benchmarks rather than fitted to make the intervention look good -- but
"calibrated honestly" is a claim about process, and the obvious challenge is
that the effect is an artefact of the particular numbers picked.

The answer to that challenge is not an argument, it is a measurement. Perturb
each constant by a fixed fraction in both directions, re-run the whole
experiment, and see whether the recovery effect survives. Reported as a tornado:
the parameters that move the result most, first.

Three ways to read the output.

**The effect keeps its sign everywhere.** Then it is a property of the mechanism
rather than of the calibration, which is the strong outcome.

**One parameter dominates.** Then the result really rests on that number, and
that dependence belongs in the write-up rather than in a footnote.

**The sign flips somewhere.** Then the honest headline is much weaker than
"+2.82%", and it is far better to find that here than to be asked about it.

Perturbing a benchmark-calibrated constant will usually push conversion or
abandonment out of its band -- that is expected and is not the question being
asked here, which is only whether the *effect* holds.

Run: python -m sim.sensitivity
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.analysis import ProportionTest
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from sim.run import run_sessions

CALIBRATION_SEED = 7
EVALUATION_SEED = 42
DEFAULT_SESSIONS = 5000
DISCOUNT = 0.15
LATENCY = 0.5
PERTURBATION = 0.20

# (module path, attribute, human label). Every constant the behavioural model
# rests on, including the ones added late -- the fee shock is exactly the kind of
# choice a reviewer should be able to see tested.
PARAMETERS = [
    ("core.friction", "B0", "baseline reluctance to quit"),
    ("core.friction", "B_PRICE", "price weight in the hazard"),
    ("core.friction", "B_DWELL", "hesitation weight"),
    ("core.friction", "B_REVISIT", "churn weight"),
    ("core.friction", "B_INTENT", "intent weight"),
    ("core.friction", "RESERVATION_SOFTNESS", "reservation band width"),
    ("core.friction", "CHECKOUT_INTENT", "willingness to reach checkout"),
    ("core.cart", "SHIPPING_FLAT", "flat shipping fee"),
    ("core.cart", "TAX_RATE", "tax rate"),
    ("live.engine", "BUDGET_WEIGHT", "budget awareness when picking products"),
]


def measure(n_sessions):
    """One full experiment: calibrate, run both arms, return the effect."""
    calib, events = run_sessions(n_sessions, CALIBRATION_SEED,
                                 collect_events=True, budget_aware=True)
    thresholds = calibrate(events, calib.sessions)
    calib.sessions.clear()

    policy = InterventionPolicy(discount_rate=DISCOUNT, dispatch_latency=LATENCY)
    sim, _ = run_sessions(n_sessions, EVALUATION_SEED,
                          detector=Detector(thresholds), interventions=policy,
                          budget_aware=True)
    flagged = [s for s in sim.sessions if s.flag]
    treated = [s for s in flagged if s.arm == TREATMENT]
    control = [s for s in flagged if s.arm == CONTROL]
    converted = lambda g: sum(1 for s in g if s.outcome == Outcome.CONVERTED)
    test = ProportionTest(converted(treated), len(treated),
                          converted(control), len(control))
    sim.sessions.clear()
    return test


def run(n_sessions=DEFAULT_SESSIONS, perturbation=PERTURBATION):
    print(f"Baseline ({n_sessions:,} sessions per run)...")
    baseline = measure(n_sessions)
    print(f"  effect {baseline.delta:+.2%}  z={baseline.z:+.2f}  "
          f"n={baseline.treat_n}/{baseline.control_n}\n")

    print(f"Perturbing each constant by +/-{perturbation:.0%}\n")
    print(f"{'parameter':<38} {'low':>9} {'high':>9} {'swing':>9} {'sign':>6}")

    rows = []
    for module_path, attribute, label in PARAMETERS:
        module = importlib.import_module(module_path)
        original = getattr(module, attribute)
        results = {}
        try:
            for direction, factor in (("low", 1 - perturbation),
                                      ("high", 1 + perturbation)):
                setattr(module, attribute, original * factor)
                results[direction] = measure(n_sessions)
        finally:
            setattr(module, attribute, original)

        low, high = results["low"].delta, results["high"].delta
        swing = abs(high - low)
        holds = (low > 0) and (high > 0)
        rows.append({
            "parameter": f"{module_path}.{attribute}",
            "label": label,
            "baseline_value": original,
            "low_delta": round(low, 5),
            "high_delta": round(high, 5),
            "low_z": round(results["low"].z, 3),
            "high_z": round(results["high"].z, 3),
            "swing": round(swing, 5),
            "sign_holds": holds,
        })
        print(f"{label:<38} {low:>+9.2%} {high:>+9.2%} {swing:>9.2%} "
              f"{'ok' if holds else 'FLIPS':>6}")

    rows.sort(key=lambda r: r["swing"], reverse=True)
    flipped = [r for r in rows if not r["sign_holds"]]

    print("\n" + "=" * 62)
    print("  most influential first:")
    for row in rows[:5]:
        print(f"    {row['label']:<38} swing {row['swing']:.2%}")
    print()
    if flipped:
        print(f"  the effect changes sign under {len(flipped)} perturbation(s):")
        for row in flipped:
            print(f"    {row['label']}  ({row['low_delta']:+.2%} / {row['high_delta']:+.2%})")
        print("  -> the headline depends on these being right.")
    else:
        print(f"  the effect keeps its sign under every {perturbation:.0%} "
              f"perturbation tested.")
        print("  -> it is a property of the mechanism, not of the calibration.")
    print("=" * 62)

    return {"baseline": baseline.summary(), "perturbation": perturbation,
            "sessions": n_sessions, "parameters": rows,
            "robust": not flipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--perturbation", type=float, default=PERTURBATION)
    parser.add_argument("--out", default="sensitivity.json")
    args = parser.parse_args()

    result = run(args.sessions, args.perturbation)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
