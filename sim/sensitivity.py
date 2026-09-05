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
from sim.parallel import run_cells, worker_count
from sim.run import run_sessions

CALIBRATION_SEED = 7
# Several evaluation seeds per cell, pooled. One run at this size has a
# standard deviation of about 1.7 percentage points -- established by
# sim/robustness.py -- so a single-run sensitivity cell cannot tell "this
# constant matters" from "this run was noisy", which is the only question the
# analysis exists to answer. The first version of this file made exactly that
# mistake and reported five sign flips that were indistinguishable from noise.
EVALUATION_SEEDS = (42, 101, 202, 303, 404)
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


def measure(n_sessions, seeds=EVALUATION_SEEDS):
    """Calibrate once, then pool several evaluation seeds into one estimate.

    Calibration depends on the world, not on the seed being measured, so it is
    done once per cell rather than once per seed -- which is both correct and
    the difference between a sixteen-minute sweep and an hour.
    """
    calib, events = run_sessions(n_sessions, CALIBRATION_SEED,
                                 collect_events=True, budget_aware=True)
    thresholds = calibrate(events, calib.sessions)
    calib.sessions.clear()

    treat_hits = treat_n = control_hits = control_n = 0
    for seed in seeds:
        policy = InterventionPolicy(discount_rate=DISCOUNT, dispatch_latency=LATENCY)
        sim, _ = run_sessions(n_sessions, seed,
                              detector=Detector(thresholds), interventions=policy,
                              budget_aware=True)
        flagged = [s for s in sim.sessions if s.flag]
        treated = [s for s in flagged if s.arm == TREATMENT]
        control = [s for s in flagged if s.arm == CONTROL]
        converted = lambda g: sum(1 for s in g if s.outcome == Outcome.CONVERTED)
        treat_hits += converted(treated)
        treat_n += len(treated)
        control_hits += converted(control)
        control_n += len(control)
        sim.sessions.clear()

    return ProportionTest(treat_hits, treat_n, control_hits, control_n)


def _cell(spec):
    """One perturbed experiment, run in its own interpreter.

    Because this is a separate process, setting the constant here cannot affect
    the parent or any sibling. The sequential version had to restore every value
    in a finally block; this cannot forget to.
    """
    module_path, attribute, factor, n_sessions = spec
    if module_path is not None:
        module = importlib.import_module(module_path)
        setattr(module, attribute, getattr(module, attribute) * factor)
    test = measure(n_sessions)
    low, high = test.confidence_interval
    return {"delta": test.delta, "z": test.z,
            "treat_n": test.treat_n, "control_n": test.control_n,
            "ci_low": low, "ci_high": high, "significant": test.significant}


def run(n_sessions=DEFAULT_SESSIONS, perturbation=PERTURBATION, workers=None):
    # Baseline plus two cells per parameter, all independent, all parallel.
    cells = [(None, None, 1.0, n_sessions)]
    for module_path, attribute, _ in PARAMETERS:
        cells.append((module_path, attribute, 1 - perturbation, n_sessions))
        cells.append((module_path, attribute, 1 + perturbation, n_sessions))

    workers_used = worker_count(workers)
    print(f"{len(cells)} cells across {workers_used} workers "
          f"({n_sessions:,} sessions x {len(EVALUATION_SEEDS)} seeds each)")
    done = [0]

    def progress(index, cell, result):
        done[0] += 1
        print(f"  [{done[0]:>2}/{len(cells)}] finished", flush=True)

    outcomes = run_cells(_cell, cells, workers=workers, on_result=progress)

    baseline = outcomes[0]
    print(f"\nBaseline effect {baseline['delta']:+.2%}  z={baseline['z']:+.2f}  "
          f"n={baseline['treat_n']}/{baseline['control_n']}")
    print(f"\nPerturbing each constant by +/-{perturbation:.0%}\n")
    print(f"{'parameter':<38} {'low':>9} {'high':>9} {'swing':>9} {'sign':>6}")

    rows = []
    for index, (module_path, attribute, label) in enumerate(PARAMETERS):
        low_result = outcomes[1 + index * 2]
        high_result = outcomes[2 + index * 2]
        low, high = low_result["delta"], high_result["delta"]
        swing = abs(high - low)
        # A sign change is only meaningful if the negative side is actually
        # resolved away from zero. An interval straddling zero is an
        # underpowered cell, not evidence that the constant matters.
        def negative_and_resolved(result):
            return result["delta"] <= 0 and result["ci_high"] < 0

        holds = not (negative_and_resolved(low_result)
                     or negative_and_resolved(high_result))
        rows.append({
            "parameter": f"{module_path}.{attribute}",
            "label": label,
            "baseline_value": getattr(importlib.import_module(module_path), attribute),
            "low_delta": round(low, 5),
            "high_delta": round(high, 5),
            "low_z": round(low_result["z"], 3),
            "high_z": round(high_result["z"], 3),
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
        print(f"  the effect turns significantly negative under "
              f"{len(flipped)} perturbation(s):")
        for row in flipped:
            print(f"    {row['label']}  ({row['low_delta']:+.2%} / {row['high_delta']:+.2%})")
        print("  -> the headline depends on these being right.")
    else:
        print(f"  the effect never turns significantly negative under any "
              f"{perturbation:.0%} perturbation tested.")
        print("  -> it is a property of the mechanism, not of the calibration.")
    print("=" * 62)

    return {"baseline": baseline, "perturbation": perturbation,
            "sessions": n_sessions, "workers": workers_used,
            "parameters": rows, "robust": not flipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--perturbation", type=float, default=PERTURBATION)
    parser.add_argument("--out", default="results/sensitivity.json")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel workers (default: cores minus headroom)")
    args = parser.parse_args()

    result = run(args.sessions, args.perturbation, args.workers)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    # Required on Windows, where each worker re-imports this module: without the
    # guard the pool would recursively spawn itself.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
