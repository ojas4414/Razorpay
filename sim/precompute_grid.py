"""Precompute the discount and latency response curves for the dashboard.

Two things need this. The sliders have to respond instantly -- a live simulation
takes tens of seconds, which is fine for a deliberate "run this" but hopeless for
dragging a control. And the break-even curve needs margin measured at enough
discount rates to show where it crosses zero, which is the single most useful
number a payments team could take from this project: below some rate the
programme pays for itself, above it you recover carts and lose money.

Two one-dimensional sweeps rather than a full grid, because the sliders move
independently and a full grid costs quadratically for information nobody reads.

Every cell is a real measured experiment -- same calibration discipline, same
arms, same statistics. Nothing here is interpolated or modelled.

Run: python -m sim.precompute_grid
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.analysis import ProportionTest, net_margin
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from sim.run import run_sessions

CALIBRATION_SEED = 7
EVALUATION_SEED = 42
DEFAULT_SESSIONS = 5000

DISCOUNTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
LATENCIES = [0.5, 1.5, 3.0, 6.0]
BASE_DISCOUNT = 0.15
BASE_LATENCY = 0.5


def measure(thresholds, discount, latency, n_sessions, eval_seed):
    policy = InterventionPolicy(discount_rate=discount, dispatch_latency=latency)
    sim, _ = run_sessions(n_sessions, eval_seed, detector=Detector(thresholds),
                          interventions=policy, budget_aware=True)
    flagged = [s for s in sim.sessions if s.flag]
    treated = [s for s in flagged if s.arm == TREATMENT]
    control = [s for s in flagged if s.arm == CONTROL]
    converted = lambda g: sum(1 for s in g if s.outcome == Outcome.CONVERTED)

    test = ProportionTest(converted(treated), len(treated),
                          converted(control), len(control))
    margin = net_margin(treated, control, discount)
    stats = policy.stats()
    sim.sessions.clear()

    return {
        "discount": discount,
        "latency": latency,
        **test.summary(),
        "margin": margin.summary(),
        "issued": stats["issued"],
        "landed": stats["landed"],
        "missed": stats["missed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--out", default="visuals/counterfactual_grid.json")
    args = parser.parse_args()

    print(f"Calibrating on seed {CALIBRATION_SEED} ({args.sessions:,} sessions)...")
    calib, events = run_sessions(args.sessions, CALIBRATION_SEED,
                                 collect_events=True, budget_aware=True)
    thresholds = calibrate(events, calib.sessions)
    print(f"  {thresholds.describe()}")
    calib.sessions.clear()

    print(f"\ndiscount sweep at {BASE_LATENCY}s dispatch")
    print(f"{'disc':>6} {'delta':>9} {'z':>7} {'net':>12} {'verdict':>9}")
    by_discount = []
    for discount in DISCOUNTS:
        row = measure(thresholds, discount, BASE_LATENCY, args.sessions, EVALUATION_SEED)
        by_discount.append(row)
        print(f"{discount:>6.0%} {row['delta']:>+9.2%} {row['z']:>+7.2f} "
              f"{row['margin']['net']:>12,.2f} "
              f"{'positive' if row['margin']['margin_positive'] else 'NEGATIVE':>9}")

    print(f"\nlatency sweep at {BASE_DISCOUNT:.0%} discount")
    print(f"{'lat':>6} {'delta':>9} {'z':>7} {'missed':>8}")
    by_latency = []
    for latency in LATENCIES:
        row = measure(thresholds, BASE_DISCOUNT, latency, args.sessions, EVALUATION_SEED)
        by_latency.append(row)
        miss = row["missed"] / row["issued"] if row["issued"] else 0
        print(f"{latency:>6.1f} {row['delta']:>+9.2%} {row['z']:>+7.2f} {miss:>8.0%}")

    # Where net margin crosses zero, by linear interpolation between the two
    # measured points that straddle it.
    break_even = None
    for a, b in zip(by_discount, by_discount[1:]):
        na, nb = a["margin"]["net"], b["margin"]["net"]
        if (na > 0) != (nb > 0) and na != nb:
            span = nb - na
            break_even = a["discount"] + (b["discount"] - a["discount"]) * (-na / span)
            break

    payload = {
        "sessions_per_cell": args.sessions,
        "calibration_seed": CALIBRATION_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "base_discount": BASE_DISCOUNT,
        "base_latency": BASE_LATENCY,
        "by_discount": by_discount,
        "by_latency": by_latency,
        "break_even_discount": round(break_even, 4) if break_even else None,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if break_even:
        print(f"\n  net margin crosses zero at a discount of {break_even:.1%}")
    else:
        print("\n  net margin does not cross zero anywhere in the swept range")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
