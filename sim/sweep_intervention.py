"""How the recovery effect responds to discount size and dispatch latency.

A single headline number would hide the two things that actually govern whether
this works: how much is offered, and how fast it arrives. A discount that lands
after the shopper has gone is worth nothing no matter how large, and a discount
that arrives instantly is worth little if it is too small to move the decision.

The extreme rows are diagnostics rather than proposals. A 90% discount is not a
business, but if it fails to move the outcome then the mechanism is broken
rather than merely weak -- so it is worth running to tell those two cases apart.

Run: python -m sim.sweep_intervention
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from sim.run import run_sessions

GRID = [
    (0.15, 3.0),
    (0.15, 0.5),
    (0.30, 3.0),
    (0.30, 0.5),
    (0.50, 0.5),
    (0.90, 0.5),   # diagnostic: is the mechanism alive at all?
]


def _rate(hits, total):
    return hits / total if total else 0.0


def run_cell(thresholds, discount, latency, n_sessions, eval_seed):
    policy = InterventionPolicy(discount_rate=discount, dispatch_latency=latency)
    sim, _ = run_sessions(n_sessions, eval_seed,
                          detector=Detector(thresholds), interventions=policy)
    flagged = [s for s in sim.sessions if s.flag]
    treated = [s for s in flagged if s.arm == TREATMENT]
    control = [s for s in flagged if s.arm == CONTROL]
    converted = lambda group: sum(1 for s in group if s.outcome == Outcome.CONVERTED)

    stats = policy.stats()
    return {
        "discount": discount,
        "latency": latency,
        "treated": len(treated),
        "control": len(control),
        "treat_conv": _rate(converted(treated), len(treated)),
        "control_conv": _rate(converted(control), len(control)),
        "landed": stats["landed"],
        "missed": stats["missed"],
        "miss_rate": _rate(stats["missed"], stats["issued"]),
    }


def sweep(n_sessions=4000, calib_seed=7, eval_seed=42, grid=GRID):
    calib_sim, calib_events = run_sessions(n_sessions, calib_seed, collect_events=True)
    thresholds = calibrate(calib_events, calib_sim.sessions)

    print(f"\n{n_sessions:,} sessions, eval seed {eval_seed}\n")
    print(f"{'disc':>6} {'lat':>5} {'n_t':>5} {'n_c':>5} {'treat':>7} "
          f"{'ctrl':>7} {'delta':>8} {'missed':>7}")
    rows = []
    for discount, latency in grid:
        row = run_cell(thresholds, discount, latency, n_sessions, eval_seed)
        rows.append(row)
        delta = row["treat_conv"] - row["control_conv"]
        print(f"{discount:>6.0%} {latency:>5.1f} {row['treated']:>5} {row['control']:>5} "
              f"{row['treat_conv']:>7.1%} {row['control_conv']:>7.1%} "
              f"{delta:>+8.2%} {row['miss_rate']:>7.0%}")

    print("\n  delta  = treatment conversion minus control, among flagged sessions")
    print("  missed = nudges that arrived after the session had already ended")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=4000)
    parser.add_argument("--calib-seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=42)
    args = parser.parse_args()
    sweep(args.sessions, args.calib_seed, args.eval_seed)


if __name__ == "__main__":
    main()
