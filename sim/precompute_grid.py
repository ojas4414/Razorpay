"""Response curves for discount rate and dispatch latency.

Two consumers. The dashboard sliders need answers instantly, and a live
simulation takes tens of seconds -- fine for a deliberate "measure this", useless
for dragging a control. And the break-even point, where net margin crosses zero,
needs enough discount rates measured to be drawn rather than asserted.

**Every point pools several seeds.** The first version measured one 5,000-session
run per point and produced a break-even of 34%, which later turned out to be
wrong: a properly pooled run at 40% -- supposedly well past break-even -- came
back solidly margin-positive. A single run at this size has a standard deviation
near 1.7 percentage points on the conversion delta alone, and net margin is
noisier still, so a curve built from single runs traces sampling error as much as
response.

**Margin carries a bootstrap interval.** Net margin depends on the rate
difference and on what the recovered baskets happened to be worth, so it varies
far more than the rate. Reporting it as a bare number made it look more precise
than it is; two policies with deltas of +6.70% and +6.69% once produced nets
1,961 apart. Where the interval straddles zero, the honest reading is that the
sign is unresolved -- including, possibly, at the break-even crossing itself.

Calibration is done once and shared: the world does not change across cells,
only the offer does, so recalibrating per cell would burn time without changing
a threshold.

Run: python -m sim.precompute_grid
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.analysis import ProportionTest, bootstrap_margin, net_margin
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from sim.parallel import run_cells, worker_count
from sim.run import run_sessions

CALIBRATION_SEED = 7
EVALUATION_SEEDS = (42, 101, 202, 303, 404)
DEFAULT_SESSIONS = 5000

DISCOUNTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
LATENCIES = [0.5, 1.5, 3.0, 6.0]
BASE_DISCOUNT = 0.15
BASE_LATENCY = 0.5


def _cell(spec):
    """One (discount, latency) point, pooled over seeds. Runs in its own process."""
    thresholds, discount, latency, n_sessions, seeds = spec

    treat_hits = treat_n = control_hits = control_n = 0
    issued = landed = missed = 0
    treated_all, control_all = [], []

    for seed in seeds:
        policy = InterventionPolicy(discount_rate=discount, dispatch_latency=latency)
        sim, _ = run_sessions(n_sessions, seed, detector=Detector(thresholds),
                              interventions=policy, budget_aware=True)
        flagged = [s for s in sim.sessions if s.flag]
        treated = [s for s in flagged if s.arm == TREATMENT]
        control = [s for s in flagged if s.arm == CONTROL]
        converted = lambda g: sum(1 for s in g if s.outcome == Outcome.CONVERTED)

        treat_hits += converted(treated)
        treat_n += len(treated)
        control_hits += converted(control)
        control_n += len(control)
        # Keep only what the margin bootstrap reads, not whole Session objects.
        treated_all += [_Lite(s) for s in treated]
        control_all += [_Lite(s) for s in control]

        stats = policy.stats()
        issued += stats["issued"]
        landed += stats["landed"]
        missed += stats["missed"]
        sim.sessions.clear()

    test = ProportionTest(treat_hits, treat_n, control_hits, control_n)
    margin = net_margin(treated_all, control_all, discount)
    low, high = bootstrap_margin(treated_all, control_all, discount, rounds=800)

    return {
        "discount": discount,
        "latency": latency,
        **test.summary(),
        "margin": margin.summary(),
        "margin_ci_low": round(low, 2),
        "margin_ci_high": round(high, 2),
        "margin_sign_resolved": (low > 0) or (high < 0),
        "issued": issued, "landed": landed, "missed": missed,
        "seeds": len(seeds),
    }


class _Lite:
    """The three fields net_margin needs, without holding a whole Session."""

    __slots__ = ("outcome", "cart_value", "discount")

    def __init__(self, session):
        self.outcome = session.outcome
        self.cart_value = session.cart_value
        self.discount = session.discount


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", default="visuals/counterfactual_grid.json")
    args = parser.parse_args()

    print(f"Calibrating once on seed {CALIBRATION_SEED} "
          f"({args.sessions:,} sessions); the world is the same in every cell.")
    calib, events = run_sessions(args.sessions, CALIBRATION_SEED,
                                 collect_events=True, budget_aware=True)
    thresholds = calibrate(events, calib.sessions)
    print(f"  {thresholds.describe()}")
    calib.sessions.clear()

    cells = [(thresholds, d, BASE_LATENCY, args.sessions, EVALUATION_SEEDS)
             for d in DISCOUNTS]
    cells += [(thresholds, BASE_DISCOUNT, l, args.sessions, EVALUATION_SEEDS)
              for l in LATENCIES]

    used = worker_count(args.workers)
    print(f"\n{len(cells)} cells across {used} workers "
          f"({args.sessions:,} sessions x {len(EVALUATION_SEEDS)} seeds each)")
    done = [0]

    def progress(index, cell, result):
        done[0] += 1
        print(f"  [{done[0]:>2}/{len(cells)}] done", flush=True)

    results = run_cells(_cell, cells, workers=args.workers, on_result=progress)
    by_discount = results[:len(DISCOUNTS)]
    by_latency = results[len(DISCOUNTS):]

    print(f"\ndiscount sweep at {BASE_LATENCY}s dispatch")
    print(f"{'disc':>6} {'delta':>9} {'z':>7} {'net':>12} {'margin 95% CI':>26} {'sign':>9}")
    for row in by_discount:
        ci = f"{row['margin_ci_low']:>11,.0f} to {row['margin_ci_high']:>10,.0f}"
        print(f"{row['discount']:>6.0%} {row['delta']:>+9.2%} {row['z']:>+7.2f} "
              f"{row['margin']['net']:>12,.2f} {ci:>26} "
              f"{'resolved' if row['margin_sign_resolved'] else 'unresolved':>9}")

    print(f"\nlatency sweep at {BASE_DISCOUNT:.0%} discount")
    print(f"{'lat':>6} {'delta':>9} {'z':>7} {'missed':>8}")
    for row in by_latency:
        miss = row["missed"] / row["issued"] if row["issued"] else 0
        print(f"{row['latency']:>6.1f} {row['delta']:>+9.2%} {row['z']:>+7.2f} "
              f"{miss:>8.0%}")

    # Only trust a crossing between two points whose margin sign is resolved.
    break_even = None
    for a, b in zip(by_discount, by_discount[1:]):
        na, nb = a["margin"]["net"], b["margin"]["net"]
        if (na > 0) != (nb > 0) and na != nb and a["margin_sign_resolved"] \
                and b["margin_sign_resolved"]:
            break_even = a["discount"] + (b["discount"] - a["discount"]) * (-na / (nb - na))
            break

    payload = {
        "sessions_per_cell": args.sessions,
        "seeds_per_cell": len(EVALUATION_SEEDS),
        "calibration_seed": CALIBRATION_SEED,
        "base_discount": BASE_DISCOUNT, "base_latency": BASE_LATENCY,
        "by_discount": by_discount, "by_latency": by_latency,
        "break_even_discount": round(break_even, 4) if break_even else None,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if break_even:
        print(f"\n  net margin crosses zero at a discount of {break_even:.1%}, "
              f"between two points whose sign is resolved")
    else:
        unresolved = sum(1 for r in by_discount if not r["margin_sign_resolved"])
        print(f"\n  no crossing between resolved points "
              f"({unresolved} of {len(by_discount)} margin signs unresolved)")
        print("  -> the break-even discount is not established by this sweep")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
