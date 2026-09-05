"""Do the detector's thresholds survive a different world?

The sharpest criticism of this project is that it is a simulator measuring
itself. The complete answer is a real clickstream with view, cart and purchase
labels -- validate the simulation-calibrated detector on sessions it did not
generate. That data is not available here (the Amazon reviews on disk are
purchases, with no abandonment signal at all), so this is the strongest
available substitute rather than the real thing, and it is worth being exact
about what it does and does not show.

**What it tests.** Thresholds are calibrated in one world and evaluated in a
materially different one: shoppers who are more price-sensitive, a different fee
structure, a different persona mix, a different reservation band. If a detector
tuned on world A still separates abandonment from conversion in world B, its
thresholds are picking up something structural about the behaviour rather than
memorising one parameterisation.

**What it does not test.** Whether real shoppers behave like any of these worlds.
Every world here shares the same generative assumptions, so agreement across
them is evidence of robustness, not of realism. Nothing in this file licenses a
claim about live traffic.

The comparison that matters is against a detector calibrated *natively* in the
new world, which is the best any threshold could do there. Transfer loss is the
gap between them: small means thresholds move between worlds, large means they
were fitted to a particular set of coefficients.

Run: python -m sim.transfer
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.cart as cart_module
import core.friction as friction
from core.session import Outcome
from recovery.detector import Detector, calibrate
from sim.run import run_sessions

HOME_SEED = 7
EVAL_SEED = 42
DEFAULT_SESSIONS = 6000

# Worlds are defined as overrides on the calibrated constants. Each is a
# deliberate, sizeable departure -- not a nudge -- because a transfer test that
# only varies things slightly proves nothing.
WORLDS = {
    "home": {},
    "price_sensitive": {
        "friction.B_PRICE": 5.0,
        "friction.RESERVATION_SOFTNESS": 0.10,
    },
    "patient_shoppers": {
        "friction.B_DWELL": 0.10,
        "friction.B_REVISIT": 0.06,
        "friction.B0": -2.40,
    },
    "expensive_delivery": {
        "cart.SHIPPING_FLAT": 140.0,
        "cart.TAX_RATE": 0.18,
    },
    "impulsive": {
        "friction.CHECKOUT_INTENT": 0.90,
        "friction.B_INTENT": 1.40,
    },
}

MODULES = {"friction": friction, "cart": cart_module}


def apply_world(overrides):
    """Set a world's constants, returning what was there before."""
    previous = {}
    for path, value in overrides.items():
        module_name, attribute = path.split(".")
        module = MODULES[module_name]
        previous[path] = getattr(module, attribute)
        setattr(module, attribute, value)
    return previous


def restore(previous):
    for path, value in previous.items():
        module_name, attribute = path.split(".")
        setattr(MODULES[module_name], attribute, value)


def score(thresholds, n_sessions, seed):
    """Precision, recall and lift for these thresholds in the current world."""
    sim, _ = run_sessions(n_sessions, seed, detector=Detector(thresholds),
                          budget_aware=True)
    carted = [s for s in sim.sessions if s.reached_cart()]
    sim.sessions.clear()
    if not carted:
        return None

    tp = sum(1 for s in carted if s.flag and s.outcome == Outcome.ABANDONED)
    fp = sum(1 for s in carted if s.flag and s.outcome == Outcome.CONVERTED)
    fn = sum(1 for s in carted if not s.flag and s.outcome == Outcome.ABANDONED)
    abandoned = tp + fn
    flagged = tp + fp
    base_rate = abandoned / len(carted)
    precision = tp / flagged if flagged else 0.0
    recall = tp / abandoned if abandoned else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "lift": precision / base_rate if base_rate else 0.0,
        "base_rate": base_rate,
        "flagged": flagged,
        "cart_sessions": len(carted),
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
    }


def calibrate_in_world(n_sessions, seed):
    sim, events = run_sessions(n_sessions, seed, collect_events=True,
                               budget_aware=True)
    thresholds = calibrate(events, sim.sessions)
    sim.sessions.clear()
    return thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--out", default="transfer.json")
    args = parser.parse_args()

    print(f"Calibrating in the home world (seed {HOME_SEED}, "
          f"{args.sessions:,} sessions)...")
    home_thresholds = calibrate_in_world(args.sessions, HOME_SEED)
    print(f"  {home_thresholds.describe()}\n")

    print(f"{'world':<20} {'lift(home θ)':>13} {'lift(native θ)':>15} "
          f"{'loss':>8} {'base':>7}")
    rows = []
    for name, overrides in WORLDS.items():
        previous = apply_world(overrides)
        try:
            transferred = score(home_thresholds, args.sessions, EVAL_SEED)
            native_thresholds = calibrate_in_world(args.sessions, HOME_SEED)
            native = score(native_thresholds, args.sessions, EVAL_SEED)
        finally:
            restore(previous)

        if not transferred or not native:
            continue
        loss = native["lift"] - transferred["lift"]
        rows.append({"world": name, "overrides": overrides,
                     "transferred": transferred, "native": native,
                     "lift_loss": round(loss, 4)})
        print(f"{name:<20} {transferred['lift']:>13.3f} {native['lift']:>15.3f} "
              f"{loss:>+8.3f} {transferred['base_rate']:>7.1%}")

    away = [r for r in rows if r["world"] != "home"]
    worst = max(away, key=lambda r: r["lift_loss"]) if away else None
    holds = all(r["transferred"]["lift"] > 1.0 for r in away)

    print("\n" + "=" * 68)
    if holds:
        print("  Home thresholds beat the base rate in every world tested.")
        print("  -> they track something structural, not one set of coefficients.")
    else:
        failed = [r["world"] for r in away if r["transferred"]["lift"] <= 1.0]
        print(f"  Home thresholds carry no information in: {', '.join(failed)}")
        print("  -> they are fitted to the home parameterisation.")
    if worst:
        print(f"  Largest transfer loss: {worst['world']} "
              f"({worst['lift_loss']:+.3f} lift vs native calibration)")
    print("\n  This is robustness across simulated worlds, not evidence about")
    print("  real shoppers. Every world here shares the same generative model.")
    print("=" * 68)

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"sessions": args.sessions, "worlds": rows,
                   "transfers": holds}, handle, indent=2)
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
