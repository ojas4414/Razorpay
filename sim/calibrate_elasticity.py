"""Fit price elasticity to a published cart-recovery benchmark.

The friction model was calibrated against abandonment and conversion bands but
never against how much a discount recovers. It came out at roughly 1% recovery,
where industry cart-recovery campaigns are commonly reported at around 10% of
targeted carts (figures usually quoted in a 5-15% band). That is a second real
quantity the model should reproduce, and fitting to it is the same kind of
calibration as fitting to the 70-85% abandonment band -- not a search for a
coefficient that makes the intervention look good.

The distinction matters, so the discipline is fixed here in advance:

* The **target is declared before the search**: recovery of 8-12% of would-be
  abandoned carts among flagged sessions, at a 15% discount.
* The **three existing benchmarks are hard constraints**. A configuration that
  hits the recovery target but pushes conversion or abandonment out of band is
  rejected, not accepted with an excuse.
* The search is over **one mechanism** -- how sharply the reservation price
  bites -- not over anything that touches outcomes directly. Two earlier
  searches produced nothing and both failures are worth recording. Sweeping the
  quit-hazard coefficient failed because conversion is gated on *reaching*
  checkout, so a lower hazard only bought a session more time to wander before
  hitting the action cap. Sweeping a smooth price drag on progression failed
  because a smooth channel moves in proportion to the price change: 15% off a
  basket shifted conversion by about a quarter of a point. Only a threshold --
  a price above which a shopper will not buy -- makes a discount matter, and it
  matters for exactly the shoppers it carries across that line.
* Whatever comes out is then measured on a **held-out seed** with a confidence
  interval, and an A/A check still has to report null.

recovery rate = (treat_conv - control_conv) / (1 - control_conv)
              = share of carts that would have been abandoned, but were not.

Run: python -m sim.calibrate_elasticity
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.friction as friction
from core.session import Outcome
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from sim.run import run_sessions

# Declared before the search. See module docstring.
RECOVERY_TARGET = (0.08, 0.12)
CONVERSION_BAND = (0.02, 0.05)
ABANDONMENT_BAND = (0.70, 0.85)
SESSION_LENGTH_REAL = 8.16
SESSION_LENGTH_TOLERANCE = 4.0

DISCOUNT = 0.15
LATENCY = 0.5

# (B_ADVANCE, ADVANCE_SCALE). B_ADVANCE is how hard price drags on progression --
# the channel a discount works through. Suppressing progression lowers
# conversion, so agent advance probabilities are scaled back up to compensate
# and keep conversion inside its band.
# (RESERVATION_SOFTNESS, ADVANCE_SCALE). Softness is the width of the band
# around the reservation price, as a fraction of it; a discount can only move
# shoppers sitting inside that band. Blocking progression lowers conversion, so
# advance probabilities are scaled up to hold conversion inside its band.
# (RESERVATION_SOFTNESS, ADVANCE_SCALE). Softness is the width of the band
# around the reservation price, as a fraction of it; a discount can only move
# shoppers sitting inside that band. Making checkout intent price-driven raises
# conversion sharply, so advance probabilities are scaled back *down* to hold
# conversion inside its band.
GRID = [
    (0.10, 0.30),
    (0.18, 0.30),
    (0.18, 0.45),
    (0.25, 0.45),
    (0.25, 0.60),
    (0.35, 0.60),
]



# Persona advance probabilities are set in Agent.__init__, so the sweep scales
# them by patching the values the constructor reads back in.
_BASE_ADVANCE = {"power_buyer": 0.225, "average_buyer": 0.15, "browser": 0.04}


def _scale_advance(scale):
    """Rescale every persona's willingness to move down the funnel."""
    import agents.agent as agent_module

    agent_module.ADVANCE_PROB.update(
        {persona: min(0.95, base * scale) for persona, base in _BASE_ADVANCE.items()}
    )


def _rate(hits, total):
    return hits / total if total else 0.0


def measure(softness, advance_scale, n_sessions, calib_seed, eval_seed):
    friction.RESERVATION_SOFTNESS = softness
    _scale_advance(advance_scale)

    # Untreated run: the three standing benchmarks, plus detector calibration.
    base, events = run_sessions(n_sessions, calib_seed, collect_events=True)
    lengths = [s.actions for s in base.sessions]
    conversion = _rate(sum(1 for s in base.sessions
                           if s.outcome == Outcome.CONVERTED), len(base.sessions))
    abandonment = base.cart_abandonment_rate() or 0.0
    mean_length = sum(lengths) / len(lengths)

    thresholds = calibrate(events, base.sessions)

    # Treated run: what a discount actually recovers.
    policy = InterventionPolicy(discount_rate=DISCOUNT, dispatch_latency=LATENCY)
    sim, _ = run_sessions(n_sessions, eval_seed,
                          detector=Detector(thresholds), interventions=policy)
    flagged = [s for s in sim.sessions if s.flag]
    treated = [s for s in flagged if s.arm == TREATMENT]
    control = [s for s in flagged if s.arm == CONTROL]
    treat_conv = _rate(sum(1 for s in treated if s.outcome == Outcome.CONVERTED),
                       len(treated))
    control_conv = _rate(sum(1 for s in control if s.outcome == Outcome.CONVERTED),
                         len(control))
    recovery = (treat_conv - control_conv) / (1 - control_conv) if control_conv < 1 else 0.0

    return {
        "softness": softness, "advance_scale": advance_scale,
        "conversion": conversion,
        "abandonment": abandonment,
        "length": mean_length,
        "recovery": recovery,
        "n_treat": len(treated), "n_control": len(control),
        "ok_conversion": CONVERSION_BAND[0] <= conversion <= CONVERSION_BAND[1],
        "ok_abandonment": ABANDONMENT_BAND[0] <= abandonment <= ABANDONMENT_BAND[1],
        "ok_length": abs(SESSION_LENGTH_REAL - mean_length) < SESSION_LENGTH_TOLERANCE,
        "ok_recovery": RECOVERY_TARGET[0] <= recovery <= RECOVERY_TARGET[1],
    }


def sweep(n_sessions=6000, calib_seed=7, eval_seed=42, grid=GRID):
    print(f"\ntarget: recovery {RECOVERY_TARGET[0]:.0%}-{RECOVERY_TARGET[1]:.0%} "
          f"at a {DISCOUNT:.0%} discount, declared before the search")
    print(f"hard constraints: conversion {CONVERSION_BAND[0]:.0%}-{CONVERSION_BAND[1]:.0%}, "
          f"abandonment {ABANDONMENT_BAND[0]:.0%}-{ABANDONMENT_BAND[1]:.0%}, "
          f"length within {SESSION_LENGTH_TOLERANCE} of {SESSION_LENGTH_REAL}")
    print(f"{n_sessions:,} sessions per cell\n")
    print(f"{'soften':>8} {'adv_x':>7} | {'conv':>7} {'aband':>7} {'len':>6} "
          f"{'recovery':>9} | {'verdict':>9}")

    original = (friction.RESERVATION_SOFTNESS, dict(_BASE_ADVANCE))
    rows = []
    try:
        for softness, advance_scale in grid:
            row = measure(softness, advance_scale, n_sessions, calib_seed, eval_seed)
            rows.append(row)
            flags = "".join([
                "c" if not row["ok_conversion"] else ".",
                "a" if not row["ok_abandonment"] else ".",
                "l" if not row["ok_length"] else ".",
                "r" if not row["ok_recovery"] else ".",
            ])
            verdict = "ACCEPT" if flags == "...." else f"fail:{flags}"
            print(f"{softness:>8.2f} {advance_scale:>7.2f} | {row['conversion']:>7.2%} "
                  f"{row['abandonment']:>7.1%} {row['length']:>6.2f} "
                  f"{row['recovery']:>9.1%} | {verdict:>9}")
    finally:
        friction.RESERVATION_SOFTNESS = original[0]
        _BASE_ADVANCE.update(original[1])

    print("\n  fail codes: c=conversion a=abandonment l=session length r=recovery")
    accepted = [r for r in rows if r["ok_conversion"] and r["ok_abandonment"]
                and r["ok_length"] and r["ok_recovery"]]
    if accepted:
        best = min(accepted, key=lambda r: abs(r["recovery"] - 0.10))
        print(f"\n  ACCEPTED: B_ADVANCE={best['b_advance']}, "
              f"advance_scale={best['advance_scale']} "
              f"-> recovery {best['recovery']:.1%}")
    else:
        print("\n  no configuration satisfied every constraint")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=6000)
    parser.add_argument("--calib-seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=42)
    args = parser.parse_args()
    sweep(args.sessions, args.calib_seed, args.eval_seed)


if __name__ == "__main__":
    main()
