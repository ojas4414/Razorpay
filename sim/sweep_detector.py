"""Detector operating points across calibration percentiles.

The percentile is the one knob: a lower cut flags more sessions, catching more
recoverable abandonment but spending more discount on sessions that cannot
respond. There is no single right answer here -- the right point is whichever
maximises net margin, which needs the intervention's cost, so this table is the
input to that decision rather than the decision itself.

Run: python -m sim.sweep_detector
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.detector import Detector, calibrate
from sim.run import run_sessions

PERCENTILES = (50, 60, 70, 75, 85, 90)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def sweep(n_sessions=2000, calib_seed=7, eval_seed=42, percentiles=PERCENTILES):
    calib_sim, calib_events = run_sessions(n_sessions, calib_seed, collect_events=True)

    print(f"\ncalibration seed {calib_seed}, evaluation seed {eval_seed}, "
          f"{n_sessions:,} sessions each\n")
    print(f"{'pctile':>7} {'flagged':>8} {'prec':>7} {'recall':>7} "
          f"{'rec.recall':>11} {'rec.prec':>9} {'wasted':>7}")

    rows = []
    for percentile in percentiles:
        thresholds = calibrate(calib_events, calib_sim.sessions, percentile=percentile)
        sim, _ = run_sessions(n_sessions, eval_seed, detector=Detector(thresholds))

        carted = [s for s in sim.sessions if s.reached_cart()]
        flagged = [s for s in carted if s.flag]
        true_pos = sum(1 for s in flagged if s.outcome == Outcome.ABANDONED)
        abandoned = sum(1 for s in carted if s.outcome == Outcome.ABANDONED)
        recoverable = [s for s in carted
                       if s.outcome == Outcome.ABANDONED and s.recoverable]
        found = sum(1 for s in recoverable if s.flag)

        row = {
            "percentile": percentile,
            "flagged": len(flagged),
            "precision": _rate(true_pos, len(flagged)),
            "recall": _rate(true_pos, abandoned),
            "recoverable_recall": _rate(found, len(recoverable)),
            "recoverable_precision": _rate(found, len(flagged)),
            "wasted": len(flagged) - found,
        }
        rows.append(row)
        print(f"{percentile:>7} {row['flagged']:>8} {row['precision']:>7.1%} "
              f"{row['recall']:>7.1%} {row['recoverable_recall']:>11.1%} "
              f"{row['recoverable_precision']:>9.1%} {row['wasted']:>7}")

    print("\n  rec.recall  = share of price-driven abandonment caught (the upside)")
    print("  rec.prec    = share of treatments that can do anything (the cost)")
    print("  wasted      = treatments that cannot help, whatever the discount")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=2000)
    parser.add_argument("--calib-seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=42)
    args = parser.parse_args()
    sweep(args.sessions, args.calib_seed, args.eval_seed)


if __name__ == "__main__":
    main()
