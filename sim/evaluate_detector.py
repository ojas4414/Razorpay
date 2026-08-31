"""Score the detector against counterfactual ground truth.

Calibrate thresholds on one seed, then measure them on a different one. Tuning a
threshold on the sessions you then score it against is fitting to the test set,
and any precision it reported would be fiction.

The scoring is only possible because this is a simulation. No intervention runs
here, so every flagged session plays out untouched and we observe what it would
have done -- the road not taken. A production detector never sees that, which is
the argument for calibrating one here before shipping it there.

Two numbers matter more than precision:

**Lift over base rate.** Cart abandonment is ~81%, so a detector that flagged
every cart session would report 81% precision while carrying no information at
all. Precision is only meaningful as a ratio against that floor.

**Recall on recoverable abandonment.** Most abandonment is not price-driven and
a discount cannot move it (core/friction.py). Finding sessions no intervention
can help is not worth anything, so the detector is also scored on the subset
that could actually be recovered.

Run: python -m sim.evaluate_detector
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.detector import Detector, calibrate
from sim.run import run_sessions

CALIBRATION_SEED = 7
EVALUATION_SEED = 42
DEFAULT_SESSIONS = 4000


def confusion(sessions):
    """Flagged-vs-abandoned over cart-reaching sessions only."""
    carted = [s for s in sessions if s.reached_cart()]
    tp = sum(1 for s in carted if s.flag and s.outcome == Outcome.ABANDONED)
    fp = sum(1 for s in carted if s.flag and s.outcome == Outcome.CONVERTED)
    fn = sum(1 for s in carted if not s.flag and s.outcome == Outcome.ABANDONED)
    tn = sum(1 for s in carted if not s.flag and s.outcome == Outcome.CONVERTED)
    return carted, tp, fp, fn, tn


def _rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def evaluate(n_sessions=DEFAULT_SESSIONS, calib_seed=CALIBRATION_SEED,
             eval_seed=EVALUATION_SEED, percentile=75):
    print(f"Calibrating on seed {calib_seed} ({n_sessions:,} sessions)...")
    calib_sim, calib_events = run_sessions(n_sessions, calib_seed, collect_events=True)
    thresholds = calibrate(calib_events, calib_sim.sessions, percentile=percentile)
    print(f"  {thresholds.describe()}")

    print(f"Evaluating on seed {eval_seed} ({n_sessions:,} sessions)...")
    eval_sim, _ = run_sessions(n_sessions, eval_seed, detector=Detector(thresholds))
    sessions = eval_sim.sessions

    carted, tp, fp, fn, tn = confusion(sessions)
    flagged = tp + fp
    abandoned = tp + fn
    base_rate = _rate(abandoned, len(carted))
    precision = _rate(tp, flagged)
    recall = _rate(tp, abandoned)
    f1 = _rate(2 * precision * recall, precision + recall)

    recoverable = [s for s in carted
                   if s.outcome == Outcome.ABANDONED and s.recoverable]
    recoverable_found = sum(1 for s in recoverable if s.flag)

    latencies = sorted(s.ended_at - s.flag.sim_time
                       for s in carted if s.flag and s.ended_at is not None)

    print(f"\n=== DETECTOR, seed {eval_seed}, calibrated on seed {calib_seed} ===")
    print(f"  cart-reaching sessions   {len(carted):>6,}")
    print(f"  of which abandoned       {abandoned:>6,}   ({base_rate:.1%} base rate)")
    print(f"  flagged at risk          {flagged:>6,}   ({_rate(flagged, len(carted)):.1%} of cart sessions)")
    print("\n  confusion matrix (rows: flagged / not)")
    print(f"                  abandoned   converted")
    print(f"    flagged       {tp:>9,}   {fp:>9,}")
    print(f"    not flagged   {fn:>9,}   {tn:>9,}")
    print(f"\n  precision                {precision:>6.1%}")
    print(f"  base rate (flag-all)     {base_rate:>6.1%}")
    lift = (precision / base_rate) if base_rate else 0.0
    print(f"  LIFT over base rate      {lift:>6.2f}x   <- the number that means something")
    print(f"  recall                   {recall:>6.1%}")
    print(f"  F1                       {f1:>6.3f}")
    print(f"\n  recoverable abandonment  {len(recoverable):>6,}   (price-driven; a discount can move these)")
    print(f"  of those, flagged        {recoverable_found:>6,}   ({_rate(recoverable_found, len(recoverable)):.1%} recall on what matters)")
    print(f"  recoverable precision    {_rate(recoverable_found, flagged):>6.1%}   (share of discounts that can do anything)")
    print(f"  wasted treatments        {flagged - recoverable_found:>6,}   (discount spent where it cannot help)")
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        print(f"\n  flag-to-outcome latency  p50 {p50:>5.1f}s   p95 {p95:>5.1f}s")
        print("  (the window an intervention has to land in)")

    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "base_rate": base_rate,
        "lift": lift,
        "flagged": flagged,
        "cart_sessions": len(carted),
        "recoverable": len(recoverable),
        "recoverable_flagged": recoverable_found,
        "recoverable_precision": _rate(recoverable_found, flagged),
    }


def main():
    parser = argparse.ArgumentParser(description="Score the detector.")
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--calib-seed", type=int, default=CALIBRATION_SEED)
    parser.add_argument("--eval-seed", type=int, default=EVALUATION_SEED)
    parser.add_argument("--percentile", type=int, default=75)
    args = parser.parse_args()
    evaluate(args.sessions, args.calib_seed, args.eval_seed, args.percentile)


if __name__ == "__main__":
    main()
