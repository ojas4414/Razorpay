"""The recovery experiment: treatment vs control, with statistics attached.

Design:

* Thresholds are calibrated on a **different seed** from the one measured.
* Sessions are randomised into arms **at flag time**, so the arms differ only in
  whether they were treated -- not in whether they were flagged.
* Only the flagged population is compared. Comparing flagged against unflagged
  would measure the detector, not the intervention.
* Comparison is like-for-like within a trigger population when targeting is
  narrowed. An earlier version compared price-shock *treated* sessions against
  *all* control sessions and produced a +6.17pp effect that was entirely
  selection bias: price-shock sessions convert at 13-18% in control against
  8% for the flagged population overall.

`--aa` runs the same machinery with nothing applied. It must report no
significant effect. That is the check that the harness is measuring rather than
manufacturing, and it is the only way to know the arms are genuinely comparable
given that treated sessions perturb the shared random stream.

Run: python -m sim.experiment
     python -m sim.experiment --aa
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.analysis import ProportionTest, net_margin, required_sample_size
from recovery.audit import AuditLog
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from sim.run import run_sessions

DEFAULT_SESSIONS = 30000
CALIBRATION_SEED = 7
EVALUATION_SEED = 42


def population(sessions, triggers):
    """Flagged sessions, restricted to the trigger set being treated."""
    flagged = [s for s in sessions if s.flag]
    if triggers is None:
        return flagged
    return [s for s in flagged if s.flag.trigger in triggers]


def run(n_sessions=DEFAULT_SESSIONS, discount=0.15, latency=0.5, triggers=None,
        calib_seed=CALIBRATION_SEED, eval_seed=EVALUATION_SEED, aa=False,
        audit_path="recovery_audit.db"):
    print(f"Calibrating on seed {calib_seed} ({n_sessions:,} sessions)...")
    calib_sim, calib_events = run_sessions(n_sessions, calib_seed, collect_events=True)
    thresholds = calibrate(calib_events, calib_sim.sessions)
    print(f"  {thresholds.describe()}")

    # An A/A run assigns arms exactly as usual but applies nothing, so any
    # difference it reports is machinery rather than treatment.
    effective_discount = 0.0 if aa else discount
    policy = InterventionPolicy(discount_rate=effective_discount,
                                dispatch_latency=latency,
                                eligible_triggers=triggers)

    label = "A/A (nothing applied)" if aa else f"{discount:.0%} discount, {latency}s latency"
    print(f"Running {label} on seed {eval_seed} ({n_sessions:,} sessions)...")
    sim, _ = run_sessions(n_sessions, eval_seed,
                          detector=Detector(thresholds), interventions=policy)

    pop = population(sim.sessions, triggers)
    treated = [s for s in pop if s.arm == TREATMENT]
    control = [s for s in pop if s.arm == CONTROL]
    converted = lambda group: sum(1 for s in group if s.outcome == Outcome.CONVERTED)

    test = ProportionTest(
        treat_hits=converted(treated), treat_n=len(treated),
        control_hits=converted(control), control_n=len(control),
    )
    margin = net_margin(treated, control, effective_discount)
    stats = policy.stats()

    print(f"\n=== {'A/A CHECK' if aa else 'RECOVERY EXPERIMENT'} ===")
    print(f"  sessions simulated       {len(sim.sessions):>8,}")
    print(f"  cart-reaching            {sim.stats['sessions_reached_cart']:>8,}")
    print(f"  flagged and in arms      {len(pop):>8,}")
    print(f"    treatment              {len(treated):>8,}")
    print(f"    control                {len(control):>8,}")
    print(f"  nudges issued            {stats['issued']:>8,}")
    print(f"    landed                 {stats['landed']:>8,}")
    print(f"    missed (too late)      {stats['missed']:>8,}   "
          f"({stats['missed'] / stats['issued']:.0%})" if stats['issued'] else "")
    print(f"    duplicates refused     {stats['duplicates_refused']:>8,}")

    print(f"\n  conversion, treatment    {test.treat_rate:>8.2%}")
    print(f"  conversion, control      {test.control_rate:>8.2%}")
    print(f"\n  {test.describe()}")

    if not aa:
        print(f"\n  --- net margin (synthetic order values; illustrative) ---")
        summary = margin.summary()
        print(f"  incremental conversions  {summary['incremental_conversions']:>10,.1f}")
        print(f"  gross recovered          {summary['gross_recovered']:>10,.2f}")
        print(f"  discount cost            {summary['discount_cost']:>10,.2f}")
        print(f"  NET                      {summary['net']:>10,.2f}   "
              f"{'positive' if summary['margin_positive'] else 'NEGATIVE'}")

    # Every decision is written to the append-only audit trail, and the
    # headline numbers are then recomputed from it. If the log and the
    # in-memory analysis disagree, one of them is wrong and it shows here
    # rather than staying silent.
    experiment_id = (f"{'aa' if aa else 'ab'}-n{n_sessions}-d{effective_discount}"
                     f"-l{latency}-s{eval_seed}")
    audit = AuditLog(audit_path)
    try:
        written = audit.record_many(pop, experiment_id, eval_seed)
        from_log = audit.outcomes_by_arm(experiment_id)
        print(f"\n  --- audit trail ({audit_path}) ---")
        print(f"  rows written             {written:>8,}   experiment {experiment_id}")
        for arm in sorted(from_log):
            entry = from_log[arm]
            rate = entry["converted"] / entry["n"] if entry["n"] else 0.0
            print(f"    {arm:<10} n={entry['n']:>6,}  converted={entry['converted']:>5,}"
                  f"  {rate:>7.2%}")
        logged_treat = from_log.get(TREATMENT, {"n": 0, "converted": 0})
        agrees = (logged_treat["n"] == test.treat_n
                  and logged_treat["converted"] == test.treat_hits)
        print(f"  reconciles with analysis {'yes' if agrees else 'NO -- INVESTIGATE'}")
    finally:
        audit.close()

    baseline = test.control_rate
    for target in (0.01, 0.02, 0.05):
        needed = required_sample_size(baseline, target)
        print(f"  to detect {target:.0%} at 80% power: {needed:,} per arm"
              if target == 0.01 else
              f"            {target:.0%}:              {needed:,} per arm")

    return {"test": test.summary(), "margin": margin.summary(), "policy": stats,
            "aa": aa, "discount": effective_discount, "latency": latency}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--discount", type=float, default=0.15)
    parser.add_argument("--latency", type=float, default=0.5)
    parser.add_argument("--triggers", default=None,
                        help="comma-separated flag triggers to treat, e.g. price_shock")
    parser.add_argument("--aa", action="store_true", help="run an A/A check")
    parser.add_argument("--out", default=None, help="write the result as JSON")
    parser.add_argument("--audit", default="recovery_audit.db",
                        help="SQLite path for the audit trail")
    args = parser.parse_args()

    triggers = frozenset(args.triggers.split(",")) if args.triggers else None
    result = run(args.sessions, args.discount, args.latency, triggers,
                 aa=args.aa, audit_path=args.audit)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
