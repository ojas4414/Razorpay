"""Does targeting treatability beat targeting risk?

The detector finds sessions about to abandon. Roughly two treatments in three
land on someone no discount could have moved, and that ratio barely shifts with
the threshold, because risk and treatability are different things. This measures
whether an uplift model does better.

The protocol, and every part of it matters:

1. Run an experiment on a **training seed**, treating everyone flagged. That
   produces an audit trail with both arms represented.
2. Fit the uplift model on that trail, and on nothing else.
3. Run again on a **held-out seed**, twice: once treating everyone flagged
   (the baseline), once treating only where the model says the lift beats the
   bill. Same seed, so the two policies face the same shoppers.
4. Compare treatments issued, effect, and net margin.

The model is then scored against ``recoverable`` -- whether each loss was
actually price-driven. It never sees that field, and no live system could; it is
the check that distinguishes a model that learned treatability from one that
memorised the outcome it was fitted on.

A warning about what this can and cannot show. Fewer, better-targeted treatments
should raise margin while *lowering* the headline conversion delta, because the
sessions dropped were ones the offer was not moving anyway. Reading a smaller
delta as a worse result would be exactly backwards, so both are reported.

Run: python -m sim.uplift_experiment
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.analysis import ProportionTest, net_margin
from recovery.assign import CONTROL, TREATMENT
from recovery.audit import to_row
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from recovery.uplift import UpliftModel, ground_truth_precision, qini_curve
from sim.run import run_sessions

CALIBRATION_SEED = 7
TRAIN_SEED = 21
EVAL_SEED = 42
DEFAULT_SESSIONS = 8000
DISCOUNT = 0.15
LATENCY = 0.5


def audit_rows(sessions, experiment_id):
    return [to_row(s, experiment_id) for s in sessions if s.flag is not None]


def evaluate(sessions, discount):
    flagged = [s for s in sessions if s.flag]
    treated = [s for s in flagged if s.arm == TREATMENT]
    control = [s for s in flagged if s.arm == CONTROL]
    converted = lambda g: sum(1 for s in g if s.outcome == Outcome.CONVERTED)
    test = ProportionTest(converted(treated), len(treated),
                          converted(control), len(control))
    margin = net_margin(treated, control, discount)
    offered = [s for s in treated if s.intervention is not None]
    useful = sum(1 for s in offered
                 if s.recoverable or s.outcome == Outcome.CONVERTED)
    return {
        "test": test,
        "margin": margin,
        "flagged": len(flagged),
        "offers": len(offered),
        "offers_that_could_matter": useful,
        "offer_precision": useful / len(offered) if offered else 0.0,
    }


def report(label, result):
    test, margin = result["test"], result["margin"]
    print(f"\n  {label}")
    print(f"    offers issued            {result['offers']:>8,}   "
          f"of {result['flagged']:,} flagged")
    print(f"    offers that could matter {result['offers_that_could_matter']:>8,}   "
          f"({result['offer_precision']:.1%})")
    print(f"    conversion delta         {test.delta:>+8.2%}   "
          f"({'sig' if test.significant else 'not sig'}, z={test.z:+.2f})")
    print(f"    gross recovered          {margin.gross_recovered:>12,.2f}")
    print(f"    discount cost            {margin.discount_cost:>12,.2f}")
    print(f"    NET                      {margin.net:>12,.2f}   "
          f"{'positive' if margin.positive else 'NEGATIVE'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--discount", type=float, default=DISCOUNT)
    parser.add_argument("--out", default="uplift_result.json")
    args = parser.parse_args()

    print(f"Calibrating detector on seed {CALIBRATION_SEED}...")
    calib, events = run_sessions(args.sessions, CALIBRATION_SEED,
                                 collect_events=True, budget_aware=True)
    thresholds = calibrate(events, calib.sessions)
    print(f"  {thresholds.describe()}")
    calib.sessions.clear()

    print(f"\nTraining run on seed {TRAIN_SEED} (treat everyone flagged)...")
    train_sim, _ = run_sessions(
        args.sessions, TRAIN_SEED, detector=Detector(thresholds),
        interventions=InterventionPolicy(discount_rate=args.discount,
                                         dispatch_latency=LATENCY),
        budget_aware=True)
    training = audit_rows(train_sim.sessions, "train")
    print(f"  {len(training):,} flagged sessions to learn from")

    model = UpliftModel.train(training)
    print(f"  model fitted on {model.trained_on:,} rows")

    print(f"\nHeld-out seed {EVAL_SEED}: baseline (treat all flagged)...")
    base_sim, _ = run_sessions(
        args.sessions, EVAL_SEED, detector=Detector(thresholds),
        interventions=InterventionPolicy(discount_rate=args.discount,
                                         dispatch_latency=LATENCY),
        budget_aware=True)
    baseline = evaluate(base_sim.sessions, args.discount)

    print(f"Held-out seed {EVAL_SEED}: uplift targeting...")
    targeted_policy = InterventionPolicy(
        discount_rate=args.discount, dispatch_latency=LATENCY,
        eligible=lambda s: model.should_treat(s.flag.features, args.discount),
    )
    uplift_sim, _ = run_sessions(
        args.sessions, EVAL_SEED, detector=Detector(thresholds),
        interventions=targeted_policy, budget_aware=True)
    targeted = evaluate(uplift_sim.sessions, args.discount)

    print("\n" + "=" * 64)
    report("baseline — treat every flagged session", baseline)
    report("uplift — treat only where lift beats the bill", targeted)

    saved = baseline["offers"] - targeted["offers"]
    net_change = targeted["margin"].net - baseline["margin"].net
    print("\n  " + "-" * 60)
    print(f"    offers avoided           {saved:>8,}   "
          f"({saved / baseline['offers']:.0%} fewer)" if baseline["offers"] else "")
    print(f"    change in net margin     {net_change:>+12,.2f}")
    print(f"    offer precision          {baseline['offer_precision']:.1%} "
          f"-> {targeted['offer_precision']:.1%}")

    truth = ground_truth_precision(model, audit_rows(base_sim.sessions, "eval"))
    if truth:
        print("\n  scored against ground truth the model never saw:")
        print(f"    recoverable share, top {truth['top_fraction']:.0%} of ranking"
              f"   {truth['recoverable_in_top']:.1%}")
        print(f"    recoverable share, the rest                "
              f"{truth['recoverable_in_rest']:.1%}")
        print(f"    baseline (all losses)                      {truth['baseline']:.1%}")
        verdict = truth["recoverable_in_top"] > truth["baseline"]
        print(f"    -> the ranking {'DOES' if verdict else 'does NOT'} concentrate "
              f"recoverable losses")
    print("=" * 64)

    curve = qini_curve(model, audit_rows(base_sim.sessions, "eval"))
    payload = {
        "sessions": args.sessions, "discount": args.discount,
        "baseline": {"offers": baseline["offers"],
                     **baseline["test"].summary(),
                     "margin": baseline["margin"].summary(),
                     "offer_precision": round(baseline["offer_precision"], 4)},
        "uplift": {"offers": targeted["offers"],
                   **targeted["test"].summary(),
                   "margin": targeted["margin"].summary(),
                   "offer_precision": round(targeted["offer_precision"], 4)},
        "offers_avoided": saved,
        "net_margin_change": round(net_change, 2),
        "ground_truth": truth,
        "qini": curve,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
