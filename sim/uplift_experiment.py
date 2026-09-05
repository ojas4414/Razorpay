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
# Several held-out seeds, pooled. A single run at this size has a standard
# deviation near 1.7 percentage points, so comparing two policies on one seed
# each measures the seeds rather than the policies -- which is what the first
# version of this comparison ended up doing.
EVAL_SEEDS = (42, 101, 202, 303, 404)
EVAL_SEED = 42
DEFAULT_SESSIONS = 8000
DISCOUNT = 0.15
LATENCY = 0.5


def audit_rows(sessions, experiment_id):
    return [to_row(s, experiment_id) for s in sessions if s.flag is not None]


def evaluate(sessions, discount, selector=None):
    """Compare arms over the population the policy would act on.

    ``selector`` must be applied to *both* arms. Without it the treatment arm
    contains sessions the policy declined to treat while the control arm
    contains everyone, so the comparison is between different populations and
    the effect is diluted by whatever share went untreated. This is the same
    like-for-like failure that made price-shock targeting look like +6.17% when
    it was +0.89%; it reappeared here because the eligibility predicate filters
    who gets an offer but not who is counted.
    """
    flagged = [s for s in sessions if s.flag]
    if selector is not None:
        flagged = [s for s in flagged if selector(s)]
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


def measure_policy(thresholds, model, discount, n_sessions, seeds,
                   targeted, selector):
    """Run one policy across several seeds and pool the arms.

    Both arms are restricted to ``selector`` -- the population the policy would
    act on -- because filtering only the treated side compares different
    populations and dilutes the effect by whatever share went untreated.
    """
    treat_hits = treat_n = control_hits = control_n = 0
    offers = useful = 0
    gross = cost = 0.0

    for seed in seeds:
        policy = InterventionPolicy(
            discount_rate=discount, dispatch_latency=LATENCY,
            eligible=(lambda s: model.should_treat(s.flag.features, discount))
            if targeted else None,
        )
        sim, _ = run_sessions(n_sessions, seed, detector=Detector(thresholds),
                              interventions=policy, budget_aware=True)
        result = evaluate(sim.sessions, discount, selector=selector)
        test = result["test"]
        treat_hits += test.treat_hits
        treat_n += test.treat_n
        control_hits += test.control_hits
        control_n += test.control_n
        offers += result["offers"]
        useful += result["offers_that_could_matter"]
        gross += result["margin"].gross_recovered
        cost += result["margin"].discount_cost
        sim.sessions.clear()

    return {
        "test": ProportionTest(treat_hits, treat_n, control_hits, control_n),
        "offers": offers,
        "offers_that_could_matter": useful,
        "offer_precision": useful / offers if offers else 0.0,
        "gross": round(gross, 2),
        "cost": round(cost, 2),
        "net": round(gross - cost, 2),
        "flagged": treat_n + control_n,
    }


def report_pooled(label, pooled):
    test = pooled["test"]
    print(f"\n  {label}")
    print(f"    offers issued            {pooled['offers']:>8,}")
    print(f"    offers that could matter {pooled['offers_that_could_matter']:>8,}   "
          f"({pooled['offer_precision']:.1%})")
    print(f"    conversion delta         {test.delta:>+8.2%}   "
          f"({'sig' if test.significant else 'not sig'}, z={test.z:+.2f}, "
          f"n={test.treat_n}/{test.control_n})")
    low, high = test.confidence_interval
    print(f"    95% CI                   {low:>+8.2%} to {high:+.2%}")
    print(f"    NET                      {pooled['net']:>12,.2f}   "
          f"{'positive' if pooled['net'] > 0 else 'NEGATIVE'}")


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
    parser.add_argument("--out", default="results/uplift_result.json")
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

    selector = lambda s: model.should_treat(s.flag.features, args.discount)
    seeds = EVAL_SEEDS

    print(f"\nHeld-out seeds {seeds}: treat everyone flagged...")
    baseline = measure_policy(thresholds, model, args.discount, args.sessions,
                              seeds, targeted=False, selector=None)

    print("Same seeds, restricted to the population uplift would pick...")
    baseline_on_selected = measure_policy(thresholds, model, args.discount,
                                          args.sessions, seeds, targeted=False,
                                          selector=selector)

    print("Same seeds, uplift targeting...")
    targeted = measure_policy(thresholds, model, args.discount, args.sessions,
                              seeds, targeted=True, selector=selector)

    print("\n" + "=" * 64)
    report_pooled("treat every flagged session", baseline)
    report_pooled("treat all of the population uplift picks "
                  "(the like-for-like control)", baseline_on_selected)
    report_pooled("uplift targeting", targeted)

    saved = baseline["offers"] - targeted["offers"]
    net_change = targeted["net"] - baseline_on_selected["net"]
    print("\n  " + "-" * 60)
    print(f"    offers avoided vs treat-all   {saved:>8,}")
    print(f"    net vs like-for-like control  {net_change:>+12,.2f}")
    print(f"    offer precision               {baseline['offer_precision']:.1%} "
          f"-> {targeted['offer_precision']:.1%}")
    # Two different questions, and conflating them is easy. The like-for-like
    # control isolates whether targeting helps *within* the population it picks.
    # The business question is whether the programme earns more overall, and
    # that is against treating everyone flagged -- restricting treatment throws
    # away every profitable offer it declines.
    print(f"    net vs treat-all              "
          f"{targeted['net'] - baseline['net']:>+12,.2f}")
    within = ("indistinguishable" if not targeted["test"].significant
              or not baseline_on_selected["test"].significant
              else "resolved")
    print(f"    within selected population    {within}")
    print(f"    OVERALL                       "
          f"{'uplift earns more' if targeted['net'] > baseline['net'] else 'treating everyone earns more'}")

    base_sim, _ = run_sessions(args.sessions, seeds[0],
                               detector=Detector(thresholds),
                               interventions=InterventionPolicy(
                                   discount_rate=args.discount,
                                   dispatch_latency=LATENCY),
                               budget_aware=True)

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
    flat = lambda p: {"offers": p["offers"], **p["test"].summary(),
                      "net": p["net"], "gross": p["gross"], "cost": p["cost"],
                      "offer_precision": round(p["offer_precision"], 4)}
    payload = {
        "sessions": args.sessions, "discount": args.discount,
        "seeds": list(seeds),
        "baseline": flat(baseline),
        "baseline_on_selected": flat(baseline_on_selected),
        "uplift": flat(targeted),
        "uplift_beats_treat_all": targeted["net"] > baseline["net"],
        "uplift_beats_like_for_like": targeted["net"] > baseline_on_selected["net"],
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
