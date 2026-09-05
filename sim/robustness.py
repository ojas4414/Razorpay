"""Does the recovery effect survive a change of seed?

Everything published so far rests on one evaluation seed. For a project whose
argument is measurement discipline, that is the obvious hole: a single
significant run is exactly what you would expect to find occasionally even when
nothing is happening, and picking the seed that produced it would be the
subtlest way to fool yourself in this whole codebase.

So the same experiment runs across many seeds and three things are reported:

**The spread.** Per-seed deltas, with how many individually reach significance.
Individual seeds are underpowered on purpose -- what matters is whether the
effect sits consistently on one side of zero, not whether any single run clears
a threshold.

**The pooled estimate.** All arms combined into one large sample, which is the
properly powered number and the one worth quoting.

**A bootstrap interval.** Resampling sessions directly, which makes no normal
approximation. If it disagrees with the analytic interval, the analytic one was
leaning on an assumption it should not have.

The A/A arm is run across the same seeds. A null on one seed is weak evidence;
a null that holds across twenty, while the treated arm does not, is the thing
that makes the result believable.

Thresholds are calibrated once on a held-out seed and reused, because they are a
property of the model rather than of the run being measured -- recalibrating per
seed would leak the evaluation data into the detector.

Run: python -m sim.robustness --seeds 15
"""
import argparse
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.analysis import ProportionTest
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from sim.run import run_sessions

CALIBRATION_SEED = 7
DEFAULT_SEEDS = 15
DEFAULT_SESSIONS = 10000
FIRST_EVAL_SEED = 100      # deliberately far from the calibration seed

BOOTSTRAP_ROUNDS = 2000


def _outcomes(sessions, triggers=None):
    """(arm, converted) for every flagged session in the compared population."""
    rows = []
    for s in sessions:
        if s.flag is None:
            continue
        if triggers is not None and s.flag.trigger not in triggers:
            continue
        if s.arm in (TREATMENT, CONTROL):
            rows.append((s.arm, s.outcome == Outcome.CONVERTED))
    return rows


def _test(rows):
    treat = [c for a, c in rows if a == TREATMENT]
    control = [c for a, c in rows if a == CONTROL]
    return ProportionTest(sum(treat), len(treat), sum(control), len(control))


def bootstrap_interval(rows, rounds=BOOTSTRAP_ROUNDS, rng=None):
    """Percentile interval on the delta, resampling sessions with replacement.

    Assumes nothing about the sampling distribution, unlike the z-interval.
    """
    rng = rng or random.Random(12345)
    treat = [c for a, c in rows if a == TREATMENT]
    control = [c for a, c in rows if a == CONTROL]
    if not treat or not control:
        return (0.0, 0.0)

    deltas = []
    for _ in range(rounds):
        t = sum(rng.choice(treat) for _ in range(len(treat))) / len(treat)
        c = sum(rng.choice(control) for _ in range(len(control))) / len(control)
        deltas.append(t - c)
    deltas.sort()
    return (deltas[int(0.025 * rounds)], deltas[int(0.975 * rounds)])


def run_arm(seeds, n_sessions, discount, latency, thresholds, label):
    """One arm of the robustness check -- treated, or A/A with nothing applied."""
    per_seed, pooled = [], []
    print(f"\n--- {label} ---")
    print(f"{'seed':>6} {'n_t':>6} {'n_c':>6} {'treat':>8} {'ctrl':>8} {'delta':>9} {'z':>7}")

    for seed in seeds:
        policy = InterventionPolicy(discount_rate=discount, dispatch_latency=latency)
        sim, _ = run_sessions(n_sessions, seed, detector=Detector(thresholds),
                              interventions=policy, budget_aware=True)
        rows = _outcomes(sim.sessions)
        pooled.extend(rows)
        test = _test(rows)
        per_seed.append({
            "seed": seed, "delta": test.delta, "z": test.z,
            "treat_n": test.treat_n, "control_n": test.control_n,
            "significant": test.significant,
        })
        print(f"{seed:>6} {test.treat_n:>6} {test.control_n:>6} "
              f"{test.treat_rate:>8.2%} {test.control_rate:>8.2%} "
              f"{test.delta:>+9.2%} {test.z:>+7.2f}")
        # Free the sessions before the next seed; only the outcomes are needed.
        sim.sessions.clear()

    return per_seed, pooled


def summarise(label, per_seed, pooled):
    deltas = [r["delta"] for r in per_seed]
    positive = sum(1 for d in deltas if d > 0)
    significant = sum(1 for r in per_seed if r["significant"])
    test = _test(pooled)
    boot_low, boot_high = bootstrap_interval(pooled)
    ci_low, ci_high = test.confidence_interval

    print(f"\n  {label}")
    print(f"    per-seed mean delta    {statistics.mean(deltas):>+8.2%}")
    print(f"    per-seed sd            {statistics.pstdev(deltas):>8.2%}")
    print(f"    seeds with delta > 0   {positive:>5} of {len(deltas)}")
    print(f"    seeds individually sig {significant:>5} of {len(deltas)}")
    print(f"    POOLED delta           {test.delta:>+8.2%}   n={test.treat_n}/{test.control_n}")
    print(f"    pooled 95% CI (z)      {ci_low:>+8.2%} to {ci_high:+.2%}")
    print(f"    pooled 95% CI (boot)   {boot_low:>+8.2%} to {boot_high:+.2%}")
    print(f"    pooled z               {test.z:>+8.2f}   "
          f"{'SIGNIFICANT' if test.significant else 'not significant'}")

    return {
        "label": label,
        "seeds": len(deltas),
        "mean_delta": round(statistics.mean(deltas), 5),
        "sd_delta": round(statistics.pstdev(deltas), 5),
        "seeds_positive": positive,
        "seeds_significant": significant,
        "pooled": test.summary(),
        "bootstrap_ci": [round(boot_low, 5), round(boot_high, 5)],
        "per_seed": [{k: (round(v, 5) if isinstance(v, float) else v)
                      for k, v in r.items()} for r in per_seed],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--discount", type=float, default=0.15)
    parser.add_argument("--latency", type=float, default=0.5)
    parser.add_argument("--out", default="robustness.json")
    args = parser.parse_args()

    seeds = list(range(FIRST_EVAL_SEED, FIRST_EVAL_SEED + args.seeds))

    print(f"Calibrating once on seed {CALIBRATION_SEED} "
          f"({args.sessions:,} sessions); thresholds are reused for every "
          f"evaluation seed so the eval data never reaches the detector.")
    calib, events = run_sessions(args.sessions, CALIBRATION_SEED,
                                 collect_events=True, budget_aware=True)
    thresholds = calibrate(events, calib.sessions)
    print(f"  {thresholds.describe()}")
    calib.sessions.clear()

    ab_seeds, ab_pooled = run_arm(seeds, args.sessions, args.discount,
                                  args.latency, thresholds,
                                  f"A/B — {args.discount:.0%} discount")
    aa_seeds, aa_pooled = run_arm(seeds, args.sessions, 0.0,
                                  args.latency, thresholds,
                                  "A/A — nothing applied")

    print("\n" + "=" * 62)
    ab = summarise(f"A/B ({args.discount:.0%} discount)", ab_seeds, ab_pooled)
    aa = summarise("A/A (nothing applied)", aa_seeds, aa_pooled)
    print("=" * 62)

    verdict = ab["pooled"]["significant"] and not aa["pooled"]["significant"]
    print(f"\n  VERDICT: {'holds' if verdict else 'DOES NOT HOLD'} — "
          f"treated effect {'is' if ab['pooled']['significant'] else 'is not'} "
          f"significant pooled, A/A {'is' if aa['pooled']['significant'] else 'is not'}.")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"ab": ab, "aa": aa, "sessions_per_seed": args.sessions,
                   "discount": args.discount, "latency": args.latency,
                   "holds": verdict}, handle, indent=2)
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
