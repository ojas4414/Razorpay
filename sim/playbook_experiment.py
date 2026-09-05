"""Does choosing the action by diagnosis beat one discount for everyone?

Three policies, measured on the same seeds:

* **untreated** — the control arm, nothing applied.
* **fixed discount** — 15% to every flagged session, once. What the rest of this
  project measured.
* **playbook** — an action chosen from what the detector observed, escalating to
  a discount only if the free action landed and the session was still in
  trouble, capped at three attempts and bounded by the spend guard.

Two things are reported that a single-intervention experiment cannot produce.

**Diagnosis accuracy.** The playbook's guess at the cause is scored against the
true driver the simulation recorded. It never sees that field, so the confusion
matrix is a real measurement of how well observable signals stand in for the
thing that actually ended the session. A pipeline that diagnoses badly will
spend correctly on the wrong problem.

**Cost by action.** A reminder and a simplified checkout are operationally cheap;
a discount is not. Margin is therefore computed per session from the rungs
actually taken, rather than from one global discount rate -- which is the whole
point of reaching for the free thing first.

Everything is pooled across seeds. One 5,000-session run carries a standard
deviation near 1.7 percentage points, so a single-run comparison measures the
seed as much as the policy; four earlier claims in this project were withdrawn
for exactly that reason.

Run: python -m sim.playbook_experiment
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Outcome
from recovery.analysis import ProportionTest
from recovery.assign import CONTROL, TREATMENT
from recovery.detector import Detector, calibrate
from recovery.intervention import InterventionPolicy
from recovery.playbook import PlaybookPolicy, diagnose
from recovery.safety import SafetyGate
from sim.run import run_sessions

CALIBRATION_SEED = 7
EVAL_SEEDS = (42, 101, 202, 303, 404)
DEFAULT_SESSIONS = 6000
DISCOUNT = 0.15
LATENCY = 0.5


def _pooled(runs):
    """Combine per-seed arm counts into one estimate."""
    t_hits = sum(r["treat_hits"] for r in runs)
    t_n = sum(r["treat_n"] for r in runs)
    c_hits = sum(r["control_hits"] for r in runs)
    c_n = sum(r["control_n"] for r in runs)
    test = ProportionTest(t_hits, t_n, c_hits, c_n)
    gross = sum(r["gross"] for r in runs)
    cost = sum(r["cost"] for r in runs)
    return {
        "test": test,
        "gross": round(gross, 2),
        "cost": round(cost, 2),
        "net": round(gross - cost, 2),
        "actions": sum((collections.Counter(r["actions"]) for r in runs),
                       collections.Counter()),
        "escalations": sum(r["escalations"] for r in runs),
        "issued": sum(r["issued"] for r in runs),
        "diagnosis": sum((collections.Counter(r["diagnosis"]) for r in runs),
                         collections.Counter()),
    }


def _score(sim, cost_of, discount_rate):
    """Arm counts and margin for one run. ``cost_of`` bills a treated session."""
    flagged = [s for s in sim.sessions if s.flag]
    treated = [s for s in flagged if s.arm == TREATMENT]
    control = [s for s in flagged if s.arm == CONTROL]
    converted = lambda g: [s for s in g if s.outcome == Outcome.CONVERTED]

    t_hits, c_hits = len(converted(treated)), len(converted(control))
    t_n, c_n = len(treated), len(control)
    t_rate = t_hits / t_n if t_n else 0.0
    c_rate = c_hits / c_n if c_n else 0.0

    control_carts = [s.cart_value for s in converted(control)]
    average_cart = sum(control_carts) / len(control_carts) if control_carts else 0.0
    gross = (t_rate - c_rate) * t_n * average_cart
    cost = sum(cost_of(s) for s in converted(treated))

    return {
        "treat_hits": t_hits, "treat_n": t_n,
        "control_hits": c_hits, "control_n": c_n,
        "gross": gross, "cost": cost,
        "diagnosis": [(diagnose(s.flag.features), s.quit_driver)
                      for s in flagged if s.outcome == Outcome.ABANDONED],
    }


def run_fixed(thresholds, n_sessions, seed, discount):
    policy = InterventionPolicy(discount_rate=discount, dispatch_latency=LATENCY)
    sim, _ = run_sessions(n_sessions, seed, detector=Detector(thresholds),
                          interventions=policy)
    row = _score(sim, lambda s: s.cart_value * s.discount, discount)
    row.update(actions=["discount_15"] * policy.stats()["landed"],
               escalations=0, issued=policy.stats()["issued"])
    sim.sessions.clear()
    return row


def run_playbook(thresholds, n_sessions, seed):
    book = PlaybookPolicy(safety=SafetyGate())
    sim, _ = run_sessions(n_sessions, seed, detector=Detector(thresholds),
                          playbook=book)
    row = _score(sim, book.cost_for, 0.0)
    stats = book.stats()
    row.update(
        actions=[a.action.name for taken in book.attempts.values()
                 for a in taken if a.landed],
        escalations=stats["escalations"], issued=stats["issued"],
    )
    sim.sessions.clear()
    return row


def report(label, pooled):
    test = pooled["test"]
    low, high = test.confidence_interval
    print(f"\n  {label}")
    print(f"    actions issued           {pooled['issued']:>8,}   "
          f"({pooled['escalations']:,} escalations)")
    print(f"    conversion delta         {test.delta:>+8.2%}   "
          f"({'sig' if test.significant else 'not sig'}, z={test.z:+.2f}, "
          f"n={test.treat_n}/{test.control_n})")
    print(f"    95% CI                   {low:>+8.2%} to {high:+.2%}")
    print(f"    gross recovered          {pooled['gross']:>12,.2f}")
    print(f"    cost of actions          {pooled['cost']:>12,.2f}")
    print(f"    NET                      {pooled['net']:>12,.2f}   "
          f"{'positive' if pooled['net'] > 0 else 'NEGATIVE'}")
    if pooled["actions"]:
        mix = "  ".join(f"{k} {v}" for k, v in pooled["actions"].most_common())
        print(f"    actions landed           {mix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--discount", type=float, default=DISCOUNT)
    parser.add_argument("--out", default="results/playbook.json")
    args = parser.parse_args()

    print(f"Calibrating on seed {CALIBRATION_SEED} ({args.sessions:,} sessions)...")
    calib, events = run_sessions(args.sessions, CALIBRATION_SEED,
                                 collect_events=True)
    thresholds = calibrate(events, calib.sessions)
    print(f"  {thresholds.describe()}")
    calib.sessions.clear()

    print(f"\nMeasuring both policies over seeds {EVAL_SEEDS}...")
    fixed_runs, book_runs = [], []
    for seed in EVAL_SEEDS:
        fixed_runs.append(run_fixed(thresholds, args.sessions, seed, args.discount))
        book_runs.append(run_playbook(thresholds, args.sessions, seed))
        print(f"  seed {seed} done", flush=True)

    fixed = _pooled(fixed_runs)
    book = _pooled(book_runs)

    print("\n" + "=" * 66)
    report(f"fixed {args.discount:.0%} discount to everyone flagged", fixed)
    report("playbook — action chosen by diagnosis, escalating", book)

    print("\n  " + "-" * 62)
    print(f"    net difference           {book['net'] - fixed['net']:>+12,.2f}")
    print(f"    verdict                  "
          f"{'playbook earns more' if book['net'] > fixed['net'] else 'fixed discount earns more'}")

    # How good is the diagnosis? Scored against a field it never sees.
    pairs = [p for r in book_runs for p in r["diagnosis"]]
    matrix = collections.Counter(pairs)
    drivers = sorted({d for _, d in pairs if d})
    guesses = sorted({g for g, _ in pairs})
    correct = sum(n for (g, d), n in matrix.items() if g == d)
    total = sum(matrix.values())

    print("\n  diagnosis vs the true driver (which it never sees)")
    header = "    " + f"{'guessed':<12}" + "".join(f"{d:>13}" for d in drivers)
    print(header)
    for g in guesses:
        row = "".join(f"{matrix.get((g, d), 0):>13,}" for d in drivers)
        print(f"    {g:<12}{row}")
    print(f"\n    agreement                {correct / total:.1%} of "
          f"{total:,} abandoned sessions")

    payload = {
        "sessions": args.sessions, "seeds": list(EVAL_SEEDS),
        "discount": args.discount,
        "fixed": {**fixed["test"].summary(), "net": fixed["net"],
                  "gross": fixed["gross"], "cost": fixed["cost"],
                  "issued": fixed["issued"]},
        "playbook": {**book["test"].summary(), "net": book["net"],
                     "gross": book["gross"], "cost": book["cost"],
                     "issued": book["issued"],
                     "escalations": book["escalations"],
                     "actions": dict(book["actions"])},
        "playbook_beats_fixed": book["net"] > fixed["net"],
        "diagnosis_agreement": round(correct / total, 4) if total else 0.0,
        "diagnosis_matrix": {f"{g}->{d}": n for (g, d), n in matrix.items()},
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
