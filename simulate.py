"""One-shot batch simulation -> validation_report.json.

This used to carry its own copy of the simulation loop, which drifted from the
one in live/engine.py that the dashboard runs. The two disagreed in ways that
mattered: this file priced products with ``GET {gateway}/products/{id}``, a
route the gateway does not expose, so every lookup 404'd, ``get_price()``
returned its 0.0 default and the budget constraint never bound in any published
run. Validating one simulator while demonstrating another is not a defensible
position, so there is now one engine and this is a driver for it.

It also no longer needs torch. Inference goes through the engine's ONNX path,
so a batch validation run costs ~30 MB instead of ~300 MB.
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from live.engine import LiveSimulation
from validation.metrics import full_validation_report, trending_products

DEFAULT_SESSIONS = 4000
DEFAULT_SEED = 42
REPORT_PATHS = ("validation_report.json", "visuals/validation_report.json")
RAW_REVIEWS = "data/raw/All_Beauty.jsonl"


def load_real_sessions(limit=50, window_size=3):
    """Item sequences from the review data, for the session-length comparison.

    Needs pandas, which is a development-only dependency -- imported here rather
    than at module scope so the rest of the file stays importable without it.
    """
    from data.loader import load_reviews, filter_, sort_split

    frame = load_reviews(RAW_REVIEWS)
    sorted_frame, _, _ = sort_split(filter_(frame))
    sequences = sorted_frame.groupby("user_id")["item_id"].apply(list).to_dict()
    return list(sequences.values())[:limit]


def run_simulation(n_sessions, seed):
    """Run the engine until ``n_sessions`` have terminated.

    keep_sessions=None retains every session: this run is bounded, and the
    report has to score all of them.
    """
    simulation = LiveSimulation.build(seed=seed, keep_sessions=None)
    events = simulation.iter_events()
    while len(simulation.sessions) < n_sessions:
        next(events)
    return simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS,
                        help=f"sessions to simulate (default {DEFAULT_SESSIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"run seed, for reproducibility (default {DEFAULT_SEED})")
    args = parser.parse_args()

    print(f"Running {args.sessions:,} sessions (seed {args.seed})...")
    simulation = run_simulation(args.sessions, args.seed)
    sessions = simulation.sessions
    print(f"  {len(sessions):,} sessions across {simulation.rounds} rounds, "
          f"{simulation.stats['total_events']:,} events")
    print(f"  policy backend: {simulation.policy.backend}")

    print("Loading real shopping sessions for comparison...")
    real_sessions = load_real_sessions()

    print("Running behavioural validation...")
    report = full_validation_report(real_sessions, sessions)

    # Raw data the dashboards read alongside the verdicts.
    report["seed"] = args.seed
    report["trending_products"] = trending_products(sessions)
    report["sim_session_lengths"] = [s.actions for s in sessions[:50]]
    report["real_session_lengths"] = [len(s) for s in real_sessions]

    for path in REPORT_PATHS:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(f"Report saved to {' and '.join(REPORT_PATHS)}")

    from visualize import run_all_visualizations

    run_all_visualizations(simulation.graph, sessions, real_sessions,
                           [a.persona for a in simulation.agents])
    print("Charts saved to visuals/")


if __name__ == "__main__":
    main()
