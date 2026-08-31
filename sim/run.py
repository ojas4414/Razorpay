"""Running the engine to a fixed number of completed sessions.

The engine yields events forever; everything downstream wants a bounded, seeded
sample. This is the one place that turns one into the other, so simulate.py and
the experiments cannot drift into running the simulation two different ways.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.engine import LiveSimulation


def run_sessions(n_sessions, seed, collect_events=False, **kwargs):
    """Run until ``n_sessions`` have terminated. Returns (simulation, events).

    ``events`` is empty unless ``collect_events`` is set: the engine keeps only
    a small rolling window for the dashboard, so anything that needs the full
    stream -- threshold calibration, for one -- has to accumulate it here.

    keep_sessions=None retains every session. These runs are bounded, so the
    list cannot grow without limit the way the live path's would.
    """
    kwargs.setdefault("keep_sessions", None)
    simulation = LiveSimulation.build(seed=seed, **kwargs)
    stream = simulation.iter_events()
    events = []

    while len(simulation.sessions) < n_sessions:
        event = next(stream)
        if collect_events:
            events.append(event)

    return simulation, events
