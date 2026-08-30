"""Session accounting, reproducibility and the event contract.

Step 3 replaces the quit decision with a friction model, which is the single
most load-bearing change in the project: every published number is downstream of
it. These tests pin the invariants that must survive that rewrite -- that
sessions are conserved, that the cart denominator means what it says, that a
seed reproduces a run, and that the dashboard's fields are still there.
"""
import pytest

from agents.funnel import Funnel_graph, topo_sort
from core.session import Outcome, Reason
from live.engine import ACTIONS, MAX_KEPT_SESSIONS, LiveSimulation

TARGET_SESSIONS = 300
SEED = 42

KNOWN_REASONS = {
    Reason.PURCHASED,
    Reason.QUIT,
    Reason.SESSION_CAP,
    Reason.INSUFFICIENT_BALANCE,
}


def run(seed=SEED, target=TARGET_SESSIONS, **kwargs):
    kwargs.setdefault("keep_sessions", None)
    sim = LiveSimulation.build(seed=seed, **kwargs)
    events = sim.iter_events()
    while len(sim.sessions) < target:
        next(events)
    return sim


@pytest.fixture(scope="module")
def sim():
    return run()


class TestFunnelIsSingleSourceOfTruth:
    def test_actions_come_from_the_funnel_graph(self):
        """ACTIONS used to be a hand-maintained copy of this ordering."""
        assert ACTIONS == topo_sort(Funnel_graph)

    def test_ordering_is_the_expected_funnel(self):
        assert ACTIONS == ["browse", "product_detail", "cart", "checkout"]


class TestSessionAccounting:
    def test_sessions_are_conserved(self, sim):
        """Every started session is closed or still in flight; none vanish."""
        s = sim.stats
        closed = s["sessions_converted"] + s["sessions_abandoned"]
        assert closed == len(sim.sessions)
        in_flight = s["sessions_started"] - closed
        assert 0 <= in_flight <= sim.n_agents

    def test_every_closed_session_has_an_outcome_and_reason(self, sim):
        for session in sim.sessions:
            assert session.outcome in (Outcome.CONVERTED, Outcome.ABANDONED)
            assert session.reason in KNOWN_REASONS
            assert session.ended_at is not None

    def test_conversion_requires_a_cart(self, sim):
        """The funnel gate should make checkout without a cart impossible."""
        converted = [x for x in sim.sessions if x.outcome == Outcome.CONVERTED]
        assert converted, "no conversions in sample -- widen the run"
        assert all(x.reached_cart() for x in converted)
        assert all(x.revenue > 0 for x in converted)
        assert sim.stats["cart_sessions_converted"] == sim.stats["sessions_converted"]

    def test_abandoned_sessions_earn_nothing(self, sim):
        for session in sim.sessions:
            if session.outcome == Outcome.ABANDONED:
                assert session.revenue == 0.0

    def test_sessions_are_bounded_by_the_cap(self, sim):
        from core.session import MAX_SESSION_ACTIONS

        assert all(x.actions <= MAX_SESSION_ACTIONS for x in sim.sessions)


class TestCartDenominator:
    def test_rate_uses_cart_reaching_sessions_only(self, sim):
        """The old metric divided by every session, which made it 1 - conversion
        by construction and not comparable to the industry figure."""
        s = sim.stats
        expected = 1.0 - s["cart_sessions_converted"] / s["sessions_reached_cart"]
        assert sim.cart_abandonment_rate() == pytest.approx(expected, abs=1e-4)

    def test_denominator_excludes_browse_only_sessions(self, sim):
        counted = sum(1 for x in sim.sessions if x.reached_cart())
        assert sim.stats["sessions_reached_cart"] == counted
        assert counted < len(sim.sessions), "expected some browse-only sessions"

    def test_rate_is_undefined_before_any_cart(self):
        sim = LiveSimulation.build(seed=1)
        assert sim.cart_abandonment_rate() is None


class TestReproducibility:
    def test_same_seed_reproduces_the_run(self):
        a, b = run(seed=7, target=150), run(seed=7, target=150)
        assert [x.summary() for x in a.sessions] == [x.summary() for x in b.sessions]
        assert a.stats["total_events"] == b.stats["total_events"]

    def test_different_seeds_diverge(self):
        a, b = run(seed=7, target=150), run(seed=8, target=150)
        assert [x.summary() for x in a.sessions] != [x.summary() for x in b.sessions]


class TestMemoryIsBounded:
    def test_live_default_caps_retained_sessions(self):
        """The gateway runs this loop for the life of the instance."""
        sim = LiveSimulation.build(seed=3)
        assert sim.keep_sessions == MAX_KEPT_SESSIONS
        events = sim.iter_events()
        while sim.stats["sessions_started"] < MAX_KEPT_SESSIONS + 200:
            next(events)
        assert len(sim.sessions) <= MAX_KEPT_SESSIONS

    def test_batch_mode_keeps_everything(self, sim):
        assert sim.keep_sessions is None
        assert len(sim.sessions) >= TARGET_SESSIONS


class TestEventContract:
    LEGACY = {"time", "agent_id", "persona", "action", "product", "product_id",
              "amount", "balance"}
    ADDED = {"sim_time", "session_id", "stage", "dwell", "stage_revisits",
             "cart_size", "cart_value"}

    def test_dashboard_fields_survive(self, sim):
        assert self.LEGACY <= set(sim.events[-1])

    def test_detector_fields_are_present(self, sim):
        assert self.ADDED <= set(sim.events[-1])

    def test_payload_is_serialisable(self, sim):
        import json

        payload = sim.payload()
        json.dumps(payload)
        assert payload["stats"]["seed"] == SEED
        assert payload["stats"]["cart_abandonment_rate"] is not None

    def test_dwell_and_revisits_are_non_negative(self, sim):
        for event in sim.events:
            assert event["dwell"] >= 0
            assert event["stage_revisits"] >= 0
