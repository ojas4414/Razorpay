"""Session lifecycle and the observables the detector reads.

The detector's two signals are stage dwell and stage revisits. If either is
computed wrongly the detector is predicting from noise, and nothing downstream
would reveal it -- the pipeline would still run and still produce a number. So
they are pinned here.
"""
from core.cart import cart_total
from core.pricing import price_of
from core.session import (
    MAX_SESSION_ACTIONS,
    Outcome,
    Reason,
    Session,
    reset_session_ids,
)


def make(**kwargs):
    kwargs.setdefault("agent_id", 1)
    kwargs.setdefault("persona", "average_buyer")
    kwargs.setdefault("started_at", 0.0)
    return Session(**kwargs)


class TestVisitCounting:
    def test_first_visit_is_not_a_revisit(self):
        s = make()
        assert s.revisits("browse") == 0
        s.enter_stage("product_detail", 1.0)
        assert s.revisits("product_detail") == 0

    def test_returning_to_a_stage_counts_as_a_revisit(self):
        s = make()
        s.enter_stage("product_detail", 1.0)
        s.enter_stage("browse", 2.0)
        s.enter_stage("product_detail", 3.0)
        assert s.revisits("browse") == 1
        assert s.revisits("product_detail") == 1

    def test_revisits_accumulate(self):
        s = make()
        for i, stage in enumerate(["product_detail", "browse", "product_detail", "browse"], 1):
            s.enter_stage(stage, float(i))
        assert s.revisits("browse") == 2
        assert s.revisits("product_detail") == 1

    def test_unvisited_stage_has_no_revisits(self):
        assert make().revisits("checkout") == 0


class TestTiming:
    def test_dwell_measures_time_in_the_current_stage(self):
        s = make()
        s.enter_stage("cart", 5.0)
        assert s.dwell(9.0) == 4.0

    def test_entering_a_stage_resets_dwell(self):
        s = make()
        s.enter_stage("cart", 5.0)
        s.enter_stage("checkout", 8.0)
        assert s.dwell(9.0) == 1.0

    def test_elapsed_measures_the_whole_session(self):
        s = make(started_at=2.0)
        s.enter_stage("cart", 5.0)
        assert s.elapsed(10.0) == 8.0

    def test_action_gaps_are_recorded(self):
        s = make(started_at=0.0)
        s.record_action(1.5)
        s.record_action(4.0)
        assert s.action_gaps == [1.5, 2.5]


class TestCart:
    def test_reached_cart_is_false_for_browse_only(self):
        """Browse-only visits are excluded from the abandonment denominator."""
        s = make()
        s.enter_stage("product_detail", 1.0)
        assert s.reached_cart() is False

    def test_reached_cart_is_true_once_something_is_added(self):
        s = make()
        s.add_to_cart("B000067E30", cart_total(["B000067E30"]))
        assert s.reached_cart() is True

    def test_cart_value_tracks_contents(self):
        ids = ["B000067E30", "B000068PBJ"]
        s = make()
        s.add_to_cart(ids[0], cart_total(ids[:1]))
        assert s.cart_value == price_of(ids[0])
        s.add_to_cart(ids[1], cart_total(ids))
        assert s.cart_value == cart_total(ids)
        assert len(s.cart) == 2


class TestTermination:
    def test_a_new_session_is_open(self):
        assert make().is_open is True

    def test_ending_closes_and_records(self):
        s = make()
        s.end(Outcome.CONVERTED, Reason.PURCHASED, 12.0, revenue=99.5)
        assert s.is_open is False
        assert s.outcome == Outcome.CONVERTED
        assert s.reason == Reason.PURCHASED
        assert s.ended_at == 12.0
        assert s.revenue == 99.5

    def test_cap_bounds_the_session(self):
        """A session that never converts or quits must still terminate."""
        s = make()
        assert s.at_cap() is False
        for i in range(MAX_SESSION_ACTIONS):
            s.record_action(float(i))
        assert s.at_cap() is True

    def test_summary_is_flat_and_complete(self):
        s = make()
        s.add_to_cart("B000067E30", cart_total(["B000067E30"]))
        s.record_action(3.0)
        s.end(Outcome.ABANDONED, Reason.QUIT, 4.0)
        summary = s.summary()
        assert summary["outcome"] == "abandoned"
        assert summary["reason"] == "quit"
        assert summary["reached_cart"] is True
        assert summary["duration"] == 4.0


class TestIdentity:
    def test_ids_are_unique(self):
        assert len({make().session_id for _ in range(50)}) == 50

    def test_reset_makes_ids_reproducible(self):
        """A seeded run must produce the same session IDs, or the audit trail
        cannot be compared across runs."""
        reset_session_ids()
        first = [make().session_id for _ in range(5)]
        reset_session_ids()
        assert [make().session_id for _ in range(5)] == first
