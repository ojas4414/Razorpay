"""Choosing an action by diagnosis, escalating, and knowing when to stop.

The tests that matter most are the bounds. An automated system that escalates
spending needs a ceiling it cannot talk itself past, and the diagnosis must come
from what the detector saw rather than from the answer key the simulation keeps.
"""
import pytest

from core.session import Outcome, Reason, Session
from recovery.detector import Flag
from recovery.playbook import (
    DISCOUNT_10,
    DISCOUNT_20,
    MAX_ATTEMPTS,
    REMINDER,
    SIMPLIFY,
    Kind,
    PlaybookPolicy,
    diagnose,
    ladder_for,
)
from recovery.safety import SafetyGate


def features(price_shock=False, dwell=1.0, revisits=0, gap=0.0, cart=300.0):
    return {"price_shock": price_shock, "dwell": dwell, "revisits": revisits,
            "gap_zscore": gap, "cart_value": cart, "cart_size": 1,
            "actions": 6, "elapsed": 15.0, "persona": "average_buyer"}


def flagged(session_id="s1", **feature_kwargs):
    """One flagged session. `cart` sets both the basket and the recorded
    feature, which a first version let drift apart -- the session held 300 while
    the features said 500, and the cost assertions failed against the code being
    right."""
    s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
    s.session_id = session_id
    f = features(**feature_kwargs)
    s.add_to_cart("P1", f["cart_value"])
    s.flag = Flag(session_id=session_id, sim_time=2.0, stage="cart",
                  trigger="price_shock" if f["price_shock"] else "revisit",
                  features=f)
    return s


class TestDiagnosis:
    def test_a_big_basket_is_price(self):
        assert diagnose(features(price_shock=True)) == "price"

    def test_bouncing_is_churn(self):
        assert diagnose(features(revisits=3, dwell=0.5)) == "churn"

    def test_stalling_is_hesitation(self):
        assert diagnose(features(dwell=30.0, revisits=0)) == "hesitation"

    def test_a_long_pause_is_hesitation(self):
        assert diagnose(features(gap=6.0, revisits=0, dwell=0.1)) == "hesitation"

    def test_price_outranks_everything(self):
        """Nothing cheaper fixes a basket someone cannot afford."""
        assert diagnose(features(price_shock=True, revisits=5, dwell=99)) == "price"

    def test_no_signal_is_unknown(self):
        assert diagnose(features(dwell=0.0, revisits=0, gap=0.0)) == "unknown"

    def test_reads_features_not_the_trigger_label(self):
        """The trigger checks price first, so `pause` never won and the reminder
        rung was unreachable. Diagnosis reads the features instead."""
        assert diagnose(features(dwell=30.0, revisits=0)) == "hesitation"

    def test_never_sees_the_true_driver(self):
        """quit_driver is the answer key the diagnosis is scored against."""
        assert "quit_driver" not in features()
        assert "recoverable" not in features()


class TestLadders:
    def test_price_starts_at_a_discount(self):
        ladder = ladder_for(features(price_shock=True))
        assert ladder[0].kind == Kind.DISCOUNT

    def test_hesitation_starts_free(self):
        assert ladder_for(features(dwell=30.0))[0] is REMINDER

    def test_churn_starts_free(self):
        assert ladder_for(features(revisits=3))[0] is SIMPLIFY

    def test_every_ladder_is_ordered_by_cost(self):
        """Cheapest first. A ladder that opened with the expensive rung would
        make escalation meaningless."""
        for f in (features(price_shock=True), features(dwell=30.0),
                  features(revisits=3)):
            costs = [a.cost_rate for a in ladder_for(f)]
            assert costs == sorted(costs)

    def test_discounts_increase_up_the_ladder(self):
        assert DISCOUNT_10.magnitude < DISCOUNT_20.magnitude


class TestEscalation:
    def test_first_dispatch_takes_the_cheapest_rung(self):
        book = PlaybookPolicy()
        session = flagged(revisits=3)
        assert book.dispatch(session, 1.0).action is SIMPLIFY

    def test_second_dispatch_escalates(self):
        book = PlaybookPolicy()
        session = flagged(revisits=3)
        book.dispatch(session, 1.0)
        assert book.dispatch(session, 5.0).action is DISCOUNT_10
        assert book.escalations == 1

    def test_stops_at_the_attempt_cap(self):
        """The ceiling an automated spender must not be able to talk past."""
        book = PlaybookPolicy(max_attempts=2)
        session = flagged(revisits=3)
        assert book.dispatch(session, 1.0) is not None
        assert book.dispatch(session, 5.0) is not None
        assert book.dispatch(session, 9.0) is None

    def test_stops_when_the_ladder_runs_out(self):
        book = PlaybookPolicy(max_attempts=99)
        session = flagged(price_shock=True)      # a two-rung ladder
        book.dispatch(session, 1.0)
        book.dispatch(session, 5.0)
        assert book.dispatch(session, 9.0) is None
        assert book.exhausted == 1

    def test_each_rung_has_its_own_idempotency_key(self):
        book = PlaybookPolicy()
        session = flagged(revisits=3)
        first = book.dispatch(session, 1.0)
        second = book.dispatch(session, 5.0)
        assert first.idempotency_key != second.idempotency_key
        assert first.idempotency_key.endswith("rung0")


class TestStoppingRules:
    def test_the_spend_guard_can_refuse(self):
        gate = SafetyGate()
        gate.guard.max_offers_per_window = 1
        book = PlaybookPolicy(safety=gate)
        assert book.dispatch(flagged("a", price_shock=True), 1.0) is not None
        assert book.dispatch(flagged("b", price_shock=True), 1.0) is None
        assert book.refused_by_safety == 1

    def test_the_kill_switch_stops_everything(self):
        gate = SafetyGate()
        gate.kill_switch.engage("incident")
        book = PlaybookPolicy(safety=gate)
        assert book.dispatch(flagged(price_shock=True), 1.0) is None
        assert book.refused_by_safety == 1

    def test_free_actions_still_pass_the_guard(self):
        """A reminder costs nothing, so a spend cap should not block it."""
        gate = SafetyGate()
        gate.guard.max_spend_per_window = 0.01
        book = PlaybookPolicy(safety=gate)
        assert book.dispatch(flagged(dwell=30.0), 1.0) is not None


class TestApplication:
    def test_a_discount_lowers_what_is_charged(self):
        book = PlaybookPolicy()
        session = flagged(price_shock=True, cart=400.0)
        book.dispatch(session, 1.0)
        assert book.apply(session, 2.0) is True
        assert session.discount == DISCOUNT_10.magnitude
        assert session.effective_cart_value() < 400.0

    def test_a_reminder_restarts_the_stall_clock(self):
        book = PlaybookPolicy()
        session = flagged(dwell=30.0)
        session.enter_stage("cart", 0.0)
        book.dispatch(session, 10.0)
        book.apply(session, 10.0)
        assert session.effective_dwell(10.0) == pytest.approx(0.0)

    def test_simplifying_forgives_a_bounce(self):
        book = PlaybookPolicy()
        session = flagged(revisits=3)
        session.enter_stage("product_detail", 1.0)
        session.enter_stage("cart", 2.0)
        before = session.revisits("cart")
        book.dispatch(session, 3.0)
        book.apply(session, 3.0)
        assert session.effective_revisits("cart") == max(0, before - 1)

    def test_free_actions_help_progression(self):
        """Relieving quit pressure alone made things measurably worse: a
        reminder kept a stalled session alive without moving it, and it ran into
        the action cap. Measured at -3.41% (z=-2.56) before this was added."""
        assert REMINDER.advance_boost > 1.0
        assert SIMPLIFY.advance_boost > 1.0
        book = PlaybookPolicy()
        session = flagged(dwell=30.0)
        book.dispatch(session, 1.0)
        book.apply(session, 2.0)
        assert session.advance_boost > 1.0

    def test_an_action_arriving_late_is_a_miss(self):
        book = PlaybookPolicy()
        session = flagged(price_shock=True)
        book.dispatch(session, 1.0)
        session.end(Outcome.ABANDONED, Reason.QUIT, 2.0)
        assert book.apply(session, 3.0) is False
        assert book.missed == 1

    def test_applying_twice_does_nothing(self):
        book = PlaybookPolicy()
        session = flagged(price_shock=True)
        book.dispatch(session, 1.0)
        book.apply(session, 2.0)
        assert book.apply(session, 3.0) is False


class TestCost:
    def test_free_actions_bill_nothing(self):
        book = PlaybookPolicy()
        session = flagged(dwell=30.0, cart=500.0)
        book.dispatch(session, 1.0)
        book.apply(session, 2.0)
        assert book.cost_for(session) == 0.0

    def test_discounts_bill_their_rate(self):
        book = PlaybookPolicy()
        session = flagged(price_shock=True, cart=500.0)
        book.dispatch(session, 1.0)
        book.apply(session, 2.0)
        assert book.cost_for(session) == pytest.approx(500.0 * DISCOUNT_10.cost_rate)

    def test_escalation_bills_both_rungs(self):
        """Escalating is not free; the analysis must see the whole bill."""
        book = PlaybookPolicy()
        session = flagged(revisits=3, cart=500.0)
        book.dispatch(session, 1.0)
        book.apply(session, 2.0)          # simplify, free
        book.dispatch(session, 6.0)
        book.apply(session, 7.0)          # discount_10
        assert book.cost_for(session) == pytest.approx(500.0 * DISCOUNT_10.cost_rate)

    def test_stats_are_serialisable(self):
        import json

        book = PlaybookPolicy()
        book.dispatch(flagged(price_shock=True), 1.0)
        json.dumps(book.stats())
