"""Arm assignment and the intervention: bounded, late, and causal only.

The load-bearing tests here are the ones that would let a fake result through if
they failed: that assignment consumes no randomness, and that an intervention
can reach an outcome only by changing what the shopper is asked to pay.
"""
import random

import pytest

from core.friction import Friction
from core.session import Outcome, Reason, Session
from recovery.assign import CONTROL, TREATMENT, arm_for
from recovery.intervention import (
    DEFAULT_DISCOUNT_RATE,
    InterventionPolicy,
)


def flagged_session(session_id="s0000001", cart_value=400.0, trigger="price_shock"):
    s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
    s.session_id = session_id
    s.add_to_cart("P1", cart_value)

    class _Flag:
        pass

    flag = _Flag()
    flag.trigger = trigger
    s.flag = flag
    return s


class TestArmAssignment:
    def test_consumes_no_randomness(self):
        """Assignment must not touch the global RNG.

        The whole simulation runs off one seeded stream, so drawing here would
        shift every later draw and the control arm would stop being the run you
        would have got with no experiment at all.
        """
        random.seed(1234)
        expected = [random.random() for _ in range(5)]

        random.seed(1234)
        for i in range(100):
            arm_for(f"s{i:07d}")
        assert [random.random() for _ in range(5)] == expected

    def test_is_deterministic(self):
        assert arm_for("s0000001") == arm_for("s0000001")

    def test_salt_reshuffles_the_split(self):
        arms_a = [arm_for(f"s{i:07d}", salt="a") for i in range(200)]
        arms_b = [arm_for(f"s{i:07d}", salt="b") for i in range(200)]
        assert arms_a != arms_b

    def test_splits_roughly_evenly(self):
        arms = [arm_for(f"s{i:07d}") for i in range(2000)]
        share = arms.count(TREATMENT) / len(arms)
        assert 0.45 < share < 0.55

    def test_zero_share_treats_nobody(self):
        """The A/A configuration: assigned as usual, treated never."""
        arms = [arm_for(f"s{i:07d}", treatment_share=0.0) for i in range(200)]
        assert set(arms) == {CONTROL}


class TestIdempotency:
    def test_one_intervention_per_session(self):
        policy = InterventionPolicy()
        session = flagged_session()
        assert policy.dispatch(session, 1.0) is not None
        assert policy.dispatch(session, 2.0) is None
        assert policy.issued == 1
        assert policy.duplicates_refused == 1

    def test_a_duplicate_is_counted_not_raised(self):
        """Retries are expected in a real system, not an error condition."""
        policy = InterventionPolicy()
        session = flagged_session()
        policy.dispatch(session, 1.0)
        for _ in range(5):
            policy.dispatch(session, 2.0)
        assert policy.duplicates_refused == 5

    def test_key_is_stable_and_per_session(self):
        assert (InterventionPolicy.idempotency_key("s1")
                != InterventionPolicy.idempotency_key("s2"))
        assert (InterventionPolicy.idempotency_key("s1")
                == InterventionPolicy.idempotency_key("s1"))


class TestLatency:
    def test_dispatch_is_scheduled_into_the_future(self):
        policy = InterventionPolicy(dispatch_latency=3.0)
        issued = policy.dispatch(flagged_session(), 10.0)
        assert issued.dispatched_at == 13.0
        assert issued.landed is False

    def test_landing_on_an_open_session_applies_the_discount(self):
        policy = InterventionPolicy(discount_rate=0.2)
        session = flagged_session()
        policy.dispatch(session, 1.0)
        assert policy.apply(session, 4.0) is True
        assert session.discount == 0.2
        assert session.intervention.landed is True

    def test_arriving_after_the_session_ended_is_a_miss(self):
        """A discount that lands too late must change nothing."""
        policy = InterventionPolicy()
        session = flagged_session()
        policy.dispatch(session, 1.0)
        session.end(Outcome.ABANDONED, Reason.QUIT, 2.0)

        assert policy.apply(session, 4.0) is False
        assert session.discount == 0.0
        assert session.intervention.missed is True
        assert policy.missed == 1

    def test_applying_twice_does_nothing(self):
        policy = InterventionPolicy()
        session = flagged_session()
        policy.dispatch(session, 1.0)
        policy.apply(session, 2.0)
        assert policy.apply(session, 3.0) is False


class TestTargeting:
    def test_ineligible_triggers_are_not_treated(self):
        policy = InterventionPolicy(eligible_triggers=frozenset({"price_shock"}))
        assert policy.dispatch(flagged_session(trigger="revisit"), 1.0) is None
        assert policy.ineligible == 1

    def test_eligible_triggers_are_treated(self):
        policy = InterventionPolicy(eligible_triggers=frozenset({"price_shock"}))
        assert policy.dispatch(flagged_session(trigger="price_shock"), 1.0) is not None

    def test_no_restriction_treats_everything(self):
        policy = InterventionPolicy()
        assert policy.dispatch(flagged_session(trigger="pause"), 1.0) is not None


class TestCausalPathway:
    """An intervention may reach an outcome only through what is charged."""

    def test_discount_lowers_effective_cart_value(self):
        session = flagged_session(cart_value=400.0)
        assert session.effective_cart_value() == 400.0
        session.discount = 0.25
        assert session.effective_cart_value() == 300.0

    def test_applying_does_not_touch_the_outcome(self):
        """No branch may mark a session recovered. If a discount saves a
        session it must do so through the hazard, like everything else."""
        policy = InterventionPolicy()
        session = flagged_session()
        policy.dispatch(session, 1.0)
        policy.apply(session, 2.0)

        assert session.outcome is None
        assert session.is_open is True
        assert session.revenue == 0.0
        assert session.quit_driver is None

    def test_discount_lowers_the_quit_hazard_through_price(self):
        """The full chain: discount -> cart value -> affordability -> hazard."""
        friction = Friction(intent=0.4, price_sensitivity=1.0)
        session = flagged_session(cart_value=800.0)

        full = friction.hazard(cart_value=session.effective_cart_value(),
                               balance=1000.0, dwell=2.0, revisits=0)
        session.discount = DEFAULT_DISCOUNT_RATE
        discounted = friction.hazard(cart_value=session.effective_cart_value(),
                                     balance=1000.0, dwell=2.0, revisits=0)
        assert discounted.probability < full.probability

    def test_no_cart_means_no_leverage(self):
        session = Session(agent_id=1, persona="average_buyer", started_at=0.0)
        session.discount = 0.5
        assert session.effective_cart_value() == 0.0


class TestPolicyStats:
    def test_reports_what_happened(self):
        policy = InterventionPolicy(discount_rate=0.2, dispatch_latency=1.0)
        landed = flagged_session("s1")
        policy.dispatch(landed, 1.0)
        policy.apply(landed, 2.0)

        late = flagged_session("s2")
        policy.dispatch(late, 1.0)
        late.end(Outcome.ABANDONED, Reason.QUIT, 1.5)
        policy.apply(late, 2.0)

        stats = policy.stats()
        assert stats["issued"] == 2
        assert stats["landed"] == 1
        assert stats["missed"] == 1
        assert stats["discount_rate"] == 0.2
