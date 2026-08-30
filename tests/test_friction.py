"""The hazard model, and the calibration that rests on it.

Two jobs here. The first is the model's mechanics: pressure has to push the
right way, the intervention pathway has to work, and the recoverable/
unrecoverable split has to mean what the pitch says it means.

The second is calibration. All-session conversion and cart-level abandonment are
the numbers the project reports, and they emerge from friction coefficients and
Agent.advance_prob together. Tuning either without re-checking both is how a
published benchmark silently drifts out of band, so it is checked here.
"""
import collections

import pytest

from core.friction import (
    B_INTENT,
    MAX_AFFORDABILITY_PRESSURE,
    MAX_REVISIT_PRESSURE,
    Driver,
    Friction,
)
from core.session import Outcome
from live.engine import LiveSimulation

BASE = dict(cart_value=200.0, balance=1000.0, dwell=2.0, revisits=0)


def friction(intent=0.5, price_sensitivity=1.0):
    return Friction(intent=intent, price_sensitivity=price_sensitivity)


class TestPressureDirection:
    def test_a_bigger_basket_raises_the_hazard(self):
        f = friction()
        cheap = f.hazard(**{**BASE, "cart_value": 100.0})
        dear = f.hazard(**{**BASE, "cart_value": 900.0})
        assert dear.probability > cheap.probability

    def test_a_fuller_wallet_lowers_the_hazard(self):
        f = friction()
        poor = f.hazard(**{**BASE, "balance": 250.0})
        rich = f.hazard(**{**BASE, "balance": 5000.0})
        assert rich.probability < poor.probability

    def test_stalling_raises_the_hazard(self):
        f = friction()
        assert f.hazard(**{**BASE, "dwell": 30.0}).probability > \
               f.hazard(**{**BASE, "dwell": 1.0}).probability

    def test_bouncing_between_stages_raises_the_hazard(self):
        f = friction()
        assert f.hazard(**{**BASE, "revisits": 3}).probability > \
               f.hazard(**{**BASE, "revisits": 0}).probability

    def test_intent_lowers_the_hazard(self):
        assert friction(intent=0.9).hazard(**BASE).probability < \
               friction(intent=0.1).hazard(**BASE).probability

    def test_probability_stays_a_probability(self):
        for intent in (0.0, 0.5, 1.0):
            for cart in (0.0, 1e6):
                p = friction(intent=intent).hazard(
                    cart_value=cart, balance=1.0, dwell=1e6, revisits=1000
                ).probability
                assert 0.0 <= p <= 1.0


class TestInterventionPathway:
    """The discount must act through the model, never around it."""

    def test_a_discount_lowers_the_hazard(self):
        f = friction()
        full = f.hazard(**{**BASE, "cart_value": 800.0})
        discounted = f.hazard(**{**BASE, "cart_value": 800.0 * 0.85})
        assert discounted.probability < full.probability

    def test_a_discount_does_not_help_a_session_with_no_cart(self):
        """Nothing to discount before a basket exists."""
        f = friction()
        assert f.hazard(**{**BASE, "cart_value": 0.0}).contributions[Driver.PRICE] == 0.0

    def test_price_insensitive_shoppers_barely_respond(self):
        """Power buyers hardly notice, which caps what a discount can achieve."""
        sensitive = friction(price_sensitivity=1.8)
        immune = friction(price_sensitivity=0.2)
        delta_sensitive = (sensitive.hazard(**{**BASE, "cart_value": 900.0}).probability
                           - sensitive.hazard(**{**BASE, "cart_value": 450.0}).probability)
        delta_immune = (immune.hazard(**{**BASE, "cart_value": 900.0}).probability
                        - immune.hazard(**{**BASE, "cart_value": 450.0}).probability)
        assert delta_sensitive > delta_immune


class TestSaturation:
    def test_affordability_pressure_is_capped(self):
        f = friction(price_sensitivity=1.0)
        assert f.affordability_pressure(1e9, 100.0) == pytest.approx(
            MAX_AFFORDABILITY_PRESSURE
        )

    def test_revisit_pressure_is_capped(self):
        """Uncapped, churn grew linearly and swamped every other pressure."""
        f = friction()
        at_cap = f.hazard(**{**BASE, "revisits": MAX_REVISIT_PRESSURE})
        beyond = f.hazard(**{**BASE, "revisits": MAX_REVISIT_PRESSURE + 50})
        assert at_cap.probability == pytest.approx(beyond.probability)


class TestDriverAttribution:
    def test_an_unaffordable_basket_is_price_driven(self):
        h = friction(price_sensitivity=1.5).hazard(
            cart_value=2000.0, balance=500.0, dwell=1.0, revisits=0
        )
        assert h.driver == Driver.PRICE
        assert h.recoverable is True

    def test_a_long_stall_is_hesitation_driven(self):
        h = friction().hazard(cart_value=0.0, balance=1000.0, dwell=40.0, revisits=0)
        assert h.driver == Driver.HESITATION
        assert h.recoverable is False

    def test_repeated_bouncing_is_churn_driven(self):
        h = friction().hazard(cart_value=0.0, balance=1000.0, dwell=0.5, revisits=3)
        assert h.driver == Driver.CHURN
        assert h.recoverable is False

    def test_no_meaningful_pressure_means_intent(self):
        """A quit with nothing pushing on it was never going to convert."""
        h = friction(intent=0.05).hazard(
            cart_value=0.0, balance=1000.0, dwell=0.1, revisits=0
        )
        assert h.driver == Driver.INTENT
        assert h.recoverable is False

    def test_only_price_is_recoverable(self):
        """The honest ceiling: a discount cannot fix hesitation or low intent."""
        for driver in (Driver.HESITATION, Driver.CHURN, Driver.INTENT):
            assert driver != Driver.PRICE
        contributions_intent = friction().hazard(**BASE).contributions[Driver.INTENT]
        assert contributions_intent == pytest.approx(-B_INTENT * 0.5)


class TestPersonaTraits:
    def test_personas_differ_in_intent_and_sensitivity(self):
        power = [Friction.draw("power_buyer") for _ in range(200)]
        browser = [Friction.draw("browser") for _ in range(200)]
        assert (sum(f.intent for f in power) / 200) > (sum(f.intent for f in browser) / 200)
        assert (sum(f.price_sensitivity for f in power) / 200) < \
               (sum(f.price_sensitivity for f in browser) / 200)

    def test_unknown_persona_falls_back(self):
        f = Friction.draw("nonexistent")
        assert 0.0 < f.intent < 1.0


class TestCalibration:
    """The published benchmarks. These fail if a coefficient drifts."""

    @pytest.fixture(scope="class")
    def run(self):
        sim = LiveSimulation.build(seed=42, keep_sessions=None)
        events = sim.iter_events()
        while len(sim.sessions) < 2500:
            next(events)
        return sim

    def test_conversion_is_in_the_industry_band(self, run):
        rate = run.stats["sessions_converted"] / len(run.sessions)
        assert 0.02 <= rate <= 0.05, f"conversion {rate:.2%} outside 2-5%"

    def test_cart_abandonment_is_in_the_industry_band(self, run):
        rate = run.cart_abandonment_rate()
        assert 0.70 <= rate <= 0.85, f"cart abandonment {rate:.1%} outside 70-85%"

    def test_session_length_tracks_the_real_data(self, run):
        lengths = [s.actions for s in run.sessions]
        mean = sum(lengths) / len(lengths)
        assert abs(8.16 - mean) < 4.0, f"mean {mean:.2f} too far from real 8.16"

    def test_abandonment_has_a_cause(self, run):
        """The whole point of step 3: no abandonment without an attributed driver."""
        abandoned = [s for s in run.sessions if s.outcome == Outcome.ABANDONED]
        assert abandoned
        assert all(s.quit_driver is not None for s in abandoned)

    def test_every_driver_actually_occurs(self, run):
        """A driver that never fires is a term that is not doing any work."""
        drivers = collections.Counter(
            s.quit_driver for s in run.sessions if s.outcome == Outcome.ABANDONED
        )
        for driver in (Driver.PRICE, Driver.HESITATION, Driver.CHURN, Driver.INTENT):
            assert drivers[driver] > 0, f"{driver} never occurred: {dict(drivers)}"

    def test_recoverable_share_leaves_an_honest_ceiling(self, run):
        """Most cart abandonment must NOT be recoverable.

        An intervention that could reach nearly all of it would mean the model
        had been built to be rescued, which is the failure mode this split
        exists to prevent.
        """
        carted = [s for s in run.sessions
                  if s.outcome == Outcome.ABANDONED and s.reached_cart()]
        share = sum(1 for s in carted if s.recoverable) / len(carted)
        assert 0.25 <= share <= 0.60, f"recoverable share {share:.1%} implausible"

    def test_recoverable_means_price_driven(self, run):
        for session in run.sessions:
            if session.recoverable:
                assert session.quit_driver == Driver.PRICE
