"""The detector: what it flags, what it refuses to flag, and what it cannot see.

The most important test here is that price shock fires without a stall. An
earlier version required a stall before evaluating any second signal, and 99% of
price-driven abandonment -- the only kind a discount can move -- never got past
that gate. The detector looked excellent (97.8% precision) while being pointed
at the sessions it could not help.
"""
import pytest

from core.session import Outcome, Reason, Session
from recovery.detector import (
    FALLBACK_PERSONA,
    Detector,
    Thresholds,
    calibrate,
    gap_zscore,
)

THRESHOLDS = Thresholds(
    dwell_seconds=5.0,
    min_revisits=1,
    gap_sigma=1.5,
    cart_value_shock={"average_buyer": 300.0, FALLBACK_PERSONA: 350.0},
)


def carted_session(persona="average_buyer", cart_value=100.0, started_at=0.0):
    s = Session(agent_id=1, persona=persona, started_at=started_at)
    s.add_to_cart("P1", cart_value)
    return s


def detector():
    return Detector(THRESHOLDS)


class TestGapZScore:
    def test_needs_a_baseline(self):
        assert gap_zscore([1.0, 1.0]) == 0.0

    def test_flat_rhythm_has_no_outlier(self):
        assert gap_zscore([2.0, 2.0, 2.0, 2.0, 2.0]) == 0.0

    def test_a_long_pause_scores_high(self):
        assert gap_zscore([1.0, 1.0, 1.0, 1.0, 9.0]) > 3.0

    def test_measured_against_the_session_itself(self):
        """A uniformly slow shopper is not an outlier against their own rhythm."""
        assert gap_zscore([9.0, 9.0, 9.0, 9.0, 9.0]) == 0.0


class TestWhatIsIgnored:
    def test_browse_only_sessions_are_never_flagged(self):
        """Not recoverable revenue, and counting them repeats the denominator
        mistake the old abandonment metric made."""
        s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
        s.enter_stage("product_detail", 1.0)
        assert detector().evaluate(s, 100.0) is None

    def test_closed_sessions_are_never_flagged(self):
        s = carted_session()
        s.end(Outcome.ABANDONED, Reason.QUIT, 10.0)
        assert detector().evaluate(s, 100.0) is None

    def test_a_calm_cheap_session_is_not_flagged(self):
        s = carted_session(cart_value=50.0)
        assert detector().evaluate(s, 1.0) is None


class TestStallRoute:
    def test_stall_alone_is_not_enough(self):
        """A slow shopper is not a leaving shopper."""
        s = carted_session(cart_value=50.0)
        assert detector().evaluate(s, 60.0) is None

    def test_hesitation_without_a_stall_is_not_enough(self):
        s = carted_session(cart_value=50.0)
        s.enter_stage("product_detail", 1.0)
        s.enter_stage("cart", 2.0)
        s.enter_stage("product_detail", 3.0)   # revisit
        flag = detector().evaluate(s, 4.0)     # dwell 1.0, under threshold
        assert flag is None

    def test_stall_plus_revisit_flags(self):
        s = carted_session(cart_value=50.0)
        s.enter_stage("product_detail", 1.0)
        s.enter_stage("cart", 2.0)
        s.enter_stage("product_detail", 3.0)   # revisit of product_detail
        flag = detector().evaluate(s, 30.0)    # dwell 27s
        assert flag is not None
        assert flag.trigger == "revisit"

    def test_stall_plus_pause_flags(self):
        s = carted_session(cart_value=50.0)
        s.action_gaps = [1.0, 1.0, 1.0, 1.0, 9.0]
        flag = detector().evaluate(s, 30.0)
        assert flag is not None
        assert flag.trigger == "pause"


class TestPriceShockRoute:
    def test_price_shock_fires_without_any_stall(self):
        """The property the whole detector was rebuilt for.

        Shoppers who leave over price do not dither first, so gating price shock
        behind a stall excludes exactly the population the intervention serves.
        """
        s = carted_session(cart_value=500.0)
        flag = detector().evaluate(s, 0.5)      # dwell 0.5s, far under threshold
        assert flag is not None
        assert flag.trigger == "price_shock"

    def test_threshold_is_per_persona(self):
        """The same basket is trivial to one segment and prohibitive to another."""
        thresholds = Thresholds(
            dwell_seconds=5.0,
            cart_value_shock={"browser": 100.0, "power_buyer": 900.0},
        )
        cheap_for_power = carted_session(persona="power_buyer", cart_value=400.0)
        dear_for_browser = carted_session(persona="browser", cart_value=400.0)
        assert Detector(thresholds).evaluate(cheap_for_power, 0.5) is None
        assert Detector(thresholds).evaluate(dear_for_browser, 0.5) is not None

    def test_unknown_persona_falls_back(self):
        assert THRESHOLDS.shock_for("someone_else") == 350.0

    def test_known_persona_uses_its_own(self):
        assert THRESHOLDS.shock_for("average_buyer") == 300.0


class TestFlagContents:
    def test_records_the_evidence_that_raised_it(self):
        """Every decision has to be replayable and explainable afterwards."""
        s = carted_session(cart_value=500.0)
        flag = detector().evaluate(s, 2.0)
        assert flag.session_id == s.session_id
        assert flag.sim_time == 2.0
        for key in ("dwell", "revisits", "gap_zscore", "cart_value",
                    "cart_size", "price_shock", "actions", "elapsed", "persona"):
            assert key in flag.features

    def test_cannot_see_latent_state(self):
        """intent and price_sensitivity decide the outcome. A detector that
        could read them would be an oracle, and its precision would mean
        nothing."""
        from core.friction import Friction

        s = carted_session(cart_value=500.0)
        s.friction = Friction(intent=0.9, price_sensitivity=1.4)
        flag = detector().evaluate(s, 2.0)
        assert "intent" not in flag.features
        assert "price_sensitivity" not in flag.features
        assert 0.9 not in flag.features.values()


class TestCalibration:
    def _events(self, n=40):
        return [
            {
                "session_id": "s1",
                "persona": "average_buyer",
                "dwell": float(i % 10),
                "stage_revisits": i % 3,
                "cart_value": 100.0 + i * 10,
                "cart_size": 1,
            }
            for i in range(n)
        ]

    def _sessions(self):
        s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
        s.session_id = "s1"
        s.add_to_cart("P1", 100.0)
        s.end(Outcome.CONVERTED, Reason.PURCHASED, 5.0, revenue=100.0)
        return [s]

    def test_derives_thresholds_from_converting_sessions(self):
        thresholds = calibrate(self._events(), self._sessions(), percentile=75)
        assert thresholds.dwell_seconds > 0
        assert thresholds.min_revisits >= 1
        assert "converting-session actions" in thresholds.calibrated_on

    def test_produces_a_per_persona_shock_table(self):
        thresholds = calibrate(self._events(), self._sessions(), percentile=75)
        assert "average_buyer" in thresholds.cart_value_shock
        assert FALLBACK_PERSONA in thresholds.cart_value_shock

    def test_a_higher_percentile_is_a_stricter_cut(self):
        loose = calibrate(self._events(), self._sessions(), percentile=50)
        strict = calibrate(self._events(), self._sessions(), percentile=90)
        assert strict.shock_for("average_buyer") >= loose.shock_for("average_buyer")

    def test_no_conversions_falls_back_to_defaults(self):
        s = Session(agent_id=1, persona="average_buyer", started_at=0.0)
        s.end(Outcome.ABANDONED, Reason.QUIT, 5.0)
        assert calibrate(self._events(), [s]).calibrated_on == "defaults"
