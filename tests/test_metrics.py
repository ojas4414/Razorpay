"""The validation metrics, on hand-built sessions.

These produce the numbers the project publishes, so each one is checked against
a case where the right answer is known by construction rather than by running
the simulation. The denominator tests matter most: using the wrong one is
exactly how the old abandonment figure came to be scored against a benchmark it
did not measure.
"""
import pytest

from core.friction import Driver
from core.session import Outcome, Reason, Session
from validation.metrics import (
    BENCHMARKS,
    abandonment_drivers,
    budget_sensitivity,
    cart_abandonment_rate,
    conversion_rate,
    full_validation_report,
    social_influence_coefficient,
    trending_products,
)


def session(persona="average_buyer", converted=False, carted=False, products=(),
            driver=None, recoverable=False, actions=5):
    s = Session(agent_id=0, persona=persona, started_at=0.0)
    s.products = list(products)
    s.actions = actions
    if carted:
        s.add_to_cart("P1", 100.0)
    if converted:
        s.end(Outcome.CONVERTED, Reason.PURCHASED, 10.0, revenue=100.0)
    else:
        s.quit_driver = driver
        s.recoverable = recoverable
        s.end(Outcome.ABANDONED, Reason.QUIT, 10.0)
    return s


class TestConversion:
    def test_counts_over_all_sessions(self):
        sessions = [session(converted=True)] + [session() for _ in range(9)]
        assert conversion_rate(sessions) == pytest.approx(0.1)

    def test_empty_is_zero_not_an_error(self):
        assert conversion_rate([]) == 0.0


class TestCartAbandonment:
    def test_denominator_is_cart_reaching_sessions_only(self):
        """Two of four carted sessions convert -> 50%, regardless of how many
        browse-only sessions surround them."""
        sessions = (
            [session(carted=True, converted=True) for _ in range(2)]
            + [session(carted=True) for _ in range(2)]
            + [session() for _ in range(96)]      # browse-only, must not count
        )
        assert cart_abandonment_rate(sessions) == pytest.approx(0.5)

    def test_is_not_one_minus_conversion(self):
        """The old metric was 1 - conversion by construction. This one is not."""
        sessions = (
            [session(carted=True, converted=True) for _ in range(2)]
            + [session(carted=True) for _ in range(2)]
            + [session() for _ in range(96)]
        )
        assert cart_abandonment_rate(sessions) != pytest.approx(
            1 - conversion_rate(sessions)
        )

    def test_none_when_nothing_reached_a_cart(self):
        """None, not a misleading zero."""
        assert cart_abandonment_rate([session() for _ in range(5)]) is None


class TestTrendConcentration:
    def test_measures_top_five_share(self):
        sessions = [session(products=["A"] * 10 + ["B", "C", "D", "E", "F", "G"])]
        # top 5 = A(10), B, C, D, E = 14 of 16
        assert social_influence_coefficient(sessions) == pytest.approx(14 / 16, abs=1e-4)

    def test_no_interactions_is_zero(self):
        assert social_influence_coefficient([session()]) == 0.0

    def test_trending_products_ranks_by_count(self):
        sessions = [session(products=["A", "A", "A", "B", "B", "C"])]
        top = trending_products(sessions, top_k=2)
        assert [t["product"] for t in top] == ["A", "B"]
        assert top[0]["count"] == 3


class TestBudgetSensitivity:
    def test_correct_ordering_passes(self):
        sessions = (
            [session(persona="power_buyer", converted=True) for _ in range(5)]
            + [session(persona="average_buyer", converted=True)]
            + [session(persona="average_buyer") for _ in range(9)]
            + [session(persona="browser") for _ in range(10)]
        )
        result = budget_sensitivity(sessions)
        assert result["verdict"] == "PASS"
        assert result["conversion_by_persona"]["power_buyer"] == 1.0
        assert result["conversion_by_persona"]["browser"] == 0.0

    def test_inverted_ordering_fails(self):
        """The old run had power buyers converting at 0.0 and average at 0.14,
        and reported no verdict at all."""
        sessions = (
            [session(persona="power_buyer") for _ in range(10)]
            + [session(persona="average_buyer", converted=True) for _ in range(5)]
            + [session(persona="average_buyer") for _ in range(5)]
        )
        assert budget_sensitivity(sessions)["verdict"] == "FAIL"


class TestAbandonmentDrivers:
    def test_recoverable_share_is_over_cart_abandonment(self):
        sessions = (
            [session(carted=True, driver=Driver.PRICE, recoverable=True) for _ in range(3)]
            + [session(carted=True, driver=Driver.CHURN) for _ in range(7)]
            + [session(carted=True, converted=True) for _ in range(5)]   # not abandoned
            + [session(driver=Driver.INTENT) for _ in range(20)]         # never carted
        )
        result = abandonment_drivers(sessions)
        assert result["cart_abandonment_sessions"] == 10
        assert result["recoverable_share"] == pytest.approx(0.3)
        assert result["by_driver"]["churn"] == pytest.approx(0.7)

    def test_empty_when_nothing_abandoned_a_cart(self):
        assert abandonment_drivers([session(converted=True, carted=True)]) == {}


class TestFullReport:
    @pytest.fixture
    def report(self):
        sim = (
            [session(carted=True, converted=True, products=["A", "B"]) for _ in range(3)]
            + [session(carted=True, driver=Driver.PRICE, recoverable=True,
                       products=["A"]) for _ in range(7)]
            + [session(products=["C"]) for _ in range(90)]
        )
        real = [["x"] * 8 for _ in range(50)]
        return full_validation_report(real, sim)

    def test_has_every_scored_metric(self, report):
        for key in ("session_length", "conversion_rate", "abandonment_rate",
                    "social_influence_coefficient", "budget_sensitivity"):
            assert key in report
            assert "verdict" in report[key]

    def test_abandonment_block_states_its_denominator(self, report):
        """The old report scored this against a benchmark measuring something
        else, without saying so."""
        assert report["abandonment_rate"]["denominator"] == "sessions that created a cart"
        assert report["abandonment_rate"]["simulated"] == pytest.approx(0.7)

    def test_verdicts_derive_from_the_benchmark_table(self, report):
        """Ranges live in one place, so the JSON, the threshold and the printed
        label cannot drift apart the way they had."""
        conv = report["conversion_rate"]
        assert conv["real_benchmark"] == BENCHMARKS["conversion_rate"]["label"]
        low = BENCHMARKS["conversion_rate"]["low"]
        high = BENCHMARKS["conversion_rate"]["high"]
        expected = "PASS" if low <= conv["simulated"] <= high else "FAIL"
        assert conv["verdict"] == expected

    def test_reports_the_recoverable_ceiling(self, report):
        assert report["abandonment_drivers"]["recoverable_share"] == pytest.approx(1.0)
