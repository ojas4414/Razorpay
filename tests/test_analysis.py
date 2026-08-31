"""The statistics that decide whether a result is a result.

The net-margin tests matter most. Gross recovered revenue counts the upside and
ignores the bill, and there is a discount rate at which a programme recovers
carts while losing money on every one -- which is what the measured run actually
does. If net_margin quietly billed only the incremental conversions it would
report a profit that does not exist.
"""
import pytest

from recovery.analysis import (
    MarginResult,
    ProportionTest,
    net_margin,
    required_sample_size,
)


class _Session:
    """Minimal stand-in: net_margin only needs these three fields."""

    def __init__(self, outcome, cart_value=100.0, discount=0.0):
        self.outcome = outcome
        self.cart_value = cart_value
        self.discount = discount


class TestProportionTest:
    def test_delta_is_the_rate_difference(self):
        test = ProportionTest(treat_hits=60, treat_n=600,
                              control_hits=50, control_n=600)
        assert test.treat_rate == pytest.approx(0.10)
        assert test.control_rate == pytest.approx(0.0833, abs=1e-3)
        assert test.delta == pytest.approx(0.0167, abs=1e-3)

    def test_a_large_clear_effect_is_significant(self):
        test = ProportionTest(treat_hits=300, treat_n=1000,
                              control_hits=150, control_n=1000)
        assert test.significant
        assert test.z > 1.96

    def test_a_tiny_effect_on_small_n_is_not(self):
        test = ProportionTest(treat_hits=34, treat_n=623,
                              control_hits=32, control_n=623)
        assert not test.significant

    def test_confidence_interval_straddles_zero_when_null(self):
        test = ProportionTest(treat_hits=34, treat_n=623,
                              control_hits=32, control_n=623)
        low, high = test.confidence_interval
        assert low < 0 < high

    def test_confidence_interval_excludes_zero_when_significant(self):
        test = ProportionTest(treat_hits=300, treat_n=1000,
                              control_hits=150, control_n=1000)
        low, high = test.confidence_interval
        assert low > 0

    def test_identical_arms_give_no_effect(self):
        test = ProportionTest(treat_hits=50, treat_n=500,
                              control_hits=50, control_n=500)
        assert test.delta == 0.0
        assert test.z == 0.0
        assert not test.significant

    def test_empty_arms_do_not_explode(self):
        test = ProportionTest(0, 0, 0, 0)
        assert test.z == 0.0
        assert test.delta == 0.0
        assert test.minimum_detectable_effect() == 0.0

    def test_mde_shrinks_as_the_sample_grows(self):
        """A null result is only meaningful against the effect it could find."""
        small = ProportionTest(30, 500, 30, 500)
        large = ProportionTest(600, 10000, 600, 10000)
        assert large.minimum_detectable_effect() < small.minimum_detectable_effect()

    def test_describe_names_the_verdict(self):
        assert "not significant" in ProportionTest(34, 623, 32, 623).describe()
        assert "SIGNIFICANT" in ProportionTest(300, 1000, 150, 1000).describe()


class TestSampleSize:
    def test_smaller_effects_need_more_data(self):
        assert required_sample_size(0.08, 0.01) > required_sample_size(0.08, 0.05)

    def test_a_realistic_ask_is_large(self):
        """Sanity: detecting 1pp on an 8% base takes thousands per arm, which is
        why the 12,000-session run could not resolve it."""
        assert required_sample_size(0.08, 0.01) > 5000

    def test_zero_effect_is_undefined(self):
        assert required_sample_size(0.08, 0.0) == 0


class TestNetMargin:
    def test_discount_is_billed_on_every_treated_conversion(self):
        """Not just the incremental ones. This is the asymmetry that sinks
        naive discounting: the offer reaches shoppers who would have bought
        anyway, and they take it."""
        treated = [_Session("converted", 100.0, 0.15) for _ in range(10)]
        treated += [_Session("abandoned", 100.0, 0.15) for _ in range(90)]
        control = [_Session("converted", 100.0) for _ in range(10)]
        control += [_Session("abandoned", 100.0) for _ in range(90)]

        result = net_margin(treated, control, 0.15)
        # Same conversion rate in both arms -> nothing incremental, all cost.
        assert result.incremental_conversions == pytest.approx(0.0)
        assert result.gross_recovered == pytest.approx(0.0)
        assert result.discount_cost == pytest.approx(10 * 100.0 * 0.15)
        assert not result.positive

    def test_a_real_lift_can_still_lose_money(self):
        """The crossover a payments team needs: carts recovered, money lost."""
        treated = [_Session("converted", 100.0, 0.5) for _ in range(12)]
        treated += [_Session("abandoned", 100.0, 0.5) for _ in range(88)]
        control = [_Session("converted", 100.0) for _ in range(10)]
        control += [_Session("abandoned", 100.0) for _ in range(90)]

        result = net_margin(treated, control, 0.5)
        assert result.incremental_conversions > 0      # it did recover carts
        assert result.gross_recovered == pytest.approx(200.0)
        assert result.discount_cost == pytest.approx(600.0)
        assert not result.positive                     # and still lost money

    def test_a_big_lift_on_a_small_discount_is_positive(self):
        treated = [_Session("converted", 100.0, 0.05) for _ in range(30)]
        treated += [_Session("abandoned", 100.0, 0.05) for _ in range(70)]
        control = [_Session("converted", 100.0) for _ in range(10)]
        control += [_Session("abandoned", 100.0) for _ in range(90)]

        result = net_margin(treated, control, 0.05)
        assert result.positive
        assert result.net == pytest.approx(2000.0 - 150.0)

    def test_empty_arms_are_zero(self):
        assert net_margin([], [], 0.15) == MarginResult(0.0, 0.0, 0.0)

    def test_summary_is_serialisable(self):
        import json

        treated = [_Session("converted", 100.0, 0.15)]
        control = [_Session("converted", 100.0)]
        json.dumps(net_margin(treated, control, 0.15).summary())
