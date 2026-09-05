"""Statistics for the recovery experiment.

A bare percentage difference between two arms is not a result. Three things have
to accompany it or it means nothing:

**A confidence interval.** "Recovered 22% more carts" with n=90 per arm is a
sentence about noise. The interval says how much of the estimate is real.

**A minimum detectable effect.** Reporting "no significant difference" is only
informative alongside the effect the experiment could have detected. A null at
MDE 8pp says almost nothing; a null at MDE 1pp says the effect is small.

**Net margin, not recovered revenue.** Every treated session that converts is
billed at a discount, including the ones that would have converted anyway. Gross
recovered revenue counts the upside and ignores the bill, and there is a
discount rate at which a programme "recovers" carts while losing money on every
one. That crossover is the number a payments team actually needs.

All of this rests on the arms being comparable, which is what the A/A check in
sim/experiment.py is for: run the same machinery with nothing applied and it
must report no effect.
"""
import math
from dataclasses import dataclass

# Two-sided 95% and the 80%-power constant, kept as names so the formulas below
# read as statistics rather than as magic numbers.
Z_95 = 1.959963985
Z_POWER_80 = 0.8416212336


@dataclass(frozen=True)
class ProportionTest:
    """A two-proportion comparison with everything needed to read it."""

    treat_hits: int
    treat_n: int
    control_hits: int
    control_n: int

    @property
    def treat_rate(self) -> float:
        return self.treat_hits / self.treat_n if self.treat_n else 0.0

    @property
    def control_rate(self) -> float:
        return self.control_hits / self.control_n if self.control_n else 0.0

    @property
    def delta(self) -> float:
        return self.treat_rate - self.control_rate

    @property
    def standard_error(self) -> float:
        """Unpooled SE, which is what the confidence interval needs."""
        if not (self.treat_n and self.control_n):
            return 0.0
        a = self.treat_rate * (1 - self.treat_rate) / self.treat_n
        b = self.control_rate * (1 - self.control_rate) / self.control_n
        return math.sqrt(a + b)

    @property
    def z(self) -> float:
        """Pooled-variance z statistic, the standard test for two proportions."""
        if not (self.treat_n and self.control_n):
            return 0.0
        pooled = ((self.treat_hits + self.control_hits)
                  / (self.treat_n + self.control_n))
        se = math.sqrt(pooled * (1 - pooled) * (1 / self.treat_n + 1 / self.control_n))
        return self.delta / se if se else 0.0

    @property
    def confidence_interval(self):
        margin = Z_95 * self.standard_error
        return (self.delta - margin, self.delta + margin)

    @property
    def significant(self) -> bool:
        return abs(self.z) > Z_95

    def minimum_detectable_effect(self, power_z=Z_POWER_80) -> float:
        """The smallest true effect this sample size could have found.

        Uses the control rate as the baseline. A null result is only meaningful
        when read against this number.
        """
        if not (self.treat_n and self.control_n):
            return 0.0
        p = self.control_rate
        harmonic = (1 / self.treat_n) + (1 / self.control_n)
        return (Z_95 + power_z) * math.sqrt(p * (1 - p) * harmonic)

    def summary(self) -> dict:
        low, high = self.confidence_interval
        return {
            "treat_rate": round(self.treat_rate, 4),
            "control_rate": round(self.control_rate, 4),
            "treat_n": self.treat_n,
            "control_n": self.control_n,
            "delta": round(self.delta, 4),
            "ci_low": round(low, 4),
            "ci_high": round(high, 4),
            "z": round(self.z, 3),
            "significant": self.significant,
            "mde": round(self.minimum_detectable_effect(), 4),
        }

    def describe(self) -> str:
        low, high = self.confidence_interval
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (f"{self.delta:+.2%} "
                f"(95% CI {low:+.2%} to {high:+.2%}, z={self.z:+.2f}, "
                f"n={self.treat_n}/{self.control_n}) -- {verdict}; "
                f"MDE {self.minimum_detectable_effect():.2%}")


def required_sample_size(baseline_rate: float, effect: float,
                         power_z=Z_POWER_80) -> int:
    """Sessions per arm needed to detect ``effect`` at 80% power.

    Declared before a run rather than after, so the experiment is sized to the
    effect that would matter instead of the effect that happened to appear.
    """
    if effect <= 0:
        return 0
    p = baseline_rate
    return math.ceil(2 * ((Z_95 + power_z) ** 2) * p * (1 - p) / (effect ** 2))


@dataclass(frozen=True)
class MarginResult:
    """What the programme earned, after paying for itself."""

    incremental_conversions: float
    gross_recovered: float
    discount_cost: float

    @property
    def net(self) -> float:
        return self.gross_recovered - self.discount_cost

    @property
    def positive(self) -> bool:
        return self.net > 0

    def summary(self) -> dict:
        return {
            "incremental_conversions": round(self.incremental_conversions, 2),
            "gross_recovered": round(self.gross_recovered, 2),
            "discount_cost": round(self.discount_cost, 2),
            "net": round(self.net, 2),
            "margin_positive": self.positive,
        }


def bootstrap_margin(treated, control, discount_rate: float, rounds: int = 1500,
                     rng=None):
    """Percentile interval on net margin, by resampling sessions.

    Margin is noisier than the conversion rate and it was being reported without
    any interval, which made it look more precise than it is. Two policies with
    effectively identical deltas (+6.70% against +6.69%) and near-identical
    offer counts produced nets 1,961 apart -- because margin depends not only on
    how many extra sessions converted but on what their baskets happened to be
    worth.

    Bootstrapped rather than derived: net margin is a function of a rate
    difference and a mean cart value together, and resampling captures their
    joint variability without an analytic approximation that would need
    defending.
    """
    import random as _random

    rng = rng or _random.Random(20260905)
    if not treated or not control:
        return (0.0, 0.0)

    nets = []
    for _ in range(rounds):
        t_sample = [rng.choice(treated) for _ in range(len(treated))]
        c_sample = [rng.choice(control) for _ in range(len(control))]
        nets.append(net_margin(t_sample, c_sample, discount_rate).net)
    nets.sort()
    return (nets[int(0.025 * rounds)], nets[int(0.975 * rounds)])


def net_margin(treated, control, discount_rate: float) -> MarginResult:
    """Net value of treating this arm, against what control did without it.

    Incremental conversions are estimated from the rate difference rather than
    counted, because which specific sessions were recovered is not observable --
    only the difference between the two arms is.

    The discount is billed on *every* treated conversion, not just the
    incremental ones. That is the asymmetry that sinks naive discounting: the
    offer goes to shoppers who were going to buy anyway, and they take it.
    """
    if not treated or not control:
        return MarginResult(0.0, 0.0, 0.0)

    converted_t = [s for s in treated if s.outcome == "converted"]
    converted_c = [s for s in control if s.outcome == "converted"]

    treat_rate = len(converted_t) / len(treated)
    control_rate = len(converted_c) / len(control)
    incremental = (treat_rate - control_rate) * len(treated)

    # Value an incremental conversion at the average full cart of a converting
    # control session: what the shopper would have paid without an offer.
    average_cart = (sum(s.cart_value for s in converted_c) / len(converted_c)
                    if converted_c else 0.0)
    gross = incremental * average_cart

    # Billed on every treated conversion that actually received the discount.
    cost = sum(s.cart_value * s.discount for s in converted_t)
    return MarginResult(incremental, gross, cost)
