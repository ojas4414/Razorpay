"""Why a session ends.

Until now the quit decision was ``random.random() < quit_prob`` -- a coin flip
independent of price, cart, stage, dwell and wallet. That is fatal for a
recovery pipeline in two separate ways. No detector can beat chance against
noise, so "we flagged at-risk sessions" would be a claim about nothing. And an
intervention that moved outcomes would be moving the coin, which is circular
rather than causal.

Quitting is now a hazard evaluated at each decision point, driven by pressures
that accumulate during a session:

    P(quit) = sigmoid( B0
                     + B_PRICE   * (cart_value / balance) * price_sensitivity
                     + B_DWELL   * (dwell / DWELL_SCALE)
                     + B_REVISIT * revisits
                     - B_INTENT  * intent )

Three properties are deliberate.

**The drivers are only partly observable.** A detector reading the event stream
sees dwell, revisits, cart value and balance. It cannot see ``intent`` or
``price_sensitivity``, which are drawn per session and never emitted. That gap
is what makes detection a real inference problem with irreducible error. Expose
everything and you build an oracle, and an oracle proves nothing.

**Abandonment splits into recoverable and unrecoverable.** Every hazard records
which term dominated it. A session that quit under price pressure is one a
discount can move; a session that quit because intent was low was never going to
buy, and no discount fixes that. This is the ceiling on what any intervention
can achieve, and reporting it is what separates a measured result from a rigged
one -- an intervention that recovers everything is an intervention that cheated.

**An intervention must act through this model, never around it.** A discount
lowers cart_value, which lowers the price term, which lowers the hazard. There
is deliberately no path that sets an outcome directly.
"""
import math
import random as _random
from dataclasses import dataclass, field


class Driver:
    """Which pressure dominated a hazard. Recorded as ground truth."""

    PRICE = "price"            # basket too big for the wallet -- a discount moves it
    HESITATION = "hesitation"  # sat too long in one stage
    CHURN = "churn"            # bounced back and forth between stages
    INTENT = "intent"          # never wanted to buy -- nothing to recover


# Drivers a price intervention can plausibly move. Everything else is the
# honest ceiling: the analysis reports recovery against this subset, not against
# all abandonment.
RECOVERABLE_DRIVERS = frozenset({Driver.PRICE})

# Coefficients, calibrated jointly with Agent.advance_prob so that all-session
# conversion lands in the 2-5% industry band and cart-level abandonment in the
# 70-85% band. tests/test_friction.py fails if either drifts out.
#
# The resulting driver split is roughly 40% price, 36% hesitation and churn, 7%
# intent. That is in the same territory as Baymard's cart-abandonment reasons
# (cost-related causes around half) but is deliberately not tuned to match their
# taxonomy, whose categories -- account creation, delivery speed, trust -- do not
# map onto these three pressures one-for-one. Claiming a fit there would be
# fitting to a number this model does not represent.
B0 = -1.80          # baseline reluctance to quit
B_PRICE = 3.20
B_DWELL = 0.28
B_REVISIT = 0.16
B_INTENT = 2.30

# Dwell is in simulated seconds. With stage dwell now accumulating across
# consecutive actions in a stage (see core.session.enter_stage) it runs to a
# mean of ~5s and a p75 of ~7.5s, so this scales hesitation into the same range
# as the other pressures.
DWELL_SCALE = 3.0

# A cart worth many times the wallet is not meaningfully worse than one worth a
# few times it -- the shopper has already decided. Capping keeps one runaway
# basket from saturating the sigmoid.
MAX_AFFORDABILITY_PRESSURE = 3.0

# Frustration from bouncing between stages saturates too. Uncapped it grew
# linearly and swamped every other pressure, so 85% of abandonment attributed to
# churn regardless of basket size -- which would have understated how much of
# the loss a price intervention could address.
MAX_REVISIT_PRESSURE = 3

# Below this, no single pressure is really responsible: the session quit from
# baseline reluctance because intent was low. Attributing those to whichever
# pressure happened to be largest would invent a cause a discount cannot fix.
MIN_ATTRIBUTABLE_PRESSURE = 0.35

# Latent traits per persona, as (low, high) ranges drawn per session. Power
# buyers mostly intend to buy and barely notice price; browsers rarely intend to
# and notice it acutely.
PERSONA_TRAITS = {
    "power_buyer":   {"intent": (0.55, 0.95), "price_sensitivity": (0.20, 0.60)},
    "average_buyer": {"intent": (0.28, 0.68), "price_sensitivity": (0.50, 1.20)},
    "browser":       {"intent": (0.05, 0.30), "price_sensitivity": (0.90, 1.80)},
}
DEFAULT_TRAITS = PERSONA_TRAITS["average_buyer"]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)          # avoids overflow for large negative x
    return exp_x / (1.0 + exp_x)


@dataclass(frozen=True)
class Hazard:
    """One quit evaluation, with the contribution of each pressure kept.

    The decomposition is the counterfactual ground truth a real deployment can
    never have: it says not just that a session was at risk, but of what.
    """

    probability: float
    contributions: dict = field(default_factory=dict)

    @property
    def driver(self) -> str:
        """The pressure that contributed most to this hazard.

        Only pushes toward quitting count. Intent pulls the other way, so a
        session with no positive pressure at all quit in spite of the model
        rather than because of it -- that is an intent-driven exit.
        """
        pushes = {k: v for k, v in self.contributions.items()
                  if k != Driver.INTENT and v > 0}
        if not pushes:
            return Driver.INTENT
        driver = max(pushes, key=pushes.get)
        if pushes[driver] < MIN_ATTRIBUTABLE_PRESSURE:
            return Driver.INTENT
        return driver

    @property
    def recoverable(self) -> bool:
        return self.driver in RECOVERABLE_DRIVERS


@dataclass
class Friction:
    """The latent state of one shopper in one session.

    Never emitted into the event stream. The detector has to infer around it,
    which is the point.
    """

    intent: float
    price_sensitivity: float

    @classmethod
    def draw(cls, persona: str, rng=_random) -> "Friction":
        traits = PERSONA_TRAITS.get(persona, DEFAULT_TRAITS)
        return cls(
            intent=rng.uniform(*traits["intent"]),
            price_sensitivity=rng.uniform(*traits["price_sensitivity"]),
        )

    def affordability_pressure(self, cart_value: float, balance: float) -> float:
        """How far the basket outruns the wallet, scaled by how much it matters.

        Zero before a cart exists: a shopper who has not chosen anything is not
        yet under price pressure, and treating browse-only visits as
        price-driven would inflate the recoverable share.
        """
        if cart_value <= 0:
            return 0.0
        ratio = cart_value / max(balance, 1.0)
        return min(ratio, MAX_AFFORDABILITY_PRESSURE) * self.price_sensitivity

    def hazard(self, cart_value: float, balance: float, dwell: float,
               revisits: int) -> Hazard:
        """Probability this session ends right now, and what is driving it."""
        contributions = {
            Driver.PRICE: B_PRICE * self.affordability_pressure(cart_value, balance),
            Driver.HESITATION: B_DWELL * (max(dwell, 0.0) / DWELL_SCALE),
            Driver.CHURN: B_REVISIT * min(max(revisits, 0), MAX_REVISIT_PRESSURE),
            Driver.INTENT: -B_INTENT * self.intent,
        }
        return Hazard(
            probability=_sigmoid(B0 + sum(contributions.values())),
            contributions=contributions,
        )

    def should_quit(self, cart_value: float, balance: float, dwell: float,
                    revisits: int, rng=_random):
        """Draw against the hazard. Returns (quit, hazard)."""
        hazard = self.hazard(cart_value, balance, dwell, revisits)
        return rng.random() < hazard.probability, hazard
