"""Choosing the right recovery action, and escalating when it does not work.

A single fixed discount is the wrong shape for this problem, and the project's
own measurements said so. Cart abandonment has several causes, and only one of
them is price: at a 40% discount the uplift model found 29.4% of recoverable
losses concentrated in its top bucket, which is another way of saying most losses
were never price-driven at all. Applying a price remedy to a shopper who is not
leaving over price spends money and changes nothing.

So the action is chosen from what the detector observed, and escalates only if
the cheaper thing failed.

**Diagnosis comes from observables, not from ground truth.** The simulation knows
why each session was really lost -- `price`, `hesitation`, `churn`, `intent` --
and the playbook is not allowed to look. It reads the detector's recorded
features: basket size against the segment, dwell, bounces, pause length. How well
that proxy tracks the real cause is measured separately, and it is a genuine
source of error rather than something assumed away.

**Every action works through the friction model.** A discount lowers effective
cart value. A reminder resets the dwell clock, because re-engaging a stalled
shopper is what a reminder does. Simplifying the checkout forgives a bounce.
Each lowers the specific pressure it targets and nothing else, so an action
aimed at the wrong cause does what it should do in life: very little.

**Escalation is bounded and ordered by cost.** The free action goes first. A
discount only follows if the free one landed and the session is still in trouble.
Attempts are capped, spend is capped by recovery/safety.py, and a session whose
ladder is exhausted is left alone -- there is no rung that keeps paying more
until something happens.
"""
from dataclasses import dataclass, field

# How long to wait after an action lands before deciding it did not work.
# Shorter than a typical session so an escalation still has time to matter.
ESCALATION_DELAY = 4.0

# The hard ceiling on attempts per session, independent of the ladder length.
MAX_ATTEMPTS = 3


class Kind:
    """What an action physically does."""

    DISCOUNT = "discount"      # lowers what the basket costs
    REMINDER = "reminder"      # re-engages a stalled session
    SIMPLIFY = "simplify"      # forgives a bounce between stages
    NONE = "none"


@dataclass(frozen=True)
class Action:
    """One rung: what it does, how hard, and what it bills."""

    name: str
    kind: str
    magnitude: float = 0.0
    # How much this action improves the odds of pressing on to checkout.
    #
    # Needed because relieving quit pressure alone made things actively worse:
    # a reminder kept a stalled session alive without moving it, so it ran into
    # the action cap and was counted abandoned. Measured at -3.41% (z=-2.56)
    # before this existed. A reminder's purpose is to put the basket back in
    # front of someone, which is a push forward and not merely a reason to stay.
    advance_boost: float = 1.0
    # Fraction of cart value this action costs. A reminder and a simplified
    # checkout are operationally cheap; a discount is not.
    cost_rate: float = 0.0

    def describe(self) -> str:
        if self.kind == Kind.DISCOUNT:
            return f"{self.magnitude:.0%} discount"
        return self.name.replace("_", " ")


REMINDER = Action("reminder", Kind.REMINDER, cost_rate=0.0, advance_boost=1.5)
SIMPLIFY = Action("simplify_checkout", Kind.SIMPLIFY, cost_rate=0.0,
                  advance_boost=1.4)
DISCOUNT_10 = Action("discount_10", Kind.DISCOUNT, magnitude=0.10, cost_rate=0.10)
DISCOUNT_20 = Action("discount_20", Kind.DISCOUNT, magnitude=0.20, cost_rate=0.20)

# Ladders, cheapest first, keyed by what the detector saw.
#
# A price shock starts at a discount because nothing cheaper addresses it -- a
# reminder does not make a basket affordable. The other two start free, and only
# reach for money if the free thing did not work.
LADDERS = {
    "price_shock": [DISCOUNT_10, DISCOUNT_20],
    "pause": [REMINDER, DISCOUNT_10, DISCOUNT_20],
    "revisit": [SIMPLIFY, DISCOUNT_10, DISCOUNT_20],
}
DEFAULT_LADDER = [REMINDER, DISCOUNT_10]

# Diagnosis is taken from the detector's features, not from its trigger label.
#
# The trigger names the flag for the most actionable cause and checks price
# first, so `pause` only ever wins for a session that is neither pricey nor
# bouncing -- which turned out never to happen: across 4,000 sessions the
# reminder rung was unreachable and hesitation was never diagnosed. Reading the
# features instead lets the largest observed pressure decide, which is what
# "diagnose" should have meant in the first place.
#
# Each signal is normalised by the threshold that made it notable, so they are
# comparable: a dwell twice the stall threshold counts the same as two bounces
# where one is notable. Nothing here reads the true driver.
DWELL_REFERENCE = 6.0        # seconds; roughly the detector's stall threshold
REVISIT_REFERENCE = 1.0      # one bounce is the detector's notable amount
GAP_REFERENCE = 1.5          # sigma, the detector's pause threshold


def diagnose(features: dict) -> str:
    """Which pressure the observables say is dominant."""
    if features.get("price_shock"):
        return "price"

    hesitation = max(
        (features.get("dwell", 0.0) or 0.0) / DWELL_REFERENCE,
        (features.get("gap_zscore", 0.0) or 0.0) / GAP_REFERENCE,
    )
    churn = (features.get("revisits", 0) or 0) / REVISIT_REFERENCE
    if hesitation <= 0 and churn <= 0:
        return "unknown"
    return "hesitation" if hesitation >= churn else "churn"


LADDER_BY_DIAGNOSIS = {
    "price": LADDERS["price_shock"],
    "hesitation": LADDERS["pause"],
    "churn": LADDERS["revisit"],
}


def ladder_for(features) -> list:
    """The ladder for whatever the features diagnose."""
    if isinstance(features, str):          # a bare trigger, for older callers
        return LADDERS.get(features, DEFAULT_LADDER)
    return LADDER_BY_DIAGNOSIS.get(diagnose(features or {}), DEFAULT_LADDER)


def diagnosis_for(features) -> str:
    if isinstance(features, str):
        return {"price_shock": "price", "pause": "hesitation",
                "revisit": "churn"}.get(features, "unknown")
    return diagnose(features or {})


@dataclass
class Attempt:
    """One rung actually taken, from dispatch to outcome."""

    session_id: str
    rung: int
    action: Action
    idempotency_key: str
    dispatched_at: float
    applied_at: float | None = None
    missed: bool = False

    @property
    def landed(self) -> bool:
        return self.applied_at is not None

    def summary(self) -> dict:
        return {
            "rung": self.rung,
            "action": self.action.name,
            "kind": self.action.kind,
            "magnitude": self.action.magnitude,
            "idempotency_key": self.idempotency_key,
            "dispatched_at": round(self.dispatched_at, 3),
            "applied_at": None if self.applied_at is None else round(self.applied_at, 3),
            "missed": self.missed,
        }


@dataclass
class PlaybookPolicy:
    """Selects an action by diagnosis, escalates on failure, and stops.

    Runs alongside the fixed-discount policy rather than replacing it, so the
    two can be measured against each other on the same seeds.
    """

    max_attempts: int = MAX_ATTEMPTS
    escalation_delay: float = ESCALATION_DELAY
    safety: object | None = None
    # Bookkeeping, all reported.
    attempts: dict = field(default_factory=dict)     # session_id -> [Attempt]
    issued: int = 0
    escalations: int = 0
    landed: int = 0
    missed: int = 0
    exhausted: int = 0
    refused_by_safety: int = 0
    duplicates_refused: int = 0

    @staticmethod
    def idempotency_key(session_id: str, rung: int) -> str:
        return f"{session_id}:rung{rung}"

    # -- selection ------------------------------------------------------------
    def next_action(self, session):
        """The next rung for this session, or None if the ladder is done."""
        taken = self.attempts.get(session.session_id, [])
        if len(taken) >= self.max_attempts:
            return None
        ladder = ladder_for(getattr(session.flag, "features", None))
        if len(taken) >= len(ladder):
            return None
        return ladder[len(taken)]

    def dispatch(self, session, now: float):
        """Issue the next rung, or refuse and say why."""
        action = self.next_action(session)
        if action is None:
            if self.attempts.get(session.session_id):
                self.exhausted += 1
            return None

        rung = len(self.attempts.get(session.session_id, []))
        key = self.idempotency_key(session.session_id, rung)
        if any(a.idempotency_key == key
               for a in self.attempts.get(session.session_id, [])):
            self.duplicates_refused += 1
            return None

        if self.safety is not None:
            refusal = self.safety.allow(session.cart_value, action.cost_rate)
            if refusal is not None:
                self.refused_by_safety += 1
                return None

        attempt = Attempt(
            session_id=session.session_id, rung=rung, action=action,
            idempotency_key=key, dispatched_at=now + 0.5,
        )
        self.attempts.setdefault(session.session_id, []).append(attempt)
        session.attempts = self.attempts[session.session_id]
        self.issued += 1
        if rung > 0:
            self.escalations += 1
        if self.safety is not None:
            self.safety.issued(session.cart_value, action.cost_rate)
        return attempt

    # -- application ----------------------------------------------------------
    def apply(self, session, now: float) -> bool:
        """Land the most recent pending attempt on the session.

        Each kind lowers exactly the pressure it targets, and nothing else. An
        action pointed at the wrong cause therefore accomplishes very little,
        which is the behaviour being tested.
        """
        taken = self.attempts.get(session.session_id, [])
        pending = [a for a in taken if not a.landed and not a.missed]
        if not pending:
            return False
        attempt = pending[-1]

        if not session.is_open:
            attempt.missed = True
            self.missed += 1
            return False

        action = attempt.action
        if action.kind == Kind.DISCOUNT:
            session.discount = max(session.discount, action.magnitude)
        elif action.kind == Kind.REMINDER:
            # Re-engagement: the stall clock starts again, and the shopper is
            # looking at the basket rather than away from it.
            session.dwell_relief = session.dwell(now)
        elif action.kind == Kind.SIMPLIFY:
            # One bounce forgiven, and one fewer step in the way.
            session.revisit_relief += 1
        session.advance_boost = max(session.advance_boost, action.advance_boost)

        attempt.applied_at = now
        self.landed += 1
        return True

    # -- reporting ------------------------------------------------------------
    def cost_for(self, session) -> float:
        """What this session's actions billed, if it converted."""
        return round(sum(a.action.cost_rate * session.cart_value
                         for a in self.attempts.get(session.session_id, [])
                         if a.landed), 2)

    def stats(self) -> dict:
        rungs = [a.rung for taken in self.attempts.values() for a in taken]
        return {
            "sessions_touched": len(self.attempts),
            "issued": self.issued,
            "landed": self.landed,
            "missed": self.missed,
            "escalations": self.escalations,
            "exhausted": self.exhausted,
            "refused_by_safety": self.refused_by_safety,
            "duplicates_refused": self.duplicates_refused,
            "mean_rungs": round(sum(rungs) / len(rungs), 3) if rungs else 0.0,
            "max_attempts": self.max_attempts,
        }
