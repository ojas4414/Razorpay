"""The recovery action: one bounded discount, dispatched with latency.

Three properties, each of which a real payments deployment would insist on.

**Bounded.** At most one intervention per session, enforced by an idempotency
key rather than by a comment saying so. A second dispatch for the same session
is refused and counted, not silently applied twice.

**Latent.** The discount does not arrive the instant a session is flagged. It is
scheduled into the same event heap the agents use, so it lands some simulated
seconds later -- and if the session has already ended by then, it misses. That
is what makes detect-to-intervene latency a real constraint rather than an
assumption, and it is why the flag-to-outcome window (p95 ~20s) matters.

**Causal only through the model.** Applying a discount lowers the session's
effective cart value. That lowers the affordability pressure in
core/friction.py, which lowers the quit hazard. There is deliberately no branch
anywhere that sets an outcome, marks a session recovered, or touches the
conversion decision. If the discount recovers a session it does so by the same
mechanism that lost it, and tests assert that the only path from intervention to
outcome runs through cart value.

A discount is not free. Every treated session that converts is billed at the
discounted price, including the ones that would have converted anyway. The
analysis has to net that cost off the recovered revenue, which is why the
audit record keeps the discount amount alongside the outcome.
"""
from dataclasses import dataclass, field

# Fraction taken off the cart when a nudge lands.
DEFAULT_DISCOUNT_RATE = 0.15

# Simulated seconds between raising a flag and the discount arriving. Non-zero
# on purpose: an intervention that applied instantly would overstate what a real
# system, with a queue and a delivery channel between detection and offer, can do.
DEFAULT_DISPATCH_LATENCY = 3.0

# The scheduler action name a pending nudge travels under. It is deliberately
# not a funnel stage -- the funnel gate must never see it as one.
NUDGE_ACTION = "nudge"


@dataclass
class Intervention:
    """One discount offer, from decision through to outcome."""

    session_id: str
    kind: str
    magnitude: float
    idempotency_key: str
    flagged_at: float
    dispatched_at: float
    applied_at: float | None = None
    # Set when the nudge arrived after the session had already ended.
    missed: bool = False

    @property
    def landed(self) -> bool:
        return self.applied_at is not None


@dataclass
class InterventionPolicy:
    """Decides the offer, enforces at-most-once, and records what happened."""

    discount_rate: float = DEFAULT_DISCOUNT_RATE
    dispatch_latency: float = DEFAULT_DISPATCH_LATENCY
    # Which detector triggers are worth treating. None means all of them.
    # Narrowing this is the only lever that raises the share of discounts
    # capable of doing anything: a session flagged for hesitation or churn was
    # not leaving over price, so a price offer cannot move it however large.
    eligible_triggers: frozenset | None = None
    # An optional predicate over the session, used for uplift targeting: treat
    # only where the expected lift beats the expected discount bill. Narrowing by
    # trigger can only pick whole categories; this can pick sessions.
    eligible: object | None = None
    _issued: dict = field(default_factory=dict)
    duplicates_refused: int = 0
    missed: int = 0
    ineligible: int = 0

    @staticmethod
    def idempotency_key(session_id: str, kind: str = "discount") -> str:
        return f"{session_id}:{kind}"

    def dispatch(self, session, now: float):
        """Record a pending intervention, or refuse a duplicate.

        Returns the Intervention when one is newly issued, else None. Refusal is
        counted rather than raised: a duplicate is an expected condition in a
        system with retries, not an error.
        """
        key = self.idempotency_key(session.session_id)
        if key in self._issued:
            self.duplicates_refused += 1
            return None

        if self.eligible_triggers is not None:
            trigger = getattr(session.flag, "trigger", None)
            if trigger not in self.eligible_triggers:
                self.ineligible += 1
                return None

        if self.eligible is not None and not self.eligible(session):
            self.ineligible += 1
            return None

        intervention = Intervention(
            session_id=session.session_id,
            kind="discount",
            magnitude=self.discount_rate,
            idempotency_key=key,
            flagged_at=now,
            dispatched_at=now + self.dispatch_latency,
        )
        self._issued[key] = intervention
        session.intervention = intervention
        return intervention

    def apply(self, session, now: float) -> bool:
        """Land a dispatched discount on the session.

        Returns True if it took effect. A nudge arriving after the session has
        closed is recorded as missed and changes nothing -- the session is
        already counted in whichever outcome it reached.
        """
        intervention = session.intervention
        if intervention is None or intervention.landed:
            return False

        if not session.is_open:
            intervention.missed = True
            self.missed += 1
            return False

        session.discount = intervention.magnitude
        intervention.applied_at = now
        return True

    @property
    def issued(self) -> int:
        return len(self._issued)

    def stats(self) -> dict:
        landed = sum(1 for i in self._issued.values() if i.landed)
        return {
            "issued": self.issued,
            "landed": landed,
            "missed": self.missed,
            "duplicates_refused": self.duplicates_refused,
            "ineligible": self.ineligible,
            "discount_rate": self.discount_rate,
            "dispatch_latency": self.dispatch_latency,
        }
