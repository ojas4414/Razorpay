"""A shopping session: the unit the recovery pipeline detects, treats and scores.

Before this existed the codebase had no session. validation/metrics.py called an
agent's entire lifetime log a "session", which made n=50 for every published
statistic and made "cart abandonment" mean "fraction of agents that never bought
anything" -- a different quantity from the industry figure, measured over a
different denominator.

A Session here is one visit with one terminal outcome. Its lifetime is bounded:
it converts, it is abandoned, or it hits a cap. That bound is what makes
sessions countable, and counting them is what turns a 50-agent simulation into
thousands of observations.

Time is in simulated seconds. The scheduler's timestamps are the clock; one
tick is one second, and every duration on a Session -- dwell, gaps, total length
-- is in those units. Wall-clock time is for display only and never enters a
measurement.
"""
import itertools
from dataclasses import dataclass, field

SECONDS_PER_TICK = 1.0

# A session that neither converts nor quits must still terminate, or it would
# hold an agent out of circulation forever and skew the denominator. Tuned with
# core.friction.B0 so mean session length stays near the real 8.16 items the
# validation path compares against.
MAX_SESSION_ACTIONS = 18

_session_counter = itertools.count(1)


class Outcome:
    CONVERTED = "converted"
    ABANDONED = "abandoned"


class Reason:
    """Why a session ended. Recorded verbatim in the audit trail."""

    PURCHASED = "purchased"
    QUIT = "quit"
    SESSION_CAP = "session_cap"
    INSUFFICIENT_BALANCE = "insufficient_balance"


def reset_session_ids():
    """Restart session numbering. Called when a run is seeded, so that a given
    seed produces the same session IDs and the audit trail is comparable."""
    global _session_counter
    _session_counter = itertools.count(1)


@dataclass
class Session:
    agent_id: int
    persona: str
    started_at: float
    session_id: str = field(default_factory=lambda: f"s{next(_session_counter):07d}")

    stage: str = "browse"
    stage_entered_at: float = 0.0
    last_action_at: float = 0.0

    cart: list = field(default_factory=list)
    cart_value: float = 0.0
    # Every product this session touched, in order. Trend concentration is
    # measured over these; it used to be read off Agent.session_log, which
    # spanned an agent's whole lifetime rather than one visit.
    products: list = field(default_factory=list)

    actions: int = 0
    stage_visits: dict = field(default_factory=dict)
    action_gaps: list = field(default_factory=list)

    # Latent quit pressure for this session (a core.friction.Friction). Held
    # here so it lives and dies with the session, but never emitted into an
    # event -- the detector has to infer around it.
    friction: object | None = None
    # Which pressure dominated at the moment of abandonment. This is
    # counterfactual ground truth: it says not just that a session was lost, but
    # of what, and therefore whether an intervention could ever have moved it.
    quit_driver: str | None = None
    recoverable: bool = False

    outcome: str | None = None
    reason: str | None = None
    ended_at: float | None = None
    revenue: float = 0.0

    def __post_init__(self):
        self.stage_entered_at = self.started_at
        self.last_action_at = self.started_at
        self.stage_visits[self.stage] = 1

    # ── observables ──────────────────────────────────────────────────────────
    def dwell(self, now: float) -> float:
        """Simulated seconds spent in the current stage."""
        return now - self.stage_entered_at

    def elapsed(self, now: float) -> float:
        return now - self.started_at

    def revisits(self, stage: str) -> int:
        """Times this stage was entered *again* after the first visit."""
        return max(0, self.stage_visits.get(stage, 0) - 1)

    # ── transitions ──────────────────────────────────────────────────────────
    def enter_stage(self, stage: str, now: float):
        """Record an action at ``stage``, counting a visit only on a real move.

        Staying put is not a new visit. An agent that browses three times in a
        row has made one continuous visit to browse, so the dwell clock keeps
        running and the visit count does not move -- which is what makes "this
        session has stalled here for N seconds" measurable at all. Resetting on
        every action, as this did originally, pinned dwell to the last
        inter-action gap and made a stall indistinguishable from a pause.

        Leaving and coming back *is* a new visit. That round trip is the
        back-and-forth hesitation looks like from outside, and it is the
        detector's second signal.
        """
        if stage != self.stage:
            self.stage_visits[stage] = self.stage_visits.get(stage, 0) + 1
            self.stage = stage
            self.stage_entered_at = now

    def record_action(self, now: float, product_id: str | None = None):
        self.action_gaps.append(now - self.last_action_at)
        self.last_action_at = now
        self.actions += 1
        if product_id is not None:
            self.products.append(product_id)

    def add_to_cart(self, product_id: str, value: float):
        self.cart.append(product_id)
        self.cart_value = value

    def at_cap(self) -> bool:
        return self.actions >= MAX_SESSION_ACTIONS

    def reached_cart(self) -> bool:
        """Only cart-reaching sessions are recoverable revenue. Browse-only
        visits are excluded from the abandonment denominator for the same
        reason the industry figure excludes them."""
        return bool(self.cart)

    # ── termination ──────────────────────────────────────────────────────────
    def end(self, outcome: str, reason: str, now: float, revenue: float = 0.0):
        self.outcome = outcome
        self.reason = reason
        self.ended_at = now
        self.revenue = revenue
        return self

    @property
    def is_open(self) -> bool:
        return self.outcome is None

    def summary(self) -> dict:
        """Flat record for the validation path and the audit trail."""
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "persona": self.persona,
            "outcome": self.outcome,
            "reason": self.reason,
            "actions": self.actions,
            "reached_cart": self.reached_cart(),
            "cart_size": len(self.cart),
            "products": len(self.products),
            "cart_value": self.cart_value,
            "quit_driver": self.quit_driver,
            "recoverable": self.recoverable,
            "revenue": self.revenue,
            "duration": None if self.ended_at is None else round(self.ended_at - self.started_at, 3),
        }
