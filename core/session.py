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
# hold an agent out of circulation forever and skew the denominator.
MAX_SESSION_ACTIONS = 30

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

    actions: int = 0
    stage_visits: dict = field(default_factory=dict)
    action_gaps: list = field(default_factory=list)

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
        """Move to ``stage`` and count the visit.

        Every entry increments the count, so revisits() is simply one less than
        the number of visits. Going back to a stage already seen is the
        back-and-forth that hesitation looks like from the outside, and it is
        the detector's second signal.
        """
        self.stage_visits[stage] = self.stage_visits.get(stage, 0) + 1
        self.stage = stage
        self.stage_entered_at = now

    def record_action(self, now: float):
        self.action_gaps.append(now - self.last_action_at)
        self.last_action_at = now
        self.actions += 1

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
            "cart_value": self.cart_value,
            "revenue": self.revenue,
            "duration": None if self.ended_at is None else round(self.ended_at - self.started_at, 3),
        }
