"""Guards that must exist before anything like this touches real money.

An automated system that hands out discounts is a system that can spend without
a human in the loop. Three failures are worth designing against explicitly, and
none of them is hypothetical -- each has happened to a real promotions engine.

**A bug that treats everyone.** A detector regression, a threshold read as zero,
a bad deploy: suddenly every session is flagged and every basket is discounted.
The spend cap stops this in minutes rather than at the end of the day.

**A campaign that runs away.** Traffic spikes, the offer rate is fine per session
but the absolute volume is not. The rate limit bounds offers per window
regardless of what the model wants.

**Needing to stop right now.** When something is wrong the first requirement is
an off switch that takes effect immediately, does not need a deploy, and records
who threw it and why.

Every refusal is counted and attributed. "The discount engine stopped issuing at
14:32 because the hourly spend cap was reached" is an answer; silently doing
nothing is not.

The limits here are in-process and reset when the process does, which is
adequate for a simulation and *not* adequate for production -- a real deployment
needs shared state so the cap holds across instances. That gap is stated rather
than papered over.
"""
import time
from collections import deque
from dataclasses import dataclass, field


class Refusal:
    """Why an offer was not issued."""

    KILLED = "kill_switch"
    RATE_LIMITED = "rate_limit"
    BUDGET_EXHAUSTED = "spend_cap"


@dataclass
class KillSwitch:
    """A switch a human can throw without a deploy."""

    engaged: bool = False
    reason: str = ""
    engaged_by: str = ""
    engaged_at: float | None = None

    def engage(self, reason: str, by: str = "operator"):
        self.engaged = True
        self.reason = reason
        self.engaged_by = by
        self.engaged_at = time.time()

    def release(self, by: str = "operator"):
        self.engaged = False
        self.reason = ""
        self.engaged_by = by
        self.engaged_at = None

    def status(self) -> dict:
        return {"engaged": self.engaged, "reason": self.reason,
                "engaged_by": self.engaged_by, "engaged_at": self.engaged_at}


@dataclass
class SpendGuard:
    """Rate and spend limits over a sliding window.

    Both are needed. A rate limit alone allows a small number of enormous
    discounts; a spend cap alone allows a flood of tiny ones, each cheap and
    collectively a denial of service on your own margin.
    """

    max_offers_per_window: int = 500
    max_spend_per_window: float = 50_000.0
    window_seconds: float = 3600.0
    # Refuse any single discount above this, whatever the model asks for. A
    # model that requests a 90% discount is a model that is broken.
    max_discount_rate: float = 0.40

    _events: deque = field(default_factory=deque)
    refusals: dict = field(default_factory=lambda: {
        Refusal.KILLED: 0, Refusal.RATE_LIMITED: 0, Refusal.BUDGET_EXHAUSTED: 0})

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def offers_in_window(self, now=None) -> int:
        self._prune(now if now is not None else time.time())
        return len(self._events)

    def spend_in_window(self, now=None) -> float:
        self._prune(now if now is not None else time.time())
        return round(sum(amount for _, amount in self._events), 2)

    def check(self, expected_spend: float, discount_rate: float, now=None):
        """Return a Refusal reason, or None if the offer may proceed."""
        now = now if now is not None else time.time()
        self._prune(now)

        if discount_rate > self.max_discount_rate:
            self.refusals[Refusal.BUDGET_EXHAUSTED] += 1
            return Refusal.BUDGET_EXHAUSTED
        if len(self._events) >= self.max_offers_per_window:
            self.refusals[Refusal.RATE_LIMITED] += 1
            return Refusal.RATE_LIMITED
        if self.spend_in_window(now) + expected_spend > self.max_spend_per_window:
            self.refusals[Refusal.BUDGET_EXHAUSTED] += 1
            return Refusal.BUDGET_EXHAUSTED
        return None

    def record(self, spend: float, now=None):
        """Book an issued offer against the window."""
        self._events.append((now if now is not None else time.time(), spend))

    def status(self, now=None) -> dict:
        return {
            "offers_in_window": self.offers_in_window(now),
            "spend_in_window": self.spend_in_window(now),
            "max_offers_per_window": self.max_offers_per_window,
            "max_spend_per_window": self.max_spend_per_window,
            "window_seconds": self.window_seconds,
            "max_discount_rate": self.max_discount_rate,
            "refusals": dict(self.refusals),
        }


@dataclass
class SafetyGate:
    """Kill switch and spend guard, checked together.

    Wired into InterventionPolicy as an eligibility check, so an offer passes
    through exactly the same gate whether the caller is the simulation, the
    dashboard, or something later that is talking to real traffic.
    """

    kill_switch: KillSwitch = field(default_factory=KillSwitch)
    guard: SpendGuard = field(default_factory=SpendGuard)

    def allow(self, cart_value: float, discount_rate: float, now=None):
        if self.kill_switch.engaged:
            self.guard.refusals[Refusal.KILLED] += 1
            return Refusal.KILLED
        return self.guard.check(cart_value * discount_rate, discount_rate, now)

    def issued(self, cart_value: float, discount_rate: float, now=None):
        self.guard.record(cart_value * discount_rate, now)

    def status(self, now=None) -> dict:
        return {"kill_switch": self.kill_switch.status(),
                "spend_guard": self.guard.status(now)}
