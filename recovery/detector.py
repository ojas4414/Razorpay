"""Flagging a session as at risk of abandonment.

A session is flagged on either of two independent routes:

**Stall plus hesitation.** The session has sat in its current stage longer than
converting sessions normally do, *and* it is bouncing between stages or has
broken its own rhythm. Elapsed time alone is a weak signal -- a slow shopper is
not a leaving shopper -- so neither half is sufficient by itself.

**Price shock.** The basket is unusually large for this shopper's segment,
whether or not the session stalled.

Hesitation is measured against the session's own earlier rhythm rather than a
global baseline, so a naturally slow shopper is not flagged for being slow, only
for slowing *down*.

The price-shock route was added after measurement, not designed in, and the
reason is worth recording because it is a fact about the problem rather than
about this code. A stall-and-hesitation detector reached 100% of
hesitation-driven abandonment and 70% of churn-driven -- unsurprisingly, since
those are literally its two signals -- but only 40.7% of the price-driven
abandonment that is the only kind a discount can move. It was a mirror of its
own inputs, pointed at the sessions it could not help.

Adding price shock *inside* the stall conjunction changed nothing at all: 99% of
the missed price-driven sessions had failed the stall gate before the second
condition was ever evaluated. Shoppers who leave over price do not dither --
they see the total and go. Requiring a stall therefore excludes, structurally,
the population the intervention exists to serve, so price shock has to be its
own route in.

The threshold is per-persona because the same basket means different things to
different shoppers: price-driven exits average 550.64 for power buyers against
331.98 for average buyers. Segment-relative basket size is something any
merchant observes from order history, without knowing a shopper's wallet.

Both signals are read from what the engine emits. Neither can see the latent
intent or price sensitivity that actually decide the outcome (core/friction.py),
so the detector is inferring under genuine uncertainty -- which is the only way
its precision and recall mean anything.

Only sessions that have built a cart are considered. Browse-only visits are not
recoverable revenue, and including them would inflate the denominator in exactly
the way the old abandonment metric did.

One approximation is worth stating plainly. The simulation observes a session
only when it acts, so a stall is detected on the action that *ends* it rather
than while it is happening. A production detector reading a live event stream
would fire on a timer instead. This makes the measured detect-to-intervene
latency optimistic, and the intervention path models a dispatch delay to
compensate.
"""
import statistics
from dataclasses import dataclass, field

# A session needs a few gaps before "unusual for this session" means anything.
MIN_GAPS_FOR_BASELINE = 4

# Floor on the spread used for the gap z-score, as a fraction of the session's
# own mean gap. Every shopper is treated as having at least this much natural
# variation; see gap_zscore for why a zero floor silently killed the signal.
MIN_SPREAD_FRACTION = 0.10

# Defaults, replaced by calibrate() against a held-out run. They are here so the
# detector is usable without calibration, not because these numbers are right.
DEFAULT_DWELL_SECONDS = 7.5
DEFAULT_MIN_REVISITS = 1
DEFAULT_GAP_SIGMA = 1.5
DEFAULT_CART_SHOCK = 340.0
# Used when a persona was not seen during calibration.
FALLBACK_PERSONA = "__default__"


@dataclass(frozen=True)
class Thresholds:
    """What counts as a stall and as hesitation."""

    dwell_seconds: float = DEFAULT_DWELL_SECONDS
    min_revisits: int = DEFAULT_MIN_REVISITS
    gap_sigma: float = DEFAULT_GAP_SIGMA
    # persona -> cart value above which the basket counts as a shock.
    cart_value_shock: dict = field(default_factory=dict)
    calibrated_on: str = "defaults"

    def shock_for(self, persona: str) -> float:
        return self.cart_value_shock.get(
            persona,
            self.cart_value_shock.get(FALLBACK_PERSONA, DEFAULT_CART_SHOCK),
        )

    def describe(self) -> str:
        shocks = ", ".join(
            f"{name}>{value:.0f}"
            for name, value in sorted(self.cart_value_shock.items())
            if name != FALLBACK_PERSONA
        )
        return (f"(dwell>{self.dwell_seconds:.2f}s AND "
                f"(revisits>={self.min_revisits} OR gap>{self.gap_sigma:.2f}sigma))"
                f" OR cart shock [{shocks}]  [{self.calibrated_on}]")


@dataclass(frozen=True)
class Flag:
    """A session judged at risk, with the evidence that judged it.

    ``features`` is kept so every decision can be replayed and explained after
    the fact -- the property a payments deployment needs far more than the flag
    itself.
    """

    session_id: str
    sim_time: float
    stage: str
    trigger: str
    features: dict = field(default_factory=dict)


def gap_zscore(action_gaps) -> float:
    """How unusual the latest pause is against this session's earlier ones.

    The spread is floored at a fraction of the session's own mean gap. Without
    that floor a perfectly regular shopper -- identical gaps, zero variance --
    divided by zero, and the guard against it returned 0.0, scoring the most
    obvious outlier available (steady rhythm, then one long pause) as perfectly
    ordinary. Allowing every shopper a little natural variation keeps the ratio
    finite and the signal alive.
    """
    if len(action_gaps) < MIN_GAPS_FOR_BASELINE:
        return 0.0
    *prior, latest = action_gaps
    baseline = statistics.mean(prior)
    if baseline <= 0:
        return 0.0
    spread = max(statistics.pstdev(prior), MIN_SPREAD_FRACTION * baseline)
    return (latest - baseline) / spread


@dataclass
class Detector:
    thresholds: Thresholds = field(default_factory=Thresholds)

    def evaluate(self, session, now: float):
        """Return a Flag if this session is at risk right now, else None."""
        if not session.reached_cart() or not session.is_open:
            return None

        dwell = session.dwell(now)
        revisits = session.revisits(session.stage)
        z = gap_zscore(session.action_gaps)

        stalled = dwell > self.thresholds.dwell_seconds
        bounced = revisits >= self.thresholds.min_revisits
        paused = z > self.thresholds.gap_sigma
        pricey = session.cart_value > self.thresholds.shock_for(session.persona)

        # Two independent routes. Price shock is deliberately not gated behind
        # the stall: shoppers who leave over price do not dither first.
        if not (pricey or (stalled and (bounced or paused))):
            return None

        # Named for the most actionable cause: a big basket is the one a
        # discount actually addresses.
        trigger = "price_shock" if pricey else ("revisit" if bounced else "pause")

        return Flag(
            session_id=session.session_id,
            sim_time=now,
            stage=session.stage,
            trigger=trigger,
            features={
                "dwell": round(dwell, 3),
                "revisits": revisits,
                "gap_zscore": round(z, 3),
                "cart_value": session.cart_value,
                "cart_size": len(session.cart),
                "price_shock": pricey,
                "actions": session.actions,
                "elapsed": round(session.elapsed(now), 3),
                "persona": session.persona,
            },
        )


def calibrate(events, sessions, percentile: int = 75) -> Thresholds:
    """Derive thresholds from how *converting* sessions behave.

    The stall threshold is the given percentile of stage dwell among sessions
    that went on to convert: longer than that is longer than a healthy session
    normally sits. Deriving it this way is the difference between a threshold
    with an answer to "why that number" and a magic constant.

    Must be run on a different seed from the one being measured. Tuning a
    threshold on the sessions you then score it against is fitting to the test
    set, and the precision it reports would be fiction.
    """
    converted = {s.session_id for s in sessions if s.outcome == "converted"}
    if not converted:
        return Thresholds()

    dwells = [e["dwell"] for e in events
              if e["session_id"] in converted and e["cart_size"] > 0]
    revisits = [e["stage_revisits"] for e in events
                if e["session_id"] in converted and e["cart_size"] > 0]
    if not dwells:
        return Thresholds()

    # The shock cut comes from the whole cart-bearing population per persona,
    # not from converting sessions: "unusually large for this segment" is a
    # statement about what that segment normally carries, not about what
    # succeeded. Segment matters because the same basket is trivial to a power
    # buyer and prohibitive to a browser.
    carts_by_persona = {}
    for event in events:
        if event["cart_size"] > 0:
            carts_by_persona.setdefault(event["persona"], []).append(event["cart_value"])

    shocks = {persona: round(_percentile(values, percentile), 2)
              for persona, values in carts_by_persona.items()}
    all_carts = [v for values in carts_by_persona.values() for v in values]
    if all_carts:
        shocks[FALLBACK_PERSONA] = round(_percentile(all_carts, percentile), 2)

    return Thresholds(
        dwell_seconds=round(_percentile(dwells, percentile), 3),
        min_revisits=max(1, int(_percentile(revisits, percentile))),
        gap_sigma=DEFAULT_GAP_SIGMA,
        cart_value_shock=shocks,
        calibrated_on=f"p{percentile} of {len(dwells)} converting-session actions",
    )


def _percentile(values, percentile):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int(len(ordered) * percentile / 100))
    return ordered[index]
