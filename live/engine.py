"""
The continuous agent simulation, as an importable engine.

Agents move through the funnel in **sessions**: one visit, one terminal outcome
-- converted, or abandoned with a reason. A session is the unit the recovery
pipeline detects on, intervenes in and scores, and it is what makes the numbers
countable: 50 agents produce 50 sessions per round, so a batch run produces
thousands of observations rather than the 50 agent-lifetimes the validation path
used to call "sessions".

Time is simulated seconds. The scheduler's timestamps are the only clock that
enters a measurement; one tick is one second (core.session.SECONDS_PER_TICK).
Wall-clock time appears in events for display and nothing else.

Event delivery is a generator. The engine yields events and holds no opinion
about where they go; simulate_live.py writes them to a file and Redis, the
gateway pushes them onto a websocket.
"""
import datetime
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import Agent
from agents.funnel import (
    Funnel_graph as FUNNEL_GRAPH,
    build as build_prerequisites,
    allowed as is_action_allowed,
    topo_sort,
)
from agents.policy_runtime import load_policy, sample_with_social
from agents.scheduler import Scheduler
from agents.social_graph import (
    social_graph as build_social_graph,
    bsf as propagate_influence,
    signal_strenght as compute_signal_strength,
)
from core.cart import cart_total
from core.friction import Driver, Friction
from core.session import Outcome, Reason, Session, reset_session_ids
from recovery.assign import TREATMENT, arm_for
from recovery.intervention import NUDGE_ACTION
from data.persona_cluster import assign_agent_personas

N_AGENTS = 50
# The funnel's own topological order, rather than a hand-maintained copy of it.
# funnel.py is the single source of truth for stage sequence; adding a stage
# there now changes the simulation without a second edit here.
ACTIONS = topo_sort(FUNNEL_GRAPH)
WINDOW_SIZE = 3
EPSILON = 0.25
MAX_IN_FILE = 60          # rolling window of events kept for the dashboard
# Completed sessions retained by default. The gateway runs this loop for as long
# as the instance is up, so the list has to be bounded or it grows without limit.
# A batch experiment passes keep_sessions=None to keep every session, because it
# has to score all of them -- and it runs to a fixed size, so it cannot leak.
MAX_KEPT_SESSIONS = 1000
# Default experiment salt; see recovery/assign.py.
DEFAULT_ARM_SALT = "recovery-v1"
CONNECTION_PROB = 0.1
INFLUENCE_MAX_DEPTH = 2
REFILL_THRESHOLD = 10.0
VOCAB_PATH = "vocab.json"

# Gap between one action and the next, in simulated seconds.
ACTION_DELAY_MIN, ACTION_DELAY_MAX = 0.5, 3.0


def load_vocab(path=VOCAB_PATH, window_size=WINDOW_SIZE):
    """Return {product_id: index}, rebuilding it from raw reviews if needed.

    The rebuild path pulls in pandas and torch, so it is imported lazily -- the
    deployed image ships vocab.json and never touches it.
    """
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)

    from agents.bc_trainer import build_vocab
    from data.loader import load_reviews, filter_, sort_split, build, dataset

    frame = load_reviews("data/raw/All_Beauty.jsonl")
    sorted_frame, train_cutoff, val_cutoff = sort_split(filter_(frame))
    pairs = build(sorted_frame, window_size=window_size)
    train_pairs, _, _ = dataset(pairs, train_cutoff, val_cutoff)
    item_to_idx = build_vocab(train_pairs)
    with open(path, "w") as handle:
        json.dump(item_to_idx, handle)
    return item_to_idx


def default_persona_mix(n_agents=N_AGENTS):
    """The 35 average / 10 power / 5 browser split the benchmarks were tuned on."""
    clusters = (
        {f"u{i}": "average_buyer" for i in range(35)}
        | {f"u{i + 35}": "power_buyer" for i in range(10)}
        | {f"u{i + 45}": "browser" for i in range(5)}
    )
    return assign_agent_personas(n_agents, clusters)


class LiveSimulation:
    """Runs agents through the funnel forever, yielding one event at a time."""

    def __init__(self, policy, item_to_idx, n_agents=N_AGENTS, max_events=MAX_IN_FILE,
                 seed=None, keep_sessions=MAX_KEPT_SESSIONS, detector=None,
                 interventions=None, treatment_share=0.5, arm_salt=DEFAULT_ARM_SALT):
        # Seed before anything draws: the social graph, the persona mix and every
        # later action all pull from the `random` module, including
        # policy_runtime.sample_logits. One seed therefore fixes an entire run,
        # which is what makes an experiment re-runnable and its arms comparable.
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            reset_session_ids()

        self.policy = policy
        self.item_to_idx = item_to_idx
        # Mirrors rl_trainer.build_reverse_vocab, inlined so the engine does not
        # have to import a torch module for one dict comprehension.
        self.idx_to_item = {idx: item for item, idx in item_to_idx.items()}
        self.product_ids = list(item_to_idx.keys())
        self.n_agents = n_agents
        self.max_events = max_events
        # Live runs keep a rolling window; a batch experiment passes None and
        # keeps every session, because it has to score all of them.
        self.keep_sessions = keep_sessions
        # Optional recovery.detector.Detector. When absent the engine behaves
        # exactly as before, which is what makes an untreated control arm and a
        # detector-only run the same simulation.
        self.detector = detector
        # Optional recovery.intervention.InterventionPolicy. Requires a detector:
        # there is nothing to treat without a flag.
        self.interventions = interventions
        self.treatment_share = treatment_share
        self.arm_salt = arm_salt

        self.graph = build_social_graph(n_agents, connection_prob=CONNECTION_PROB)
        self.prereqs = build_prerequisites(FUNNEL_GRAPH)
        self.agents = [
            Agent(agent_id=i, persona=persona)
            for i, persona in enumerate(default_persona_mix(n_agents))
        ]

        self.events = []
        self.sessions = []          # completed Session objects
        self.rounds = 0
        self.stats = {
            "total_events": 0,
            "total_purchases": 0,
            "total_spend": 0.0,
            "sessions_started": 0,
            "sessions_converted": 0,
            "sessions_abandoned": 0,
            "sessions_reached_cart": 0,
            "cart_sessions_converted": 0,
            "started_at": datetime.datetime.now().isoformat(),
        }

    @classmethod
    def build(cls, n_agents=N_AGENTS, vocab_path=VOCAB_PATH, **kwargs):
        """Load the policy and vocabulary, then construct the simulation."""
        policy = load_policy()
        return cls(policy, load_vocab(vocab_path), n_agents=n_agents, **kwargs)

    # -- metrics --------------------------------------------------------------
    def cart_abandonment_rate(self):
        """Abandonment over sessions that built a cart -- the industry denominator.

        The old metric divided by every session, browse-only visits included, so
        it was 1 - conversion by construction and not comparable to the ~70%
        figure it was being scored against. This one excludes visits that never
        created a cart, which is what makes the comparison legitimate.
        """
        reached = self.stats["sessions_reached_cart"]
        if not reached:
            return None
        return round(1.0 - self.stats["cart_sessions_converted"] / reached, 4)

    # -- payload --------------------------------------------------------------
    def payload(self, status="running"):
        """The exact shape the dashboard's live feed consumes."""
        elapsed = (
            datetime.datetime.now()
            - datetime.datetime.fromisoformat(self.stats["started_at"])
        ).total_seconds()
        return {
            "status": status,
            "events": list(reversed(self.events)),   # newest first
            "stats": {
                **self.stats,
                "events_per_minute": round((self.stats["total_events"] / max(elapsed, 1)) * 60, 1),
                "elapsed_seconds": round(elapsed),
                "policy_backend": self.policy.backend,
                "agents": self.n_agents,
                "rounds": self.rounds,
                "cart_abandonment_rate": self.cart_abandonment_rate(),
                "seed": self.seed,
            },
        }

    def _record(self, agent, session, action, product_id, sim_time, dwell, price=None):
        event = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),   # display only
            "sim_time": round(sim_time, 3),
            "session_id": session.session_id,
            "agent_id": agent.agent_id,
            "persona": agent.persona,
            "action": action,
            "stage": session.stage,
            "dwell": round(dwell, 3),
            "stage_revisits": session.revisits(session.stage),
            "cart_size": len(session.cart),
            "cart_value": session.cart_value,
            "product": product_id[-10:],     # last 10 chars of the ASIN, for display
            "product_id": product_id,        # full id -- what Kafka consumers expect
            "amount": round(price, 2) if price else None,
            "balance": round(agent.balance, 2),
        }
        self.events.append(event)
        del self.events[:-self.max_events]
        self.stats["total_events"] += 1
        if action == "checkout" and price:
            self.stats["total_purchases"] += 1
            self.stats["total_spend"] += price
        return event

    # -- simulation -----------------------------------------------------------
    def next_product(self, agent):
        history = list(agent.history)
        while len(history) < WINDOW_SIZE:
            history = [None] + history
        state_ids = [
            self.item_to_idx.get(item, 0) if item else 0
            for item in history[-WINDOW_SIZE:]
        ]
        chosen = sample_with_social(
            self.policy.logits(state_ids),
            agent.social_signal,
            self.item_to_idx,
        )
        product_id = self.idx_to_item.get(chosen, self.product_ids[0])
        agent.social_signal = agent.social_signal[-3:]
        if random.random() < EPSILON:
            product_id = random.choice(self.product_ids)
        return product_id

    def propagate_social(self, agent_id, product_id):
        influenced = propagate_influence(self.graph, agent_id, max_depth=INFLUENCE_MAX_DEPTH)
        for influenced_id, signal in compute_signal_strength(influenced, product_id).items():
            self.agents[influenced_id].social_signal.append(signal)

    @staticmethod
    def _delay():
        """Simulated seconds until this agent's next action."""
        return random.uniform(ACTION_DELAY_MIN, ACTION_DELAY_MAX)

    def _start_round(self):
        """Open one session per agent and schedule its first browse."""
        scheduler = Scheduler()
        sessions = {}
        for agent in self.agents:
            start = random.uniform(0, 1.0)
            agent.action_history = []
            agent.history.clear()
            sessions[agent.agent_id] = Session(
                agent_id=agent.agent_id,
                persona=agent.persona,
                started_at=start,
                friction=Friction.draw(agent.persona),
            )
            self.stats["sessions_started"] += 1
            scheduler.push(timestamp=start, agent_id=agent.agent_id, action="browse")
        return scheduler, sessions

    def _close(self, session):
        """Record a terminated session against the run's totals."""
        if session.outcome == Outcome.CONVERTED:
            self.stats["sessions_converted"] += 1
        else:
            self.stats["sessions_abandoned"] += 1
        if session.reached_cart():
            self.stats["sessions_reached_cart"] += 1
            if session.outcome == Outcome.CONVERTED:
                self.stats["cart_sessions_converted"] += 1
        self.sessions.append(session)
        if self.keep_sessions is not None:
            del self.sessions[:-self.keep_sessions]

    def iter_events(self):
        """Yield every agent action forever. One round = one session per agent."""
        while True:
            self.rounds += 1
            scheduler, sessions = self._start_round()

            while scheduler.heap:
                result = scheduler.pop()
                if result is None:
                    break
                sim_time, agent_id, action = result
                agent = self.agents[agent_id]
                session = sessions[agent_id]

                # A nudge is not a funnel action -- it is the recovery offer
                # arriving. It must be handled before the funnel gate, which
                # would reject an action that is not a stage.
                if action == NUDGE_ACTION:
                    if self.interventions is not None:
                        self.interventions.apply(session, sim_time)
                    continue

                if not session.is_open:
                    continue

                # Funnel gate -- an agent cannot check out before it has a cart.
                if not is_action_allowed(agent.action_history, action, self.prereqs):
                    scheduler.push(sim_time + self._delay(), agent_id, "browse")
                    continue

                product_id = self.next_product(agent)

                # How long the agent sat in its current stage before acting. Read
                # before the transition, because that is the hesitation the
                # detector has to see.
                dwell = session.dwell(sim_time)

                price = None
                if action == "checkout":
                    total = session.effective_cart_value()
                    if not agent.can_afford(total):
                        # The most unambiguously price-driven exit there is: the
                        # basket cost more than the wallet held. A discount
                        # moves these by definition, so they count as
                        # recoverable rather than going unattributed.
                        session.quit_driver = Driver.PRICE
                        session.recoverable = True
                        session.end(Outcome.ABANDONED, Reason.INSUFFICIENT_BALANCE, sim_time)
                        self._close(session)
                        continue
                    price = total
                    agent.complete_purchase(total)
                    self.propagate_social(agent_id, product_id)

                session.enter_stage(action, sim_time)
                session.record_action(sim_time, product_id)
                if action == "cart":
                    session.add_to_cart(product_id, cart_total(session.cart + [product_id]))

                agent.record_action(product_id)
                agent.action_history.append(action)

                yield self._record(agent, session, action, product_id, sim_time, dwell, price)

                # Judged on what the stream shows, before the quit draw resolves
                # this action. At most one flag per session.
                if self.detector is not None and session.flag is None:
                    session.flag = self.detector.evaluate(session, sim_time)
                    if session.flag is not None:
                        # Randomised at flag time, so the arms differ only in
                        # whether they were treated. arm_for hashes rather than
                        # drawing, so assignment consumes no randomness and an
                        # untreated session behaves exactly as it would have
                        # with no experiment running at all.
                        session.arm = arm_for(
                            session.session_id, self.arm_salt, self.treatment_share
                        )
                        if session.arm == TREATMENT and self.interventions is not None:
                            issued = self.interventions.dispatch(session, sim_time)
                            if issued is not None:
                                scheduler.push(
                                    issued.dispatched_at, agent_id, NUDGE_ACTION
                                )

                if action == "checkout":
                    session.end(Outcome.CONVERTED, Reason.PURCHASED, sim_time, revenue=price)
                    self._close(session)
                    if agent.balance < REFILL_THRESHOLD:
                        agent.refill_balance()
                    continue

                # Quitting is a hazard, not a coin flip: it rises with the
                # basket outrunning the wallet, with hesitation, and with
                # bouncing between stages. The hazard also records which of
                # those dominated, which is what lets the analysis separate
                # abandonment a discount could move from abandonment it could
                # not. See core/friction.py.
                quit_now, hazard = session.friction.should_quit(
                    cart_value=session.effective_cart_value(),
                    balance=agent.balance,
                    dwell=session.dwell(sim_time),
                    revisits=session.revisits(session.stage),
                )
                if quit_now:
                    session.quit_driver = hazard.driver
                    session.recoverable = hazard.recoverable
                    session.end(Outcome.ABANDONED, Reason.QUIT, sim_time)
                    self._close(session)
                    continue

                if session.at_cap():
                    # Hitting the cap is the simulation bounding a session, not
                    # a shopper deciding anything -- but the session was under
                    # real pressure when it got there, so attribute it the same
                    # way rather than leaving a sixth of abandonment with no
                    # cause recorded.
                    capped = session.friction.hazard(
                        cart_value=session.effective_cart_value(),
                        balance=agent.balance,
                        dwell=session.dwell(sim_time),
                        revisits=session.revisits(session.stage),
                    )
                    session.quit_driver = capped.driver
                    session.recoverable = capped.recoverable
                    session.end(Outcome.ABANDONED, Reason.SESSION_CAP, sim_time)
                    self._close(session)
                    continue

                if agent.should_advance_funnel():
                    next_action = ACTIONS[ACTIONS.index(action) + 1]
                else:
                    next_action = "browse"
                scheduler.push(sim_time + self._delay(), agent_id, next_action)
