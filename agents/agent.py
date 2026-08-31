from collections import deque
import random

# Persona parameters, as tables rather than branches in __init__.
#
# advance_prob is this persona's baseline willingness to move a stage down the
# funnel. It is only a baseline: the engine multiplies it by
# core.friction.advance_multiplier, so a basket that outruns the wallet drags
# progression down and a discount lifts it back. That is the channel through
# which a recovery offer can reach an outcome.
#
# These were halved from their original values during calibration -- the old
# numbers pushed 43% of sessions into a cart against a real funnel's 10-15%,
# putting all-session conversion at ~10% against the 2-5% industry benchmark.
ADVANCE_PROB = {
    "power_buyer": 0.225,
    "average_buyer": 0.15,
    "browser": 0.04,
}

BALANCE_RANGE = {
    "power_buyer": (3000, 5000),
    "average_buyer": (500, 2000),
    "browser": (50, 300),
}

DEFAULT_PERSONA = "average_buyer"

class Agent:
#the reson we have hsitory limit is  for performance and memory as 50 agents with unbound memroy forever is not feasible
    def __init__(self, agent_id: str, persona: str, history_limit: int = 3):
        self.agent_id = agent_id
        self.persona = persona
        self.history = deque(maxlen=history_limit)
        self.history_limit = history_limit

        self.social_signal = []
        self.action_history = []

        # Persona-driven behaviour, read from the tables above. There is no
        # quit_prob: abandonment is a hazard driven by basket, wallet,
        # hesitation and churn -- see core/friction.py.
        low, high = BALANCE_RANGE.get(persona, BALANCE_RANGE[DEFAULT_PERSONA])
        self.balance = random.uniform(low, high)
        self.advance_prob = ADVANCE_PROB.get(persona, ADVANCE_PROB[DEFAULT_PERSONA])

    def record_action(self, item_id: str):
        self.history.append(item_id)

    def can_afford(self, price: float) -> bool:
        return self.balance >= price

    def complete_purchase(self, price: float):
        self.balance -= price

    def new_session_balance(self):
        """Redraw this shopper's spending power for a fresh visit.

        Balance used to carry across sessions, topped up only after a successful
        checkout and only once it fell below 10. That made depletion one-way: an
        agent who spent down to 200 could no longer afford a 350 basket, so never
        checked out, so never refilled, and was stuck for the rest of the run.
        Conversion therefore decayed with run length -- 4.6-5.0% at 2,500
        sessions against 3.4-3.5% at 10,000, consistently across seeds -- so
        every published figure depended on how long the simulation had been
        left running.

        Redrawing per visit makes the run stationary, which is what lets a
        number mean anything. It also matches what a session is: one visit by a
        shopper with their segment's normal spending power, not one instalment
        of a lifetime budget.
        """
        low, high = BALANCE_RANGE.get(self.persona, BALANCE_RANGE[DEFAULT_PERSONA])
        self.balance = random.uniform(low, high)
