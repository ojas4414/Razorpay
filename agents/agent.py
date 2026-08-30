from collections import deque
import random

class Agent:
#the reson we have hsitory limit is  for performance and memory as 50 agents with unbound memroy forever is not feasible
    def __init__(self, agent_id: str, persona: str, history_limit: int = 3):
        self.agent_id = agent_id
        self.persona = persona
        self.history = deque(maxlen=history_limit)
        self.history_limit = history_limit

        self.social_signal = []
        self.action_history = []

        # Persona-driven behavior — lives in Agent, not in simulate.py.
        #
        # advance_prob is how readily this persona moves a stage down the funnel.
        # These were halved from their original values during calibration: the
        # old numbers pushed 43% of sessions into a cart, against a real funnel's
        # 10-15%, which put all-session conversion at ~10% versus the 2-5%
        # industry benchmark the report scores against. They now land at 3.5%.
        #
        # There is no quit_prob. Abandonment is a hazard driven by basket,
        # wallet, hesitation and churn -- see core/friction.py.
        if persona == "power_buyer":
            self.balance = random.uniform(3000, 5000)
            self.advance_prob = 0.225
        elif persona == "browser":
            self.balance = random.uniform(50, 300)
            self.advance_prob = 0.04
        else:  # average_buyer
            self.balance = random.uniform(500, 2000)
            self.advance_prob = 0.15

    def record_action(self, item_id: str):
        self.history.append(item_id)

    def should_advance_funnel(self) -> bool:
        return random.random() < self.advance_prob

    def can_afford(self, price: float) -> bool:
        return self.balance >= price

    def complete_purchase(self, price: float):
        self.balance -= price

    def refill_balance(self):
        """Small top-up when balance is near zero — not the original full budget."""
        if self.persona == "power_buyer":
            self.balance = random.uniform(50, 200)
        elif self.persona == "average_buyer":
            self.balance = random.uniform(20, 100)
        else:  # browser
            self.balance = random.uniform(5, 25)
