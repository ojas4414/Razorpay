"""Targeting shoppers a discount can move, rather than shoppers who will leave.

The detector answers "who is about to abandon". That is the wrong question, and
the measurements say so: recoverable precision sits at 32-35% *whatever* the
threshold, because abandonment and treatability are different populations. Two
treatments in three land on a shopper no offer could have helped, and every one
of them is a discount paid for nothing.

The right question is "whose outcome would change if we intervened" -- the
incremental effect, not the risk. That is an uplift model:

    uplift(x) = P(convert | treated, x) - P(convert | control, x)

estimated here with a T-learner: one model fitted on the treated arm, one on the
control arm, and the difference between them. The training data is the audit
trail of an experiment that has already run, which is a pleasing loop -- the
record of what was decided becomes the input to deciding better.

**The decision rule falls out of the margin arithmetic.** Treating a session is
worth it when the expected incremental revenue beats the expected discount bill:

    uplift * cart  >  discount_rate * cart * P(convert | treated)

Cart value cancels, leaving `uplift > discount_rate * p_treated`. So the
threshold is not a knob to tune -- it is derived from the offer being made, and
it moves automatically when the offer changes.

Logistic regression in numpy rather than scikit-learn, because scikit-learn is a
development dependency here and the gateway deliberately stays light -- the same
reason inference runs on ONNX rather than torch.

**What makes this checkable:** the simulation records, per session, whether the
loss was actually price-driven. A real deployment never sees that. So the model's
ranking can be scored against ground truth it was never shown, which is the only
honest way to know whether an uplift model has learned anything or has merely
rediscovered the outcome it was fitted on.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np

# Read off the detector's flag record. Deliberately all behavioural: no product
# identity, nothing about who the shopper is beyond a coarse segment.
NUMERIC_FEATURES = ["dwell", "revisits", "gap_zscore", "cart_value",
                    "cart_size", "actions", "elapsed"]
PERSONAS = ["average_buyer", "power_buyer", "browser"]
FEATURE_NAMES = NUMERIC_FEATURES + ["price_shock"] + [f"is_{p}" for p in PERSONAS]


def featurise(features: dict) -> list:
    """One flag's evidence as a fixed-length vector."""
    row = [float(features.get(name, 0.0) or 0.0) for name in NUMERIC_FEATURES]
    row.append(1.0 if features.get("price_shock") else 0.0)
    persona = features.get("persona")
    row.extend(1.0 if persona == p else 0.0 for p in PERSONAS)
    return row


def rows_from_audit(audit_rows) -> tuple:
    """(X, treated, converted) from audit trail records."""
    X, treated, converted = [], [], []
    for row in audit_rows:
        features = row["features_at_flag"]
        if isinstance(features, str):
            features = json.loads(features)
        X.append(featurise(features))
        treated.append(1 if row["arm"] == "treatment" else 0)
        converted.append(1 if row["outcome"] == "converted" else 0)
    return np.asarray(X, dtype=np.float64), np.asarray(treated), np.asarray(converted)


class Logistic:
    """L2-regularised logistic regression, fitted by gradient descent.

    Small and dependency-free on purpose. The signal here is weak and the sample
    is a few thousand rows, so a heavier learner would mostly fit noise.
    """

    def __init__(self, l2=1.0, learning_rate=0.35, epochs=900):
        self.l2 = l2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))),
                        np.exp(np.clip(z, -60, 60)) / (1.0 + np.exp(np.clip(z, -60, 60))))

    def fit(self, X, y):
        n, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        if n == 0:
            return self
        # Start the intercept at the base rate so a weak signal does not have to
        # spend its gradient climbing out of a bad initialisation.
        base = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.bias = math.log(base / (1 - base))

        for _ in range(self.epochs):
            predicted = self._sigmoid(X @ self.weights + self.bias)
            error = predicted - y
            self.weights -= self.learning_rate * ((X.T @ error) / n + self.l2 * self.weights / n)
            self.bias -= self.learning_rate * error.mean()
        return self

    def predict_proba(self, X):
        if self.weights is None:
            return np.zeros(len(X))
        return self._sigmoid(X @ self.weights + self.bias)


@dataclass
class UpliftModel:
    """A T-learner: one model per arm, uplift is the gap between them."""

    treated: Logistic = field(default_factory=Logistic)
    control: Logistic = field(default_factory=Logistic)
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    trained_on: int = 0

    # -- fitting --------------------------------------------------------------
    def _normalise(self, X):
        return (X - self.mean) / self.scale

    @classmethod
    def train(cls, audit_rows, l2=1.0) -> "UpliftModel":
        X, treated, converted = rows_from_audit(audit_rows)
        model = cls()
        if len(X) == 0:
            return model

        model.mean = X.mean(axis=0)
        # Guard against a constant column, which would divide by zero.
        model.scale = np.where(X.std(axis=0) < 1e-9, 1.0, X.std(axis=0))
        Z = model._normalise(X)

        model.treated = Logistic(l2=l2).fit(Z[treated == 1], converted[treated == 1])
        model.control = Logistic(l2=l2).fit(Z[treated == 0], converted[treated == 0])
        model.trained_on = len(X)
        return model

    # -- prediction -----------------------------------------------------------
    def uplift(self, features: dict) -> float:
        """Expected change in conversion probability if this session is treated."""
        if self.mean is None:
            return 0.0
        Z = self._normalise(np.asarray([featurise(features)], dtype=np.float64))
        return float(self.treated.predict_proba(Z)[0] - self.control.predict_proba(Z)[0])

    def treated_probability(self, features: dict) -> float:
        if self.mean is None:
            return 0.0
        Z = self._normalise(np.asarray([featurise(features)], dtype=np.float64))
        return float(self.treated.predict_proba(Z)[0])

    def should_treat(self, features: dict, discount_rate: float) -> bool:
        """Treat only when the expected lift beats the expected discount bill.

        Derived rather than tuned: expected incremental revenue is
        ``uplift * cart``, expected cost is ``discount * cart * P(convert|treated)``,
        and the cart cancels. The bar therefore rises automatically with the size
        of the offer, which is the behaviour you want and would otherwise have to
        remember to implement.
        """
        return self.uplift(features) > discount_rate * self.treated_probability(features)

    # -- scoring --------------------------------------------------------------
    def rank(self, audit_rows) -> list:
        """Sessions ordered by predicted uplift, most treatable first."""
        scored = []
        for row in audit_rows:
            features = row["features_at_flag"]
            if isinstance(features, str):
                features = json.loads(features)
            scored.append((self.uplift(features), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored


def qini_curve(model: UpliftModel, audit_rows, buckets: int = 10) -> list:
    """Cumulative incremental conversions as targeting is widened.

    At each decile of the uplift ranking, the difference in conversion rate
    between the arms among sessions ranked that high. A model that has learned
    something shows most of its gain in the first buckets and flattens; a model
    that has learned nothing traces the diagonal.
    """
    ranked = model.rank(audit_rows)
    if not ranked:
        return []

    curve = []
    per_bucket = max(1, len(ranked) // buckets)
    for index in range(1, buckets + 1):
        head = [row for _, row in ranked[: index * per_bucket]]
        treated = [r for r in head if r["arm"] == "treatment"]
        control = [r for r in head if r["arm"] == "control"]
        if not treated or not control:
            continue
        treat_rate = sum(1 for r in treated if r["outcome"] == "converted") / len(treated)
        control_rate = sum(1 for r in control if r["outcome"] == "converted") / len(control)
        curve.append({
            "fraction_targeted": round(index / buckets, 3),
            "n": len(head),
            "uplift": round(treat_rate - control_rate, 5),
            "incremental": round((treat_rate - control_rate) * len(treated), 2),
        })
    return curve


def ground_truth_precision(model: UpliftModel, audit_rows, top_fraction=0.35) -> dict:
    """Does the ranking actually find the recoverable losses?

    Scored against ``recoverable`` -- whether the loss was price-driven -- which
    the model never sees and which no live system can observe. This is the check
    that separates a model that learned treatability from one that memorised the
    outcome it was fitted on.
    """
    ranked = model.rank(audit_rows)
    lost = [(score, row) for score, row in ranked if row["outcome"] == "abandoned"]
    if not lost:
        return {}

    cut = max(1, int(len(lost) * top_fraction))
    top = [row for _, row in lost[:cut]]
    rest = [row for _, row in lost[cut:]]
    share = lambda group: (sum(1 for r in group if r["recoverable"]) / len(group)
                           if group else 0.0)
    return {
        "top_fraction": top_fraction,
        "recoverable_in_top": round(share(top), 4),
        "recoverable_in_rest": round(share(rest), 4),
        "baseline": round(share(top + rest), 4),
        "n_top": len(top),
        "n_rest": len(rest),
    }
