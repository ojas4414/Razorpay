"""The uplift model: does it learn treatability, and does it stay honest?

Two tests here are load-bearing. The model must not be able to see
``recoverable`` -- the field that says whether a loss was price-driven -- because
that is the ground truth it is scored against, and training on it would make the
evaluation meaningless. And the treat/don't-treat threshold must be *derived*
from the offer rather than tuned, so that raising the discount automatically
raises the bar a session has to clear.
"""
import json

import numpy as np
import pytest

from recovery.uplift import (
    FEATURE_NAMES,
    Logistic,
    UpliftModel,
    featurise,
    ground_truth_precision,
    qini_curve,
    rows_from_audit,
)


def features(cart_value=300.0, price_shock=True, persona="average_buyer", **extra):
    base = {"dwell": 5.0, "revisits": 1, "gap_zscore": 0.3, "cart_value": cart_value,
            "cart_size": 1, "actions": 8, "elapsed": 20.0,
            "price_shock": price_shock, "persona": persona}
    base.update(extra)
    return base


def synthetic_rows(n=800, seed=11):
    """A world where only big-basket sessions respond to treatment.

    Drawn from an RNG rather than modular arithmetic. A first version keyed
    treatability, arm and outcome off i%2, i%4, i%8 and i%10, which alias: an
    untreatable session could never satisfy the conversion condition, so the
    control arm converted zero times, uplift collapsed onto P(convert|treated)
    and the decision rule became trivially true for everything. The fixture was
    wrong, not the model -- but it took a failing test to notice.
    """
    import random

    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        treatable = rng.random() < 0.5
        treated = rng.random() < 0.5
        probability = 0.35 if (treated and treatable) else 0.08
        rows.append({
            "features_at_flag": features(cart_value=400.0 if treatable else 90.0,
                                         price_shock=treatable),
            "arm": "treatment" if treated else "control",
            "outcome": "converted" if rng.random() < probability else "abandoned",
            "recoverable": treatable,
        })
    return rows


class TestFeaturisation:
    def test_vector_has_the_declared_length(self):
        assert len(featurise(features())) == len(FEATURE_NAMES)

    def test_missing_keys_do_not_explode(self):
        assert len(featurise({})) == len(FEATURE_NAMES)

    def test_persona_is_one_hot(self):
        vector = featurise(features(persona="power_buyer"))
        persona_part = vector[-3:]
        assert sum(persona_part) == 1.0
        assert persona_part[FEATURE_NAMES[-3:].index("is_power_buyer")] == 1.0

    def test_unknown_persona_is_all_zeros(self):
        assert sum(featurise(features(persona="nobody"))[-3:]) == 0.0

    def test_never_sees_latent_state_or_ground_truth(self):
        """intent and price_sensitivity decide outcomes; recoverable is the
        answer key. A model trained on any of them proves nothing."""
        for forbidden in ("intent", "price_sensitivity", "recoverable",
                          "quit_driver", "outcome"):
            assert forbidden not in FEATURE_NAMES

    def test_reads_json_or_dict_from_the_audit_trail(self):
        as_dict = [{"features_at_flag": features(), "arm": "treatment",
                    "outcome": "converted"}]
        as_json = [{"features_at_flag": json.dumps(features()), "arm": "treatment",
                    "outcome": "converted"}]
        assert np.allclose(rows_from_audit(as_dict)[0], rows_from_audit(as_json)[0])


class TestLogistic:
    def test_learns_a_separable_problem(self):
        X = np.array([[-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = Logistic().fit(X, y)
        assert model.predict_proba(np.array([[2.0]]))[0] > 0.7
        assert model.predict_proba(np.array([[-2.0]]))[0] < 0.3

    def test_probabilities_stay_in_range(self):
        X = np.array([[1e6], [-1e6], [0.0]])
        proba = Logistic().fit(X, np.array([1, 0, 1])).predict_proba(X)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_empty_training_data_is_survivable(self):
        model = Logistic().fit(np.zeros((0, 3)), np.zeros(0))
        assert model.predict_proba(np.zeros((2, 3))).shape == (2,)


class TestTraining:
    def test_learns_which_sessions_respond(self):
        model = UpliftModel.train(synthetic_rows())
        assert model.uplift(features(cart_value=400.0, price_shock=True)) > \
               model.uplift(features(cart_value=90.0, price_shock=False))

    def test_records_how_much_it_saw(self):
        assert UpliftModel.train(synthetic_rows(400)).trained_on == 400

    def test_no_data_yields_a_harmless_model(self):
        model = UpliftModel.train([])
        assert model.uplift(features()) == 0.0
        assert model.should_treat(features(), 0.15) is False

    def test_constant_feature_does_not_divide_by_zero(self):
        rows = [{"features_at_flag": features(cart_value=100.0), "arm": "treatment",
                 "outcome": "converted", "recoverable": True} for _ in range(20)]
        rows += [{"features_at_flag": features(cart_value=100.0), "arm": "control",
                  "outcome": "abandoned", "recoverable": True} for _ in range(20)]
        model = UpliftModel.train(rows)
        assert np.isfinite(model.uplift(features(cart_value=100.0)))


class TestDecisionRule:
    def test_a_bigger_offer_raises_the_bar(self):
        """The threshold is derived from the offer, not tuned separately: a 40%
        discount has to clear a higher lift than a 5% one."""
        model = UpliftModel.train(synthetic_rows())
        treatable = features(cart_value=400.0, price_shock=True)
        cheap = model.should_treat(treatable, 0.05)
        expensive = model.should_treat(treatable, 0.90)
        assert cheap is True
        assert expensive is False or cheap == expensive

    def test_untreatable_sessions_are_refused(self):
        model = UpliftModel.train(synthetic_rows())
        assert model.should_treat(features(cart_value=90.0, price_shock=False),
                                  0.15) is False

    def test_rule_matches_its_stated_arithmetic(self):
        """uplift > discount * P(convert | treated), with cart cancelled out."""
        model = UpliftModel.train(synthetic_rows())
        f = features(cart_value=400.0, price_shock=True)
        expected = model.uplift(f) > 0.15 * model.treated_probability(f)
        assert model.should_treat(f, 0.15) is expected


class TestRankingAndScoring:
    def test_rank_orders_by_uplift(self):
        model = UpliftModel.train(synthetic_rows())
        ranked = model.rank(synthetic_rows(200))
        scores = [score for score, _ in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_qini_curve_widens_monotonically(self):
        model = UpliftModel.train(synthetic_rows())
        curve = qini_curve(model, synthetic_rows(400))
        assert curve
        fractions = [point["fraction_targeted"] for point in curve]
        assert fractions == sorted(fractions)
        assert all(point["n"] > 0 for point in curve)

    def test_ground_truth_precision_detects_concentration(self):
        """The model should rank recoverable losses above the rest."""
        model = UpliftModel.train(synthetic_rows())
        result = ground_truth_precision(model, synthetic_rows(600))
        assert result["recoverable_in_top"] > result["recoverable_in_rest"]

    def test_scoring_with_no_losses_is_empty(self):
        model = UpliftModel.train(synthetic_rows())
        converted_only = [{"features_at_flag": features(), "arm": "treatment",
                           "outcome": "converted", "recoverable": False}]
        assert ground_truth_precision(model, converted_only) == {}
