"""
Tests for the metrics and the cost model.

The cost model is the number the entire business case rests on. If it is
wrong, the annualised savings figure is wrong, and nothing about the code
running successfully would reveal that. So the central test computes a
four-row example by hand and asserts the code agrees.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import (
    best_operating_point,
    cost_curve,
    ranking_metrics,
    review_rate_metrics,
)

COST_SETTINGS = {
    "review_cost": 4.0,
    "chargeback_fee": 25.0,
    "friction_cost": 1.0,
    "recovery_rate": 0.90,
}


def test_perfect_ranking_scores_one():
    """A model that orders every fraud above every legitimate row is perfect."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    result = ranking_metrics(y, scores)

    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_constant_scores_hit_the_known_floor():
    """
    A model that predicts the same value for everyone has learnt nothing.

    On correct metrics that must give ROC-AUC of exactly 0.5 and PR-AUC of
    exactly the fraud rate. This is the same check the dummy model passed in
    the real training run, and it is the one that proves the metric code
    itself is sound.
    """
    y = np.array([0, 0, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])

    result = ranking_metrics(y, scores)

    assert result["roc_auc"] == pytest.approx(0.5)
    assert result["pr_auc"] == pytest.approx(0.25)
    assert result["pr_auc_baseline"] == pytest.approx(0.25)


def test_cost_curve_matches_hand_arithmetic():
    """
    The cost model, worked out on paper.

    Four transactions, sorted by score:
        score 0.9, fraud, $100
        score 0.8, legit, $50
        score 0.7, fraud, $200
        score 0.1, legit, $10

    Total fraud value $300, two frauds, so the baseline cost of doing
    nothing is 300 + 2 x 25 = $350.

    Flagging the top one:
        missed:      1 fraud worth $200  ->  200 + 25       = $225.00
        caught:      1 fraud worth $100  ->  4 + 0.1x(125)  =  $16.50
        false alarm: none                                    =   $0.00
        total                                                = $241.50
        savings      350 - 241.50                            = $108.50
    """
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    amounts = np.array([100.0, 50.0, 200.0, 10.0])

    curve = cost_curve(y, scores, amounts, **COST_SETTINGS)

    # Flagging nothing costs the full baseline and saves nothing.
    assert curve.loc[0, "total_cost"] == pytest.approx(350.0)
    assert curve.loc[0, "savings"] == pytest.approx(0.0)

    # Flagging the top one.
    assert curve.loc[1, "total_cost"] == pytest.approx(241.50)
    assert curve.loc[1, "savings"] == pytest.approx(108.50)
    assert curve.loc[1, "frauds_caught"] == pytest.approx(1.0)
    assert curve.loc[1, "precision"] == pytest.approx(1.0)


def test_cost_curve_catches_every_fraud_at_the_end():
    """Flagging everything must catch every fraud and leave none missed."""
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    amounts = np.array([100.0, 50.0, 200.0, 10.0])

    curve = cost_curve(y, scores, amounts, **COST_SETTINGS)
    last = curve.iloc[-1]

    assert last["frauds_caught"] == pytest.approx(2.0)
    assert last["frauds_missed"] == pytest.approx(0.0)
    assert last["recall"] == pytest.approx(1.0)
    assert last["review_rate"] == pytest.approx(1.0)


def test_capacity_constraint_is_respected():
    """The constrained optimum must never exceed the review budget."""
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.binomial(1, 0.05, n)
    scores = np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1)
    amounts = rng.gamma(2.0, 60.0, n)

    curve = cost_curve(y, scores, amounts, **COST_SETTINGS)

    unconstrained = best_operating_point(curve, capacity_rate=None)
    constrained = best_operating_point(curve, capacity_rate=0.02)

    assert constrained["review_rate"] <= 0.02 + 1e-9
    # A constraint can never help, so the unconstrained answer is at least
    # as good. This catches a sign error in the minimisation.
    assert unconstrained["savings"] >= constrained["savings"] - 1e-6


def test_review_rate_metrics_pick_the_top_scores():
    """At a 50% review rate on four rows, the top two are reviewed."""
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    result = review_rate_metrics(y, scores, 0.5)

    assert result["n_reviewed"] == 2
    assert result["frauds_caught"] == 1
    assert result["recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
