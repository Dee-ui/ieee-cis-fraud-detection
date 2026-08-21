"""
Every metric this project reports, in one place.

Three groups:
  1. Ranking metrics: how well the model orders transactions by risk
  2. Review-rate metrics: what you catch at a given manual review budget
  3. The cost model: what a threshold is actually worth in money

Keeping them together means every model is measured identically, so any
difference between two models is the model and not the measuring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(y_true, scores) -> dict:
    """
    How well does the model order transactions by risk?

    PR-AUC (average precision) is the primary metric. Its baseline is the
    fraud rate itself, about 0.035 here, so the lift figure tells you how
    many times better than guessing the model is.

    ROC-AUC is reported because it was the competition metric and is widely
    understood. It is less useful here, because with 569,877 legitimate
    transactions the false positive rate barely moves no matter how many
    real customers you wrongly flag.
    """
    y = np.asarray(y_true)
    prevalence = float(y.mean())
    pr_auc = float(average_precision_score(y, scores))

    return {
        "pr_auc": pr_auc,
        "pr_auc_baseline": prevalence,
        "pr_auc_lift": pr_auc / prevalence if prevalence else float("nan"),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def review_rate_metrics(y_true, scores, review_rate: float) -> dict:
    """
    If the team can review this share of transactions, what do they catch?

    This is the metric a business person actually understands. "We review
    the riskiest 1% and catch 55% of all fraud" is a sentence that needs no
    explanation, unlike an area under a curve.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype="float64")
    n = len(y)

    n_reviewed = max(1, int(round(n * review_rate)))

    # Sort descending. mergesort is stable, so ties always break the same
    # way and the numbers are reproducible between runs.
    order = np.argsort(-s, kind="mergesort")
    reviewed = order[:n_reviewed]

    caught = float(y[reviewed].sum())
    total_fraud = float(y.sum())

    return {
        "review_rate": review_rate,
        "n_reviewed": n_reviewed,
        "threshold": float(s[order[n_reviewed - 1]]),
        "frauds_caught": int(caught),
        "recall": caught / total_fraud if total_fraud else 0.0,
        "precision": caught / n_reviewed,
    }


def cost_curve(
    y_true,
    scores,
    amounts,
    review_cost: float,
    chargeback_fee: float,
    friction_cost: float,
    recovery_rate: float,
) -> pd.DataFrame:
    """
    Total cost at every possible threshold.

    How it works, because the trick is worth knowing. Sort every transaction
    by score, riskiest first. Then "flag the top k" for every k from 0 to n
    is just a running total, and cumulative sums give the count and value of
    fraud caught at every k in a single pass.

    That makes this exact rather than a sample of a few hundred thresholds,
    and it runs in well under a second on 118,000 rows.

    The four outcomes, priced:
      missed fraud   : the amount is lost, plus a chargeback fee
      caught fraud   : a review is paid for, and the part not recovered is lost
      false alarm    : a review is paid for, plus a small friction cost
      correct pass   : nothing

    Costs are weighted by the real transaction amount, so a missed $2,000
    fraud counts for more than a missed $20 one. A flat per-fraud penalty
    cannot express that.
    """
    y = np.asarray(y_true, dtype="float64")
    s = np.asarray(scores, dtype="float64")
    # A blank amount would poison every sum. There should be none, since
    # TransactionAmt has no missing values, but a guard costs nothing.
    a = np.nan_to_num(np.asarray(amounts, dtype="float64"), nan=0.0)

    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted, a_sorted = y[order], s[order], a[order]

    n = len(y_sorted)
    k = np.arange(n + 1)  # 0 flagged, 1 flagged, ... all flagged

    # Prepend a zero so index k means "the first k rows".
    caught = np.concatenate([[0.0], np.cumsum(y_sorted)])
    caught_value = np.concatenate([[0.0], np.cumsum(y_sorted * a_sorted)])

    total_fraud = caught[-1]
    total_fraud_value = caught_value[-1]

    missed = total_fraud - caught
    missed_value = total_fraud_value - caught_value
    false_alarms = k - caught

    cost_missed = missed_value + missed * chargeback_fee
    cost_caught = caught * review_cost + (1 - recovery_rate) * (
        caught_value + caught * chargeback_fee
    )
    cost_false_alarm = false_alarms * (review_cost + friction_cost)

    total_cost = cost_missed + cost_caught + cost_false_alarm

    # Doing nothing at all: every fraud is missed.
    baseline_cost = total_fraud_value + total_fraud * chargeback_fee

    # The threshold that produces exactly k flags is the score of the k-th
    # row. Flagging nothing needs a threshold above every score.
    thresholds = np.concatenate([[np.inf], s_sorted])

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(caught, k, out=np.zeros_like(caught), where=k > 0)

    return pd.DataFrame(
        {
            "n_flagged": k,
            "review_rate": k / n,
            "threshold": thresholds,
            "frauds_caught": caught,
            "frauds_missed": missed,
            "false_alarms": false_alarms,
            "recall": caught / total_fraud if total_fraud else 0.0,
            "precision": precision,
            "cost_missed": cost_missed,
            "cost_caught": cost_caught,
            "cost_false_alarm": cost_false_alarm,
            "total_cost": total_cost,
            "savings": baseline_cost - total_cost,
        }
    )


def best_operating_point(
    curve: pd.DataFrame, capacity_rate: float | None = None
) -> dict:
    """
    Find the cheapest threshold, optionally limited by review capacity.

    Two answers are useful and they are usually different. The unconstrained
    optimum is what the maths wants. The constrained optimum is what the team
    can actually staff. The gap between them is the price of the constraint,
    which is exactly the number to take to a manager when asking for another
    analyst.
    """
    working = curve
    if capacity_rate is not None:
        working = curve[curve["review_rate"] <= capacity_rate]
        if working.empty:
            working = curve.head(1)

    best = working.loc[working["total_cost"].idxmin()]

    return {
        "threshold": float(best["threshold"]),
        "review_rate": float(best["review_rate"]),
        "n_flagged": int(best["n_flagged"]),
        "recall": float(best["recall"]),
        "precision": float(best["precision"]),
        "total_cost": float(best["total_cost"]),
        "savings": float(best["savings"]),
    }


def evaluate(
    y_true,
    scores,
    amounts,
    review_rates: list[float],
    cost_settings: dict,
    capacity_rate: float,
) -> dict:
    """Run every metric at once and return one flat dictionary."""
    results = ranking_metrics(y_true, scores)

    for rate in review_rates:
        point = review_rate_metrics(y_true, scores, rate)
        label = f"{rate:.3%}".rstrip("0").rstrip("%").replace(".", "p")
        results[f"recall_at_{label}pct"] = point["recall"]
        results[f"precision_at_{label}pct"] = point["precision"]

    curve = cost_curve(y_true, scores, amounts, **cost_settings)
    unconstrained = best_operating_point(curve, capacity_rate=None)
    constrained = best_operating_point(curve, capacity_rate=capacity_rate)

    results["best_savings_unconstrained"] = unconstrained["savings"]
    results["best_review_rate_unconstrained"] = unconstrained["review_rate"]
    results["best_savings_within_capacity"] = constrained["savings"]
    results["best_threshold_within_capacity"] = constrained["threshold"]
    results["best_recall_within_capacity"] = constrained["recall"]

    return results


def downsample_curve(curve: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """
    Thin the cost curve before writing it to a file.

    The full curve has one row per transaction, which is 118,109 rows. That
    is right for finding the exact minimum and unnecessary for a CSV nobody
    will read line by line.
    """
    if len(curve) <= max_rows:
        return curve
    step = len(curve) // max_rows
    return curve.iloc[::step].reset_index(drop=True)
