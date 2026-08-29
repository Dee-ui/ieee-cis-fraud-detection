"""
Turning a raw transaction into a score.

The same three steps the training pipeline used, in the same order, using
the same saved objects: build the frame, transform, predict.

The service scores one transaction at a time. That is only safe because
transform is row-independent, which the test suite asserts. If any
transformation depended on the other rows in a batch, a single-row request
would silently produce a different answer from the same row inside a batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.config import (
    EXPLANATION_TOP_N,
    ID_COLUMN,
    REQUIRED_REQUEST_FIELDS,
)


class MissingFieldsError(ValueError):
    """Raised when a transaction lacks the fields needed to score it at all."""


def validate_transaction(transaction: dict[str, Any]) -> None:
    missing = [
        field
        for field in REQUIRED_REQUEST_FIELDS
        if field not in transaction or transaction[field] is None
    ]
    if missing:
        raise MissingFieldsError(
            f"missing required fields: {missing}. "
            f"Everything else may be omitted and is treated as blank."
        )


def build_scoring_frame(
    transactions: list[dict[str, Any]], expected_columns: list[str]
) -> pd.DataFrame:
    """
    Build a frame with every column the transformer expects.

    Start with all expected columns blank, then overlay whatever the caller
    actually sent. A caller with six fields and one with four hundred both
    end up with a frame of the right shape, and columns nobody sent stay
    blank, which the model handles natively.

    Anything the caller sends that is not an expected column is ignored
    rather than causing an error, so an upstream system adding a new field
    does not break the service.
    """
    frame = pd.DataFrame(
        {column: pd.Series([np.nan] * len(transactions)) for column in expected_columns}
    )

    for position, transaction in enumerate(transactions):
        for key, value in transaction.items():
            if key in frame.columns:
                frame.at[position, key] = value

    return frame


def score(artifacts, transactions: list[dict[str, Any]]) -> np.ndarray:
    """Probability of fraud for each transaction."""
    from src.serving.artifacts import expected_raw_columns

    frame = build_scoring_frame(transactions, expected_raw_columns(artifacts.engineer))
    features = artifacts.engineer.transform(frame)
    return artifacts.model.predict_proba(features[artifacts.feature_names])[:, 1]


def score_with_features(artifacts, transactions: list[dict[str, Any]]):
    """Score, and also return the feature frame, for when an explanation is wanted."""
    from src.serving.artifacts import expected_raw_columns

    frame = build_scoring_frame(transactions, expected_raw_columns(artifacts.engineer))
    features = artifacts.engineer.transform(frame)[artifacts.feature_names]
    probabilities = artifacts.model.predict_proba(features)[:, 1]
    return probabilities, features


def explain_one(artifacts, features: pd.DataFrame, position: int = 0) -> list[dict]:
    """
    Which features pushed this one prediction up or down.

    SHAP splits a prediction into a contribution per feature. Positive
    contributions pushed towards fraud, negative away from it. This is the
    difference between "the model says 0.87" and "the model says 0.87
    because this card has been seen twice and the amount is 30 times its
    usual", which is the version a human can act on.
    """
    import shap

    explainer = shap.TreeExplainer(artifacts.model)
    values = explainer(features.iloc[[position]])

    array = values.values
    if array.ndim == 3:  # some libraries return one set per class
        array = array[:, :, 1]
    contributions = array[0]

    order = np.argsort(-np.abs(contributions))[:EXPLANATION_TOP_N]

    results = []
    for index in order:
        raw_value = features.iloc[position, index]
        results.append(
            {
                "feature": str(features.columns[index]),
                "value": None if pd.isna(raw_value) else float(raw_value),
                "contribution": float(contributions[index]),
            }
        )
    return results


def decide(probability: float, threshold: float) -> str:
    """
    The operational decision, not just the number.

    The threshold is 0.4222, chosen by the cost model at a 2% review
    capacity. It is deliberately not 0.5: nothing about 0.5 relates to this
    problem, it is just the middle of the range.
    """
    return "review" if probability >= threshold else "pass"


def transaction_id(transaction: dict[str, Any]):
    return transaction.get(ID_COLUMN)
