"""
Tests for the prediction service.

These use synthetic artefacts rather than the real ones, so they run in CI
with no dataset and no model files, the same as every other test.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.serving.scoring import (
    MissingFieldsError,
    build_scoring_frame,
    decide,
    validate_transaction,
)


def test_missing_required_fields_are_rejected():
    """A transaction with almost nothing in it must be refused, not scored."""
    with pytest.raises(MissingFieldsError) as error:
        validate_transaction({"TransactionAmt": 50.0})

    # The message must name what is missing, or the caller cannot fix it.
    assert "TransactionID" in str(error.value)


def test_complete_transaction_passes_validation():
    validate_transaction(
        {
            "TransactionID": 1,
            "TransactionDT": 86400,
            "TransactionAmt": 50.0,
            "ProductCD": "W",
            "card1": 1000,
        }
    )


def test_scoring_frame_has_every_expected_column():
    """
    A caller sending three fields must still produce a full-width frame.

    Columns nobody sent stay blank, which the model handles natively. This
    is what lets the API accept a partial transaction instead of demanding
    all 434 raw columns.
    """
    expected = ["TransactionAmt", "card1", "addr1", "V1", "V2"]
    frame = build_scoring_frame(
        [{"TransactionAmt": 50.0, "card1": 1000, "unknown_field": "ignored"}],
        expected,
    )

    assert list(frame.columns) == expected
    assert frame.loc[0, "TransactionAmt"] == 50.0
    assert np.isnan(frame.loc[0, "V1"])
    assert "unknown_field" not in frame.columns  # extra fields ignored, not fatal


def test_scoring_frame_handles_several_transactions():
    frame = build_scoring_frame(
        [{"card1": 1}, {"card1": 2}, {"card1": 3}], ["card1", "addr1"]
    )
    assert len(frame) == 3
    assert frame["card1"].tolist() == [1, 2, 3]


def test_scoring_frame_accepts_text_fields():
    """
    A transaction with text fields like DeviceType must not blow up the
    frame. Columns default to float64 in pandas unless told otherwise, and
    writing a string into a float64 column raises rather than upcasting.
    """
    expected = ["TransactionAmt", "DeviceType", "ProductCD", "card4"]
    frame = build_scoring_frame(
        [
            {
                "TransactionAmt": 31.95,
                "DeviceType": "desktop",
                "ProductCD": "W",
                "card4": "visa",
            }
        ],
        expected,
    )

    assert frame.loc[0, "DeviceType"] == "desktop"
    assert frame.loc[0, "ProductCD"] == "W"


def test_decision_uses_the_threshold_not_a_half():
    """
    The threshold is 0.4222, chosen by the cost model. Nothing about 0.5
    relates to this problem, and a score of 0.45 must be reviewed.
    """
    assert decide(0.45, 0.4222) == "review"
    assert decide(0.40, 0.4222) == "pass"
    assert decide(0.4222, 0.4222) == "review"  # exactly at the line counts


def test_service_scores_a_transaction_end_to_end(fitted_engineer, synthetic_joined):
    """
    The full path: raw transaction in, probability out.

    Uses the synthetic transformer from conftest and a stand-in model, so it
    needs no real artefacts and runs in CI.
    """
    from dataclasses import dataclass

    from sklearn.dummy import DummyClassifier

    from src.serving.scoring import score

    features = fitted_engineer.transform(synthetic_joined)
    model = DummyClassifier(strategy="prior").fit(features, synthetic_joined["isFraud"])

    @dataclass
    class FakeArtifacts:
        engineer: object
        model: object
        metadata: dict

        @property
        def feature_names(self):
            return list(self.metadata["feature_names"])

    artifacts = FakeArtifacts(
        engineer=fitted_engineer,
        model=model,
        metadata={"feature_names": fitted_engineer.feature_names_},
    )

    transaction = synthetic_joined.iloc[0].to_dict()
    probabilities = score(artifacts, [transaction])

    assert len(probabilities) == 1
    assert 0.0 <= probabilities[0] <= 1.0
