"""
The shapes of what goes in and comes out of the API.

Pydantic models do two jobs at once. They validate incoming data, rejecting
anything malformed with a clear message instead of letting it reach the
model. And FastAPI reads them to build the interactive documentation page,
so the /docs form is generated from these definitions rather than written
by hand.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from config.config import REQUIRED_REQUEST_FIELDS


class TransactionRequest(BaseModel):
    """
    One transaction to score.

    Deliberately a free-form dictionary rather than 434 named fields. The
    raw table has hundreds of anonymised columns, and a caller will rarely
    have all of them. Anything absent becomes blank, which the model handles
    natively at every split.

    The fields in REQUIRED_REQUEST_FIELDS are the exception: without them a
    score would be built on almost nothing, so we reject the request instead
    of returning a confident number.
    """

    transaction: dict[str, Any] = Field(
        ...,
        description="Raw transaction fields. Missing columns are treated as blank.",
        json_schema_extra={
            "example": {
                "TransactionID": 3663549,
                "TransactionDT": 18403224,
                "TransactionAmt": 31.95,
                "ProductCD": "W",
                "card1": 10409,
                "card2": 111.0,
                "card4": "visa",
                "card6": "debit",
                "addr1": 325.0,
                "D1": 14.0,
                "D15": 0.0,
                "C1": 1.0,
                "C13": 1.0,
                "C14": 1.0,
                "P_emaildomain": "gmail.com",
                "DeviceType": "desktop",
            }
        },
    )
    explain: bool = Field(
        default=False,
        description="Return the features that pushed this score up or down.",
    )


class BatchRequest(BaseModel):
    """Several transactions in one call."""

    transactions: list[dict[str, Any]] = Field(..., min_length=1)


class FeatureContribution(BaseModel):
    """One feature's push on one prediction."""

    feature: str
    value: float | None
    contribution: float = Field(
        ..., description="Positive pushes towards fraud, negative away from it."
    )


class PredictionResponse(BaseModel):
    """The answer for one transaction."""

    transaction_id: Any | None = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float
    decision: str = Field(..., description="'review' or 'pass'")
    model_version: str
    explanation: list[FeatureContribution] | None = None


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    model_family: str | None = None
    n_features: int | None = None


class ModelCard(BaseModel):
    """What the service can tell you about the model it is serving."""

    model_family: str
    model_version: str | None
    n_features: int
    threshold: float
    expected_review_rate: float
    validation_pr_auc: float | None
    cv_pr_auc_mean: float | None
    trained_on_rows: int | None
    required_fields: list[str] = REQUIRED_REQUEST_FIELDS
