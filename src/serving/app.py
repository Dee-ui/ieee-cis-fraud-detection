"""
The FastAPI prediction service.

Endpoints:
    GET  /health         is it alive and is the model loaded
    GET  /model          what model is being served, and how it performs
    POST /predict        score one transaction
    POST /predict/batch  score several
    GET  /docs           interactive documentation, generated automatically

The /docs page is worth knowing about: FastAPI builds it from the type
definitions in schemas.py, so anyone can open it in a browser, fill in a
form, and get a real fraud score back without installing anything.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from config.config import (
    ENABLE_EXPLANATIONS,
    MAX_BATCH_SIZE,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from src.serving.artifacts import load_artifacts
from src.serving.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelCard,
    PredictionResponse,
    TransactionRequest,
)
from src.serving.scoring import (
    MissingFieldsError,
    decide,
    explain_one,
    score_with_features,
    transaction_id,
)

# Held at module level so the artefacts are loaded once at startup rather
# than on every request. Loading a 28 MB transformer per request would make
# the service unusably slow.
STATE: dict = {"artifacts": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the service starts and once when it stops.

    Loading here rather than at import time means a failure produces a
    service that reports itself unhealthy, instead of a container that
    crashes on boot with no explanation.
    """
    print(f"Starting {SERVICE_NAME} {SERVICE_VERSION}")
    started = time.time()
    try:
        STATE["artifacts"] = load_artifacts()
        print(f"  Ready in {time.time() - started:.1f}s")
    except Exception as error:  # noqa: BLE001
        STATE["error"] = str(error)
        print(f"  FAILED to load artefacts: {error}")

    yield

    print("Shutting down")


app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description=(
        "Scores card transactions for fraud risk.\n\n"
        "Send a raw transaction and get back a probability, a decision, and "
        "optionally an explanation of what drove the score. Most fields are "
        "optional: anything you leave out is treated as unknown, which the "
        "model handles natively.\n\n"
        "The threshold is not 0.5. It is chosen by a cost model at a 2% "
        "manual review capacity."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


def _artifacts():
    """Fetch the loaded artefacts, or fail with a clear message."""
    if STATE["artifacts"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {STATE['error'] or 'still starting'}",
        )
    return STATE["artifacts"]


@app.get("/", include_in_schema=False)
async def root():
    """Send anyone landing on the root straight to the documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Is the service alive and usable?

    Deliberately never fails, even when the model is missing. A health check
    that errors tells you nothing; one that returns "degraded" with a reason
    tells you what to fix.
    """
    artifacts = STATE["artifacts"]
    return HealthResponse(
        status="ok" if artifacts else "degraded",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        model_loaded=artifacts is not None,
        model_family=artifacts.metadata["model_family"] if artifacts else None,
        n_features=len(artifacts.feature_names) if artifacts else None,
    )


@app.get("/model", response_model=ModelCard)
async def model_card():
    """What is being served, and how well it did when it was measured."""
    artifacts = _artifacts()
    metadata = artifacts.metadata

    return ModelCard(
        model_family=metadata["model_family"],
        model_version=str(metadata.get("registered_version")),
        n_features=len(artifacts.feature_names),
        threshold=artifacts.threshold,
        expected_review_rate=float(metadata.get("chosen_review_rate", 0.02)),
        validation_pr_auc=metadata.get("selection_pr_auc"),
        cv_pr_auc_mean=metadata.get("cv_pr_auc_mean"),
        trained_on_rows=metadata.get("trained_on_rows"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TransactionRequest):
    """Score one transaction."""
    artifacts = _artifacts()

    try:
        from src.serving.scoring import validate_transaction

        validate_transaction(request.transaction)
    except MissingFieldsError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        probabilities, features = score_with_features(artifacts, [request.transaction])
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"scoring failed: {error}"
        ) from error

    probability = float(probabilities[0])

    explanation = None
    if request.explain and ENABLE_EXPLANATIONS:
        try:
            explanation = explain_one(artifacts, features, 0)
        except Exception as error:  # noqa: BLE001
            # An explanation is a bonus. Losing it must not lose the score.
            print(f"  explanation failed: {error}")

    return PredictionResponse(
        transaction_id=transaction_id(request.transaction),
        fraud_probability=probability,
        threshold=artifacts.threshold,
        decision=decide(probability, artifacts.threshold),
        model_version=str(artifacts.metadata.get("registered_version")),
        explanation=explanation,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """
    Score several transactions in one call.

    Capped so a single caller cannot occupy the service indefinitely. A real
    deployment would also rate-limit per client.
    """
    artifacts = _artifacts()

    if len(request.transactions) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"batch of {len(request.transactions)} exceeds the "
            f"limit of {MAX_BATCH_SIZE}",
        )

    try:
        from src.serving.scoring import validate_transaction

        for position, transaction in enumerate(request.transactions):
            try:
                validate_transaction(transaction)
            except MissingFieldsError as error:
                raise HTTPException(
                    status_code=422, detail=f"transaction {position}: {error}"
                ) from error

        probabilities, _ = score_with_features(artifacts, request.transactions)
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"scoring failed: {error}"
        ) from error

    predictions = [
        PredictionResponse(
            transaction_id=transaction_id(transaction),
            fraud_probability=float(probability),
            threshold=artifacts.threshold,
            decision=decide(float(probability), artifacts.threshold),
            model_version=str(artifacts.metadata.get("registered_version")),
        )
        for transaction, probability in zip(
            request.transactions, probabilities, strict=True
        )
    ]

    return BatchResponse(predictions=predictions, count=len(predictions))
