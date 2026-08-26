"""
MLflow setup and a small compatibility shim.

MLflow 3 renamed the log_model argument from artifact_path to name. The old
one still works but warns, and which is preferred varies across patch
releases. Rather than guess, the helper below inspects the function and uses
whichever it accepts.

That is a useful habit whenever a library is mid-transition: check what is
actually installed instead of assuming.
"""

from __future__ import annotations

import inspect
from typing import Any

import mlflow

from config.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

# sklearn's flavor saves via skops, which round-trips the model on save to
# verify every object type it contains is on an allow-list (this stops a
# malicious pickle-like file from running code on load). Fitted sklearn
# models can legitimately contain a few types skops doesn't trust by
# default -- numpy.dtype shows up on LogisticRegression, for example.
# Since we trained these models ourselves, trusting them here is safe;
# add to this list if a future model trips over a different type (the
# exception message names the type to add).
SKLEARN_SKOPS_TRUSTED_TYPES = ["numpy.dtype"]


def configure_mlflow(experiment_name: str | None = None) -> str:
    """Point MLflow at the local database and select the experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    name = experiment_name or MLFLOW_EXPERIMENT_NAME
    mlflow.set_experiment(name)
    return MLFLOW_TRACKING_URI


def flavor_module(flavor: str):
    """Return the MLflow logger for a model library."""
    import mlflow.catboost
    import mlflow.lightgbm
    import mlflow.sklearn
    import mlflow.xgboost

    return {
        "sklearn": mlflow.sklearn,
        "lightgbm": mlflow.lightgbm,
        "xgboost": mlflow.xgboost,
        "catboost": mlflow.catboost,
    }[flavor]


def log_model_compatibly(flavor: str, model: Any, name: str, signature=None):
    """
    Log a model, using whichever argument name this MLflow version wants.

    For the sklearn flavor, also passes skops_trusted_types so that
    fitted models containing types like numpy.dtype don't fail MLflow's
    save-time reload verification (see SKLEARN_SKOPS_TRUSTED_TYPES above).
    Only passed if the installed MLflow's log_model actually accepts it,
    following the same "check what is installed" approach as the
    name/artifact_path handling below.

    Returns the ModelInfo object, which carries the model_uri needed to
    register the model afterwards.
    """
    module = flavor_module(flavor)
    parameters = inspect.signature(module.log_model).parameters

    kwargs: dict[str, Any] = {"signature": signature}
    if flavor == "sklearn" and "skops_trusted_types" in parameters:
        kwargs["skops_trusted_types"] = SKLEARN_SKOPS_TRUSTED_TYPES

    if "name" in parameters:
        return module.log_model(model, name=name, **kwargs)
    return module.log_model(model, artifact_path=name, **kwargs)


def log_params_safely(params: dict) -> None:
    """
    Log parameters, keeping each value within MLflow's length limit.

    MLflow rejects very long parameter values. Truncating is better than
    having the whole run fail because one setting was a long list.
    """
    for key, value in params.items():
        text = str(value)
        if len(text) > 480:
            text = text[:477] + "..."
        mlflow.log_param(key, text)


def log_metrics_safely(metrics: dict, prefix: str = "") -> None:
    """Log only the numeric entries, skipping anything MLflow cannot store."""
    for key, value in metrics.items():
        if (
            isinstance(value, (int, float)) and value == value
        ):  # value == value filters NaN
            mlflow.log_metric(f"{prefix}{key}", float(value))
