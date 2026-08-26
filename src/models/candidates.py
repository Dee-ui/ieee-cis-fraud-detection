"""
The model candidates and how each one is trained.

Each library handles early stopping differently: LightGBM through callbacks,
XGBoost through the constructor, CatBoost through fit. Rather than spread
those differences through the pipeline, each gets a small adapter here and
the pipeline treats them all identically.

Adding another model later means adding one entry to build_candidates and
changing nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.config import EARLY_STOPPING_ROUNDS, RANDOM_SEED


@dataclass
class Candidate:
    """One model, its settings, and how to fit it."""

    name: str
    flavor: str  # which MLflow logger to use
    build: Callable[[int], Any]  # takes max rounds, returns an estimator
    fit: Callable  # (model, X_tr, y_tr, X_va, y_va) -> (model, best_round)
    params: dict = field(default_factory=dict)
    supports_shap: bool = False


# ---------------------------------------------------------
# Fit adapters
# ---------------------------------------------------------


def _fit_plain(model, X_train, y_train, X_valid, y_valid):
    """For models with no early stopping: the dummy and logistic regression."""
    model.fit(X_train, y_train)
    return model, None


def _fit_lightgbm(model, X_train, y_train, X_valid, y_valid):
    """
    LightGBM takes early stopping as a callback.

    eval_metric "average_precision" is PR-AUC, so training stops when the
    metric we actually care about stops improving, rather than when log loss
    does. Those are not the same point on an imbalanced problem.

    LightGBM 4.7 deprecated the old eval_set argument in favour of separate
    eval_X and eval_y. Rather than guess which form the installed version
    wants, we look at what fit actually accepts. The same approach is used
    for the MLflow log_model change in src/utils/mlflow_utils.py.
    """
    import inspect

    import lightgbm as lgb

    fit_parameters = inspect.signature(model.fit).parameters
    if "eval_X" in fit_parameters:
        evaluation = {"eval_X": X_valid, "eval_y": y_valid}
    else:
        evaluation = {"eval_set": [(X_valid, y_valid)]}

    model.fit(
        X_train,
        y_train,
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
        **evaluation,
    )
    return model, int(model.best_iteration_)


def _fit_xgboost(model, X_train, y_train, X_valid, y_valid):
    """
    XGBoost 2 and later take early stopping in the constructor, not in fit.

    "aucpr" is XGBoost's name for PR-AUC. best_iteration counts from zero,
    so we add one to get a round count.
    """
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=200)
    return model, int(model.best_iteration) + 1


def _fit_catboost(model, X_train, y_train, X_valid, y_valid):
    """
    CatBoost takes the evaluation set in fit and rolls back to the best
    iteration itself when use_best_model is on.
    """
    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        verbose=200,
    )
    return model, int(model.get_best_iteration()) + 1


# ---------------------------------------------------------
# The candidates
# ---------------------------------------------------------


def build_candidates(
    max_rounds: int, include: list[str] | None = None
) -> list[Candidate]:
    """
    Build the list of models to train.

    The two baselines are not filler. The dummy establishes the true floor,
    so every later number has something honest to be measured against. The
    logistic regression forces the boosted trees to earn their complexity
    rather than being assumed better because they are fashionable.
    """
    candidates: list[Candidate] = []

    # --- Baseline 1: predict the same thing for everyone ------------------
    # PR-AUC comes out at the fraud rate and ROC-AUC at exactly 0.5. We fit
    # it rather than asserting those numbers, because a floor you measured
    # is worth more than a floor you assumed.
    candidates.append(
        Candidate(
            name="dummy",
            flavor="sklearn",
            build=lambda rounds: DummyClassifier(strategy="prior"),
            fit=_fit_plain,
            params={"strategy": "prior"},
        )
    )

    # --- Baseline 2: classical linear model -------------------------------
    # Wrapped in a Pipeline because logistic regression cannot handle blanks
    # or wildly different scales, unlike the trees. The imputer and scaler
    # are fitted inside the pipeline, so they learn from training data only.
    candidates.append(
        Candidate(
            name="logistic_regression",
            flavor="sklearn",
            build=lambda rounds: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            fit=_fit_plain,
            params={"max_iter": 1000, "solver": "lbfgs"},
        )
    )

    # --- LightGBM -----------------------------------------------------------
    lightgbm_params = {
        "n_estimators": max_rounds,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbose": -1,
    }
    candidates.append(
        Candidate(
            name="lightgbm",
            flavor="lightgbm",
            build=lambda rounds, p=lightgbm_params: __import__(
                "lightgbm"
            ).LGBMClassifier(**{**p, "n_estimators": rounds}),
            fit=_fit_lightgbm,
            params=lightgbm_params,
            supports_shap=True,
        )
    )

    # --- XGBoost -------------------------------------------------------------
    xgboost_params = {
        "n_estimators": max_rounds,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }
    candidates.append(
        Candidate(
            name="xgboost",
            flavor="xgboost",
            build=lambda rounds, p=xgboost_params: __import__("xgboost").XGBClassifier(
                **{**p, "n_estimators": rounds}
            ),
            fit=_fit_xgboost,
            params=xgboost_params,
            supports_shap=True,
        )
    )

    # --- CatBoost -------------------------------------------------------------
    catboost_params = {
        "iterations": max_rounds,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "eval_metric": "PRAUC",
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "random_seed": RANDOM_SEED,
        "allow_writing_files": False,  # stops CatBoost littering catboost_info/
    }
    candidates.append(
        Candidate(
            name="catboost",
            flavor="catboost",
            build=lambda rounds, p=catboost_params: __import__(
                "catboost"
            ).CatBoostClassifier(**{**p, "iterations": rounds}),
            fit=_fit_catboost,
            params=catboost_params,
            supports_shap=True,
        )
    )

    if include:
        candidates = [c for c in candidates if c.name in include]

    return candidates


def rebuild_for_refit(candidate: Candidate, n_rounds: int):
    """
    Build a fresh copy of a model with a fixed number of rounds and no early
    stopping, for retraining on all labelled data where there is no held-out
    set to stop against.
    """
    model = candidate.build(n_rounds)

    # XGBoost keeps early stopping in the constructor, so it has to be
    # switched off explicitly or fit will demand an eval_set it will not get.
    if candidate.name == "xgboost":
        model.set_params(early_stopping_rounds=None)

    return model


def expanding_window_splits(times: np.ndarray, n_splits: int):
    """
    Cross-validation folds that respect time.

    The time range is cut into equal-sized chunks. Each fold trains on
    everything up to a point and validates on the chunk immediately after,
    so the training window expands with each fold:

        fold 1: train on chunk 1,        validate on chunk 2
        fold 2: train on chunks 1 to 2,  validate on chunk 3
        fold 3: train on chunks 1 to 3,  validate on chunk 4
        ...

    This is the same shape as the real problem repeated several times: learn
    from the past, predict the next period. Ordinary k-fold would train on
    the future and score the past, which is not a thing you can ever do.
    """
    edges = np.quantile(times, np.linspace(0, 1, n_splits + 2))

    for index in range(1, n_splits + 1):
        train_mask = times <= edges[index]
        valid_mask = (times > edges[index]) & (times <= edges[index + 1])
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        yield index, train_mask, valid_mask
