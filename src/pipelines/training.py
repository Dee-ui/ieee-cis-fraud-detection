"""
Model training stage.

Input:  data/processed/train_features.parquet
        data/processed/test_features.parquet
Output: models/final_model.joblib
        models/final_model_metadata.json
        data/processed/kaggle_submission.csv
        reports/model_comparison.csv, threshold_analysis.csv, cost_curve.csv,
                cv_results.csv, feature_importance.csv, training_summary.md
        reports/figures/11 to 15
        reports/explainability/*.png
        Every run recorded in MLflow.

Run with:
    python run.py --step training
    python run.py --step training --quick
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from config.config import (
    COST_CHARGEBACK_FEE,
    COST_FALSE_ALARM_FRICTION,
    COST_REVIEW_PER_CASE,
    CV_N_SPLITS,
    CV_RESULTS_FILE,
    COST_CURVE_FILE,
    EXPLAINABILITY_DIR,
    FEATURE_IMPORTANCE_FILE,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    FIGURES_DIR,
    FINAL_MODEL_FILE,
    FRAUD_RECOVERY_RATE,
    HEADLINE_REVIEW_RATES,
    ID_COLUMN,
    KAGGLE_SUBMISSION_FILE,
    MAX_BOOSTING_ROUNDS,
    MODEL_ALIAS_CANDIDATE,
    MODEL_COMPARISON_FILE,
    MODEL_METADATA_FILE,
    QUICK_BOOSTING_ROUNDS,
    REFERENCE_DATETIME,
    REGISTERED_MODEL_NAME,
    REVIEW_CAPACITY_RATE,
    SHAP_SAMPLE_SIZE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    THRESHOLD_ANALYSIS_FILE,
    TIME_COLUMN,
    TRAINING_SUMMARY_FILE,
    TRAIN_SPLIT_LABEL,
    UID_ABLATION_TOLERANCE,
    UID_FEATURE_MARKERS,
    VALID_SPLIT_LABEL,
    RANDOM_SEED,
    ensure_directories,
)
from src.models.candidates import (
    build_candidates,
    expanding_window_splits,
    rebuild_for_refit,
)
from src.utils.metrics import (
    best_operating_point,
    cost_curve,
    downsample_curve,
    evaluate,
    ranking_metrics,
    review_rate_metrics,
)
from src.utils.mlflow_utils import (
    configure_mlflow,
    log_metrics_safely,
    log_model_compatibly,
    log_params_safely,
)
from src.utils.model_plots import (
    plot_cost_curve,
    plot_cv_stability,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_score_distribution,
)

COST_SETTINGS = {
    "review_cost": COST_REVIEW_PER_CASE,
    "chargeback_fee": COST_CHARGEBACK_FEE,
    "friction_cost": COST_FALSE_ALARM_FRICTION,
    "recovery_rate": FRAUD_RECOVERY_RATE,
}


def _as_date(seconds: float) -> str:
    reference = pd.Timestamp(REFERENCE_DATETIME)
    return (reference + pd.to_timedelta(int(seconds), unit="s")).date().isoformat()


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything except the carried-along columns is a feature."""
    excluded = {ID_COLUMN, TIME_COLUMN, TARGET_COLUMN, SPLIT_COLUMN}
    return [column for column in frame.columns if column not in excluded]


def _uid_features(features: list[str]) -> list[str]:
    """Find the uid family by rule, so the list cannot go stale."""
    return [
        name
        for name in features
        if any(marker in name for marker in UID_FEATURE_MARKERS)
    ]


def _score(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability of fraud, as a plain array."""
    return model.predict_proba(X)[:, 1]


# =========================================================
# Phase 2: train and compare the candidates
# =========================================================

def _train_candidates(
    candidates, X_train, y_train, X_valid, y_valid, amounts_valid, max_rounds
):
    rows = []
    score_sets = {}
    fitted = {}

    for candidate in candidates:
        print(f"\n  --- {candidate.name} ---")
        started = time.time()

        with mlflow.start_run(run_name=f"candidate_{candidate.name}"):
            mlflow.set_tag("phase", "candidate_comparison")
            mlflow.set_tag("model_family", candidate.name)
            log_params_safely({**candidate.params, "n_features": X_train.shape[1]})

            model = candidate.build(max_rounds)
            model, best_round = candidate.fit(
                model, X_train, y_train, X_valid, y_valid
            )

            scores = _score(model, X_valid)
            metrics = evaluate(
                y_valid,
                scores,
                amounts_valid,
                HEADLINE_REVIEW_RATES,
                COST_SETTINGS,
                REVIEW_CAPACITY_RATE,
            )
            elapsed = time.time() - started

            log_metrics_safely(metrics, prefix="valid_")
            mlflow.log_metric("fit_seconds", elapsed)
            if best_round is not None:
                mlflow.log_metric("best_round", best_round)

            signature = infer_signature(X_valid.head(50), scores[:50])
            log_model_compatibly(candidate.flavor, model, "model", signature=signature)

            print(
                f"    PR-AUC {metrics['pr_auc']:.5f}  "
                f"({metrics['pr_auc_lift']:.1f}x baseline)   "
                f"ROC-AUC {metrics['roc_auc']:.5f}   "
                f"{elapsed / 60:.1f} min"
            )

            rows.append(
                {
                    "model": candidate.name,
                    "best_round": best_round,
                    "fit_minutes": round(elapsed / 60, 2),
                    **metrics,
                }
            )
            score_sets[candidate.name] = scores
            fitted[candidate.name] = (candidate, model)

    return pd.DataFrame(rows), score_sets, fitted


# =========================================================
# Phase 4: the uid ablation
# =========================================================

def _run_uid_ablation(
    candidate, X_train, y_train, X_valid, y_valid, amounts_valid,
    max_rounds, baseline_pr_auc, uid_features,
):
    """
    Train the winner again without the uid features and compare.

    The decision rule was fixed before the result was seen (D-36): if
    removing them costs less than UID_ABLATION_TOLERANCE of PR-AUC, remove
    them, because they are blank on 82% of test rows.
    """
    print(f"\n  --- uid ablation: retraining {candidate.name} without "
          f"{len(uid_features)} uid features ---")

    kept = [column for column in X_train.columns if column not in set(uid_features)]

    with mlflow.start_run(run_name=f"ablation_no_uid_{candidate.name}"):
        mlflow.set_tag("phase", "uid_ablation")
        log_params_safely(
            {**candidate.params, "n_features": len(kept), "uid_removed": True}
        )

        model = candidate.build(max_rounds)
        model, best_round = candidate.fit(
            model, X_train[kept], y_train, X_valid[kept], y_valid
        )

        scores = _score(model, X_valid[kept])
        metrics = evaluate(
            y_valid, scores, amounts_valid, HEADLINE_REVIEW_RATES,
            COST_SETTINGS, REVIEW_CAPACITY_RATE,
        )
        log_metrics_safely(metrics, prefix="valid_")

    difference = baseline_pr_auc - metrics["pr_auc"]
    drop_uid = difference < UID_ABLATION_TOLERANCE

    print(f"    with uid   : PR-AUC {baseline_pr_auc:.5f}")
    print(f"    without uid: PR-AUC {metrics['pr_auc']:.5f}")
    print(f"    difference : {difference:+.5f}  "
          f"(pre-registered tolerance {UID_ABLATION_TOLERANCE})")
    print(f"    DECISION   : {'drop the uid features' if drop_uid else 'keep the uid features'}")

    return {
        "with_uid_pr_auc": baseline_pr_auc,
        "without_uid_pr_auc": metrics["pr_auc"],
        "difference": difference,
        "tolerance": UID_ABLATION_TOLERANCE,
        "drop_uid": bool(drop_uid),
        "uid_features": uid_features,
        "model": model if drop_uid else None,
        "kept_features": kept,
        "best_round": best_round,
        "metrics": metrics,
    }


# =========================================================
# Phase 5: time-aware cross-validation
# =========================================================

def _cross_validate(candidate, X, y, times, n_rounds, n_splits):
    """
    Expanding-window folds, with the round count fixed.

    Early stopping inside each fold would let each fold choose its own best
    stopping point using its own validation data, which makes every fold
    look slightly better than it is. Fixing the count first keeps this an
    honest stability check rather than another round of tuning. That is D-40.
    """
    print(f"\n  Cross-validating {candidate.name} over {n_splits} expanding windows ...")
    rows = []

    for fold, train_mask, valid_mask in expanding_window_splits(times, n_splits):
        model = rebuild_for_refit(candidate, n_rounds)
        model.fit(X[train_mask], y[train_mask])

        scores = _score(model, X[valid_mask])
        metrics = ranking_metrics(y[valid_mask], scores)

        rows.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "valid_start": _as_date(times[valid_mask].min()),
                "valid_end": _as_date(times[valid_mask].max()),
                **metrics,
            }
        )
        print(
            f"    fold {fold}: train {int(train_mask.sum()):>7,}  "
            f"valid {int(valid_mask.sum()):>7,}  "
            f"PR-AUC {metrics['pr_auc']:.5f}"
        )

    return pd.DataFrame(rows)


# =========================================================
# Phase 7: SHAP
# =========================================================

def _explain(model, X_valid, feature_names):
    """
    Explain the model with SHAP, on a sample.

    SHAP works out how much each feature pushed one prediction away from the
    average. Averaging those across many rows gives an importance ranking
    that reflects real influence on predictions, unlike the built-in
    importance of a tree model, which just counts how often a feature was
    used for a split.

    A sample is used because explaining all 118,108 validation rows would
    take far longer and change nothing about the answer.
    """
    try:
        import shap
    except ImportError:
        print("    shap not available, skipping")
        return None, None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_size = min(SHAP_SAMPLE_SIZE, len(X_valid))
    sample = X_valid.sample(sample_size, random_state=RANDOM_SEED)

    print(f"    computing SHAP values on {sample_size:,} rows ...")
    explainer = shap.TreeExplainer(model)
    values = explainer(sample)

    # Some libraries return one set of values per class. For a binary
    # problem we want the positive class.
    if values.values.ndim == 3:
        values = values[:, :, 1]

    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)

    shap.plots.beeswarm(values, max_display=25, show=False)
    plt.title("What drives the model, top 25 features")
    plt.savefig(EXPLAINABILITY_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("    saved shap_beeswarm.png")

    shap.plots.bar(values, max_display=25, show=False)
    plt.title("Average impact on the prediction")
    plt.savefig(EXPLAINABILITY_DIR / "shap_bar.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("    saved shap_bar.png")

    # One worked example: the row the model considered riskiest.
    riskiest = int(np.argmax(np.abs(values.values).sum(axis=1)))
    shap.plots.waterfall(values[riskiest], max_display=18, show=False)
    plt.title("One transaction explained")
    plt.savefig(EXPLAINABILITY_DIR / "shap_waterfall_example.png",
                bbox_inches="tight", dpi=130)
    plt.close()
    print("    saved shap_waterfall_example.png")

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return importance, sample_size


# =========================================================
# The stage
# =========================================================

def run_training(quick: bool = False, only_models: list[str] | None = None) -> dict:
    print("=" * 60)
    print("STAGE: MODEL TRAINING")
    print("=" * 60)

    ensure_directories()
    tracking_uri = configure_mlflow()
    print(f"  MLflow tracking: {tracking_uri}")

    max_rounds = QUICK_BOOSTING_ROUNDS if quick else MAX_BOOSTING_ROUNDS
    if quick:
        print(f"  QUICK MODE: boosting capped at {max_rounds} rounds. "
              "Results are for checking the code runs, not for reporting.")

    # --- phase 1: load and split ------------------------------------------
    print(f"\n  Loading {FEATURES_TRAIN_FILE.name} ...")
    data = pd.read_parquet(FEATURES_TRAIN_FILE)
    features = _feature_columns(data)
    print(f"    {len(data):,} rows, {len(features)} features")

    train_mask = (data[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL).to_numpy()
    valid_mask = (data[SPLIT_COLUMN] == VALID_SPLIT_LABEL).to_numpy()

    X_train = data.loc[train_mask, features]
    y_train = data.loc[train_mask, TARGET_COLUMN].to_numpy()
    X_valid = data.loc[valid_mask, features]
    y_valid = data.loc[valid_mask, TARGET_COLUMN].to_numpy()
    amounts_valid = data.loc[valid_mask, "TransactionAmt"].to_numpy()

    valid_times = data.loc[valid_mask, TIME_COLUMN].to_numpy()
    valid_days = (valid_times.max() - valid_times.min()) / 86400

    print(f"    train {len(X_train):,} rows, {int(y_train.sum()):,} frauds")
    print(f"    valid {len(X_valid):,} rows, {int(y_valid.sum()):,} frauds, "
          f"{valid_days:.0f} days")

    uid_features = _uid_features(features)
    print(f"    uid family: {len(uid_features)} features")

    # --- phase 2: candidates ------------------------------------------------
    print("\n  Training candidates ...")
    candidates = build_candidates(max_rounds, include=only_models)
    comparison, score_sets, fitted = _train_candidates(
        candidates, X_train, y_train, X_valid, y_valid, amounts_valid, max_rounds
    )

    comparison = comparison.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(MODEL_COMPARISON_FILE, index=False)
    print(f"\n  Wrote {MODEL_COMPARISON_FILE.name}")

    # --- phase 3: pick a winner -----------------------------------------------
    winner_name = comparison.loc[0, "model"]
    winner_candidate, winner_model = fitted[winner_name]
    winner_pr_auc = float(comparison.loc[0, "pr_auc"])
    winner_round = comparison.loc[0, "best_round"]
    print(f"\n  Winner: {winner_name}, validation PR-AUC {winner_pr_auc:.5f}")

    # --- phase 4: the uid ablation ---------------------------------------------
    ablation = None
    if uid_features and winner_candidate.supports_shap:
        ablation = _run_uid_ablation(
            winner_candidate, X_train, y_train, X_valid, y_valid, amounts_valid,
            max_rounds, winner_pr_auc, uid_features,
        )
        if ablation["drop_uid"]:
            features = ablation["kept_features"]
            winner_model = ablation["model"]
            winner_round = ablation["best_round"]
            X_train = X_train[features]
            X_valid = X_valid[features]
            score_sets[winner_name] = _score(winner_model, X_valid)
            print(f"    feature count now {len(features)}")

    winner_scores = score_sets[winner_name]

    # --- phase 5: cross-validation ------------------------------------------------
    n_rounds = int(winner_round) if winner_round else max_rounds
    all_X = data[features]
    all_y = data[TARGET_COLUMN].to_numpy()
    all_times = data[TIME_COLUMN].to_numpy()

    cv_results = _cross_validate(
        winner_candidate, all_X, all_y, all_times, n_rounds, CV_N_SPLITS
    )
    cv_results.to_csv(CV_RESULTS_FILE, index=False)
    print(f"    PR-AUC across folds: mean {cv_results['pr_auc'].mean():.5f}, "
          f"spread {cv_results['pr_auc'].std():.5f}")

    # --- phase 6: thresholds and cost ------------------------------------------------
    print("\n  Threshold and cost analysis ...")
    curve = cost_curve(y_valid, winner_scores, amounts_valid, **COST_SETTINGS)
    unconstrained = best_operating_point(curve, capacity_rate=None)
    constrained = best_operating_point(curve, capacity_rate=REVIEW_CAPACITY_RATE)

    downsample_curve(curve).to_csv(COST_CURVE_FILE, index=False)

    threshold_rows = []
    for rate in HEADLINE_REVIEW_RATES:
        point = review_rate_metrics(y_valid, winner_scores, rate)
        at_rate = curve.iloc[point["n_reviewed"]]
        threshold_rows.append(
            {
                **point,
                "total_cost": float(at_rate["total_cost"]),
                "savings": float(at_rate["savings"]),
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(THRESHOLD_ANALYSIS_FILE, index=False)

    baseline_cost = float(curve.loc[0, "total_cost"])
    annual_factor = 365.0 / max(valid_days, 1.0)

    print(f"    doing nothing costs        : ${baseline_cost:,.0f} over {valid_days:.0f} days")
    print(f"    cheapest overall           : {unconstrained['review_rate']:.2%} reviewed, "
          f"saves ${unconstrained['savings']:,.0f}")
    print(f"    cheapest within {REVIEW_CAPACITY_RATE:.0%} capacity: "
          f"{constrained['review_rate']:.2%} reviewed, saves ${constrained['savings']:,.0f}")
    print(f"    annualised saving          : ${constrained['savings'] * annual_factor:,.0f}")

    # --- phase 7: SHAP -------------------------------------------------------------------
    print("\n  Explaining the model ...")
    importance, shap_rows = (None, None)
    if winner_candidate.supports_shap:
        importance, shap_rows = _explain(winner_model, X_valid, features)
        if importance is not None:
            importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
            print(f"    Wrote {FEATURE_IMPORTANCE_FILE.name}")
            print("    top 10 features:")
            for _, row in importance.head(10).iterrows():
                print(f"      {row['feature']:<45} {row['mean_abs_shap']:.5f}")

    # --- charts ------------------------------------------------------------------------
    print("\n  Generating charts ...")
    plot_model_comparison(comparison, FIGURES_DIR)
    plot_precision_recall_curves(y_valid, score_sets, FIGURES_DIR)
    plot_cost_curve(curve, unconstrained, constrained, REVIEW_CAPACITY_RATE, FIGURES_DIR)
    plot_score_distribution(y_valid, winner_scores, FIGURES_DIR)
    plot_cv_stability(cv_results, FIGURES_DIR)

    # --- phase 8: final model, registry, submission ---------------------------------------
    # Retrain on every labelled row. Validation chose the settings; the model
    # that ships should still see all the data. The round count is scaled by
    # how much more data it now sees, which is the standard adjustment. D-41.
    scale = len(data) / len(X_train)
    final_rounds = max(1, int(round(n_rounds * scale)))
    print(f"\n  Retraining {winner_name} on all {len(data):,} labelled rows "
          f"({n_rounds} rounds scaled by {scale:.2f} to {final_rounds}) ...")

    with mlflow.start_run(run_name=f"final_{winner_name}") as final_run:
        mlflow.set_tag("phase", "final")
        mlflow.set_tag("model_family", winner_name)
        log_params_safely(
            {
                **winner_candidate.params,
                "n_estimators": final_rounds,
                "n_features": len(features),
                "uid_features_dropped": bool(ablation and ablation["drop_uid"]),
                "trained_on_rows": len(data),
            }
        )

        final_model = rebuild_for_refit(winner_candidate, final_rounds)
        final_model.fit(all_X, all_y)

        # These are the validation numbers from the model selection step, not
        # a score for the final model. The final model has no clean holdout
        # left, which is exactly why we validated before retraining.
        log_metrics_safely(
            {
                "selection_pr_auc": winner_pr_auc,
                "cv_pr_auc_mean": float(cv_results["pr_auc"].mean()),
                "cv_pr_auc_std": float(cv_results["pr_auc"].std()),
                "chosen_threshold": constrained["threshold"],
                "savings_within_capacity": constrained["savings"],
                "annualised_savings": constrained["savings"] * annual_factor,
            }
        )

        signature = infer_signature(all_X.head(50), _score(final_model, all_X.head(50)))
        model_info = log_model_compatibly(
            winner_candidate.flavor, final_model, "model", signature=signature
        )

        for path in (
            MODEL_COMPARISON_FILE, THRESHOLD_ANALYSIS_FILE,
            CV_RESULTS_FILE, COST_CURVE_FILE,
        ):
            if path.exists():
                mlflow.log_artifact(str(path))

        final_run_id = final_run.info.run_id

    joblib.dump(final_model, FINAL_MODEL_FILE)
    print(f"  Saved {FINAL_MODEL_FILE.name} "
          f"({FINAL_MODEL_FILE.stat().st_size / 1024 ** 2:.1f} MB)")

    # Register it and point the candidate alias at this version.
    registered_version = None
    try:
        registered = mlflow.register_model(model_info.model_uri, REGISTERED_MODEL_NAME)
        registered_version = registered.version
        mlflow.MlflowClient().set_registered_model_alias(
            REGISTERED_MODEL_NAME, MODEL_ALIAS_CANDIDATE, registered_version
        )
        print(f"  Registered as {REGISTERED_MODEL_NAME} version "
              f"{registered_version}, alias '{MODEL_ALIAS_CANDIDATE}'")
    except Exception as error:  # noqa: BLE001
        print(f"  Registry step failed: {error}")
        print("  The model file and the MLflow run are still saved.")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": winner_name,
        "mlflow_run_id": final_run_id,
        "registered_version": registered_version,
        "n_features": len(features),
        "feature_names": features,
        "uid_features_dropped": bool(ablation and ablation["drop_uid"]),
        "n_estimators": final_rounds,
        "selection_pr_auc": winner_pr_auc,
        "cv_pr_auc_mean": float(cv_results["pr_auc"].mean()),
        "chosen_threshold": constrained["threshold"],
        "chosen_review_rate": constrained["review_rate"],
        "cost_assumptions": COST_SETTINGS,
        "review_capacity_rate": REVIEW_CAPACITY_RATE,
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Wrote {MODEL_METADATA_FILE.name}")

    # --- Kaggle submission ---------------------------------------------------------
    print(f"\n  Scoring the test set ...")
    test = pd.read_parquet(FEATURES_TEST_FILE)
    test_scores = _score(final_model, test[features])
    pd.DataFrame(
        {ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: test_scores}
    ).to_csv(KAGGLE_SUBMISSION_FILE, index=False)
    print(f"    Wrote {KAGGLE_SUBMISSION_FILE.name} ({len(test):,} rows)")

    results = {
        "comparison": comparison,
        "winner": winner_name,
        "winner_pr_auc": winner_pr_auc,
        "ablation": ablation,
        "cv_results": cv_results,
        "unconstrained": unconstrained,
        "constrained": constrained,
        "threshold_table": threshold_table,
        "baseline_cost": baseline_cost,
        "annual_factor": annual_factor,
        "valid_days": valid_days,
        "importance": importance,
        "n_features": len(features),
        "final_rounds": final_rounds,
        "registered_version": registered_version,
        "final_run_id": final_run_id,
    }
    _write_summary(results)

    print("\n" + "=" * 60)
    print("TRAINING HEADLINES")
    print("=" * 60)
    print(f"  Winner                : {winner_name}")
    print(f"  Validation PR-AUC     : {winner_pr_auc:.5f} "
          f"({winner_pr_auc / 0.0349:.1f}x baseline)")
    print(f"  CV PR-AUC             : {cv_results['pr_auc'].mean():.5f} "
          f"+/- {cv_results['pr_auc'].std():.5f}")
    print(f"  Features used         : {len(features)}")
    print(f"  Chosen threshold      : {constrained['threshold']:.4f} "
          f"at {constrained['review_rate']:.2%} review rate")
    print(f"  Recall at that point  : {constrained['recall']:.1%}")
    print(f"  Annualised saving     : ${constrained['savings'] * annual_factor:,.0f}")
    print(f"\n  Full report: {TRAINING_SUMMARY_FILE}")

    return results


def _write_summary(results: dict) -> None:
    """Write the human-readable training summary."""
    lines: list[str] = []
    add = lines.append

    constrained = results["constrained"]
    unconstrained = results["unconstrained"]
    cv = results["cv_results"]

    add("# Model Training Summary")
    add("")
    add("Generated automatically by `src/pipelines/training.py`. "
        "Do not edit by hand, it is overwritten on every run.")
    add("")

    add("## 1. Candidate comparison")
    add("")
    display = results["comparison"][
        ["model", "pr_auc", "pr_auc_lift", "roc_auc", "best_round", "fit_minutes"]
    ].round(5)
    add(display.to_markdown(index=False))
    add("")
    add(f"Winner: **{results['winner']}**, validation PR-AUC "
        f"**{results['winner_pr_auc']:.5f}**.")
    add("")

    if results["ablation"]:
        ablation = results["ablation"]
        add("## 2. The uid ablation")
        add("")
        add(f"Six uid features are blank on about 82% of test rows, so the "
            f"winner was retrained without them. The decision rule was fixed "
            f"in advance: drop them if the cost is under "
            f"{ablation['tolerance']} PR-AUC.")
        add("")
        add("| Model | Validation PR-AUC |")
        add("|-------|-------------------|")
        add(f"| with uid features | {ablation['with_uid_pr_auc']:.5f} |")
        add(f"| without uid features | {ablation['without_uid_pr_auc']:.5f} |")
        add(f"| difference | {ablation['difference']:+.5f} |")
        add("")
        add(f"**Decision: {'dropped' if ablation['drop_uid'] else 'kept'}.** "
            f"Final feature count {results['n_features']}.")
        add("")

    add("## 3. Stability across time")
    add("")
    add(cv[["fold", "train_rows", "valid_rows", "valid_start",
            "valid_end", "pr_auc", "roc_auc"]].round(5).to_markdown(index=False))
    add("")
    add(f"Mean PR-AUC **{cv['pr_auc'].mean():.5f}**, "
        f"spread **{cv['pr_auc'].std():.5f}**. Each fold trains on more "
        "history than the last and is scored on the period straight after, "
        "which is the same shape as the real problem.")
    add("")

    add("## 4. What it is worth")
    add("")
    add("Costs use the assumptions in `config/config.py`. They are stated "
        "assumptions, not figures supplied by a business. See step4.md "
        "section 3.")
    add("")
    add("| Assumption | Value |")
    add("|------------|-------|")
    add(f"| Analyst review | ${COST_REVIEW_PER_CASE:.2f} per case |")
    add(f"| Chargeback fee | ${COST_CHARGEBACK_FEE:.2f} per missed fraud |")
    add(f"| False alarm friction | ${COST_FALSE_ALARM_FRICTION:.2f} |")
    add(f"| Fraud recovered when caught | {FRAUD_RECOVERY_RATE:.0%} |")
    add(f"| Review capacity | {REVIEW_CAPACITY_RATE:.0%} of transactions |")
    add("")
    add(f"Over the {results['valid_days']:.0f} day validation period, doing "
        f"nothing costs **${results['baseline_cost']:,.0f}** in fraud losses "
        "and chargeback fees.")
    add("")
    add("| Operating point | Review rate | Recall | Savings |")
    add("|-----------------|-------------|--------|---------|")
    add(f"| Cheapest overall | {unconstrained['review_rate']:.2%} | "
        f"{unconstrained['recall']:.1%} | ${unconstrained['savings']:,.0f} |")
    add(f"| Cheapest within capacity | {constrained['review_rate']:.2%} | "
        f"{constrained['recall']:.1%} | ${constrained['savings']:,.0f} |")
    add("")
    add(f"**Annualised, at the within-capacity operating point: "
        f"${constrained['savings'] * results['annual_factor']:,.0f} a year.**")
    add("")
    add(f"The chosen threshold is **{constrained['threshold']:.4f}**.")
    add("")
    add("Recall and cost at each headline review rate:")
    add("")
    add(results["threshold_table"][
        ["review_rate", "n_reviewed", "threshold", "recall", "precision", "savings"]
    ].round(5).to_markdown(index=False))
    add("")

    if results["importance"] is not None:
        add("## 5. What drives the model")
        add("")
        add(results["importance"].head(20).round(5).to_markdown(index=False))
        add("")
        add("Charts in `reports/explainability/`.")
        add("")

    add("## 6. Carried into Step 5")
    add("")
    add(f"1. Registered model `{REGISTERED_MODEL_NAME}` version "
        f"{results['registered_version']}, alias `{MODEL_ALIAS_CANDIDATE}`.")
    add(f"2. MLflow run id `{results['final_run_id']}`.")
    add(f"3. Operating threshold {constrained['threshold']:.4f}, chosen by "
        "cost within review capacity, not left at 0.5.")
    add("4. Watch the uid family in drift monitoring, whether or not it was "
        "dropped. It was the clearest train-to-test shift in the data.")
    add("5. `models/final_model_metadata.json` holds the exact feature list "
        "the service must supply.")
    add("")

    TRAINING_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {TRAINING_SUMMARY_FILE.name}")
