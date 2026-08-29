"""
Monitoring stage.

Answers two questions that production has to answer without labels.

  1. Is the model still working?
     Only measurable on labelled data, so we score the held-out validation
     period week by week using a model that never saw it.

  2. Has the data changed?
     Measurable immediately, so we compare every month of the unlabelled
     test period against the training distribution.

Input:  data/processed/train_features.parquet
        data/processed/test_features.parquet
        models/selection_model.joblib   (built here if missing)
        reports/feature_importance.csv
Output: reports/monitoring/*
        reports/figures/16 to 19

Run with:
    python run.py --step monitoring
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from config.config import (
    ALERT_RATE_TOLERANCE,
    DASHBOARD_DATA_FILE,
    DRIFT_LOW_CONFIDENCE_ROWS,
    DRIFT_SUMMARY_FILE,
    DRIFT_TOP_FEATURES,
    FEATURE_DRIFT_FILE,
    FEATURE_IMPORTANCE_FILE,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    FIGURES_DIR,
    MODEL_METADATA_FILE,
    PERIOD_METRICS_FILE,
    PSI_SIGNIFICANT,
    REFERENCE_DATETIME,
    RETRAIN_WEIGHTED_PSI,
    SCORE_DRIFT_FILE,
    SELECTION_MODEL_FILE,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_SPLIT_LABEL,
    VALID_SPLIT_LABEL,
    WATCH_WEIGHTED_PSI,
    ensure_directories,
)
from src.monitoring.drift import compare_features, weighted_drift_score
from src.utils.metrics import ranking_metrics
from src.utils.monitoring_plots import (
    plot_alert_rate,
    plot_feature_drift,
    plot_performance_over_time,
    plot_score_drift,
)


def _timestamps(seconds: np.ndarray) -> pd.Series:
    """Turn the seconds counter into readable dates, for grouping only."""
    reference = pd.Timestamp(REFERENCE_DATETIME)
    return reference + pd.to_timedelta(pd.Series(seconds), unit="s")


def _load_metadata() -> dict:
    if not MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(
            f"{MODEL_METADATA_FILE} not found.\n"
            f"Run  python run.py --step training  first."
        )
    return json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))


def _get_selection_model(train_frame: pd.DataFrame, metadata: dict):
    """
    Load the model trained on the training portion only, or build one.

    The final model has seen every labelled row, including the validation
    period, so scoring that period with it would be meaningless. We need one
    that genuinely never saw those weeks.

    A saved selection model records which training run produced it. If that
    does not match the current metadata, the file is stale and is rebuilt.
    Without this check, monitoring silently reports on whichever model
    happened to be current the last time it ran, which is how the Step 5 run
    ended up describing CatBoost while LightGBM was in production.
    """
    fingerprint = f"{metadata.get('model_family')}::{metadata.get('mlflow_run_id')}"

    if SELECTION_MODEL_FILE.exists():
        try:
            saved = joblib.load(SELECTION_MODEL_FILE)
        except Exception:  # noqa: BLE001
            saved = None

        if isinstance(saved, dict) and saved.get("fingerprint") == fingerprint:
            print(
                f"  Loading {SELECTION_MODEL_FILE.name} "
                f"({metadata.get('model_family')}) ..."
            )
            return saved["model"]

        print(f"  {SELECTION_MODEL_FILE.name} was built for a different run.")
        print("  Rebuilding it so monitoring describes the current model.")

    print("  Training a selection model on the train portion ...")

    from src.models.candidates import build_candidates, rebuild_for_refit

    family = metadata.get("model_family", "lightgbm")
    rounds = max(50, int(metadata.get("n_estimators", 600) / 1.25))

    candidates = build_candidates(rounds, include=[family])
    if not candidates:
        raise ValueError(f"cannot rebuild model family '{family}'")

    features = metadata["feature_names"]
    train_rows = train_frame[train_frame[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL]

    model = rebuild_for_refit(candidates[0], rounds)
    model.fit(train_rows[features], train_rows[TARGET_COLUMN].to_numpy())

    # Save the model together with the fingerprint of the run it belongs to.
    joblib.dump({"fingerprint": fingerprint, "model": model}, SELECTION_MODEL_FILE)
    print(f"  Saved {SELECTION_MODEL_FILE.name}")
    return model


def _weekly_performance(
    valid_frame: pd.DataFrame, model, features: list[str]
) -> pd.DataFrame:
    """
    PR-AUC week by week on the held-out validation period.

    The only honest performance number available, because this is the last
    labelled data the model did not train on.
    """
    scores = model.predict_proba(valid_frame[features])[:, 1]
    labels = valid_frame[TARGET_COLUMN].to_numpy()
    weeks = _timestamps(valid_frame[TIME_COLUMN].to_numpy()).dt.to_period("W")

    records = []
    for week, index in pd.Series(range(len(weeks))).groupby(weeks.to_numpy()):
        positions = index.to_numpy()
        week_labels = labels[positions]

        # A week with no fraud at all cannot produce a PR-AUC.
        if week_labels.sum() < 10 or len(positions) < 500:
            continue

        metrics = ranking_metrics(week_labels, scores[positions])
        fraud_rate = float(week_labels.mean())

        records.append(
            {
                "period": str(week),
                "rows": len(positions),
                "frauds": int(week_labels.sum()),
                "fraud_rate": fraud_rate,
                "pr_auc": metrics["pr_auc"],
                # PR-AUC's floor is the fraud rate of the period being measured,
                # and that rate moves week to week. Comparing raw PR-AUC across
                # weeks compares numbers standing on different floors. Lift
                # divides it out, so the weeks become comparable.
                "pr_auc_lift": (
                    metrics["pr_auc"] / fraud_rate if fraud_rate else float("nan")
                ),
                "roc_auc": metrics["roc_auc"],
                # Marked so short, partial weeks at the edges of the period can
                # be discounted rather than read as a trend.
                "is_full_week": len(positions) >= 15_000,
            }
        )

    return pd.DataFrame(records)


def run_monitoring() -> dict:
    print("=" * 60)
    print("STAGE: MONITORING")
    print("=" * 60)

    ensure_directories()
    metadata = _load_metadata()
    features = metadata["feature_names"]
    threshold = float(metadata["chosen_threshold"])
    expected_rate = float(metadata["chosen_review_rate"])

    print(f"  Model: {metadata['model_family']}, {len(features)} features")
    print(
        f"  Operating threshold: {threshold:.4f} "
        f"(expected alert rate {expected_rate:.2%})"
    )

    # --- load ----------------------------------------------------------
    print(f"\n  Loading {FEATURES_TRAIN_FILE.name} ...")
    train_frame = pd.read_parquet(FEATURES_TRAIN_FILE)
    reference = train_frame[train_frame[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL]
    valid_frame = train_frame[train_frame[SPLIT_COLUMN] == VALID_SPLIT_LABEL]
    print(f"    reference (train portion): {len(reference):,} rows")

    model = _get_selection_model(train_frame, metadata)

    # --- 1. performance on labelled held-out data -----------------------
    print("\n  Measuring performance week by week on held-out labelled data ...")
    period_metrics = _weekly_performance(valid_frame, model, features)
    period_metrics.to_csv(PERIOD_METRICS_FILE, index=False)
    for _, row in period_metrics.iterrows():
        print(
            f"    {row['period']}  rows {int(row['rows']):>6,}  "
            f"frauds {int(row['frauds']):>4,}  PR-AUC {row['pr_auc']:.4f}"
        )

    # --- 2. drift on the unlabelled test period --------------------------
    print(f"\n  Loading {FEATURES_TEST_FILE.name} ...")
    test_frame = pd.read_parquet(FEATURES_TEST_FILE)
    test_months = _timestamps(test_frame[TIME_COLUMN].to_numpy()).dt.to_period("M")
    print(f"    {len(test_frame):,} rows across {test_months.nunique()} months")

    importance = (
        pd.read_csv(FEATURE_IMPORTANCE_FILE)
        if FEATURE_IMPORTANCE_FILE.exists()
        else pd.DataFrame({"feature": features, "mean_abs_shap": 1.0})
    )
    top_features = importance.nlargest(DRIFT_TOP_FEATURES, "mean_abs_shap")[
        "feature"
    ].tolist()

    print("\n  Comparing each month against the training distribution ...")
    drift_frames = []
    score_records = []

    for month in sorted(test_months.unique()):
        month_rows = test_frame[(test_months == month).to_numpy()]
        if len(month_rows) < 1000:
            continue

        label = str(month)
        drift = compare_features(reference, month_rows, features, label)
        drift_frames.append(drift)

        scores = model.predict_proba(month_rows[features])[:, 1]
        alert_rate = float((scores >= threshold).mean())
        weighted = weighted_drift_score(drift, importance)

        significant = int((drift["psi"] > PSI_SIGNIFICANT).sum())
        significant_top = int(
            (drift[drift["feature"].isin(top_features)]["psi"] > PSI_SIGNIFICANT).sum()
        )

        score_records.append(
            {
                "period": label,
                "rows": len(month_rows),
                "score_mean": float(scores.mean()),
                "score_p50": float(np.percentile(scores, 50)),
                "score_p90": float(np.percentile(scores, 90)),
                "score_p99": float(np.percentile(scores, 99)),
                "alert_rate": alert_rate,
                "alert_rate_ratio": (
                    alert_rate / expected_rate if expected_rate else float("nan")
                ),
                "weighted_psi": weighted,
                "features_significant": significant,
                "top_features_significant": significant_top,
            }
        )

        print(
            f"    {label}  rows {len(month_rows):>7,}  "
            f"alerts {alert_rate:>6.2%}  "
            f"weighted PSI {weighted:>6.3f}  "
            f"drifted {significant:>3}/{len(features)} "
            f"(top20: {significant_top})"
        )

    feature_drift = pd.concat(drift_frames, ignore_index=True)
    feature_drift.to_csv(FEATURE_DRIFT_FILE, index=False)

    score_drift = pd.DataFrame(score_records)
    score_drift.to_csv(SCORE_DRIFT_FILE, index=False)

    # --- 3. the verdict ---------------------------------------------------
    latest = score_drift.iloc[-1]
    worst_weighted = float(score_drift["weighted_psi"].max())
    alert_ratio = float(latest["alert_rate_ratio"])
    alert_off = abs(alert_ratio - 1.0) > ALERT_RATE_TOLERANCE

    if (
        worst_weighted > RETRAIN_WEIGHTED_PSI
        or int(latest["top_features_significant"]) >= 3
    ):
        verdict = "RETRAIN"
    elif worst_weighted > WATCH_WEIGHTED_PSI or alert_off:
        verdict = "WATCH"
    else:
        verdict = "OK"

    print(f"\n  Verdict: {verdict}")
    print(
        f"    worst weighted PSI       : {worst_weighted:.4f} "
        f"(retrain above {RETRAIN_WEIGHTED_PSI})"
    )
    print(
        f"    latest alert rate        : {latest['alert_rate']:.2%} "
        f"against an expected {expected_rate:.2%}"
    )
    print(f"    top-20 features drifted  : {int(latest['top_features_significant'])}")

    # --- 4. charts ----------------------------------------------------------
    print("\n  Generating charts ...")
    if not period_metrics.empty:
        plot_performance_over_time(period_metrics, FIGURES_DIR)
    plot_feature_drift(feature_drift, top_features, FIGURES_DIR)
    plot_score_drift(score_drift, FIGURES_DIR)
    plot_alert_rate(score_drift, expected_rate, FIGURES_DIR)

    # --- 5. the dashboard file ------------------------------------------------
    # Small on purpose. Per D-45 the Step 7 dashboard must load in under three
    # seconds, so everything it shows has to be precomputed into a file this
    # size rather than derived from the feature tables at page load.
    dashboard = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "model_family": metadata["model_family"],
        "registered_version": metadata.get("registered_version"),
        "n_features": len(features),
        "threshold": threshold,
        "expected_alert_rate": expected_rate,
        "selection_pr_auc": metadata.get("selection_pr_auc"),
        "cv_pr_auc_mean": metadata.get("cv_pr_auc_mean"),
        "verdict": verdict,
        "worst_weighted_psi": worst_weighted,
        "weekly_performance": period_metrics.to_dict(orient="records"),
        "monthly_drift": score_drift.to_dict(orient="records"),
        "top_drifted_features": feature_drift.nlargest(15, "psi")[
            ["period", "feature", "psi", "missing_reference", "missing_current"]
        ].to_dict(orient="records"),
    }
    DASHBOARD_DATA_FILE.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(
        f"  Wrote {DASHBOARD_DATA_FILE.name} "
        f"({DASHBOARD_DATA_FILE.stat().st_size / 1024:.0f} KB)"
    )

    results = {
        "period_metrics": period_metrics,
        "feature_drift": feature_drift,
        "score_drift": score_drift,
        "top_features": top_features,
        "verdict": verdict,
        "worst_weighted_psi": worst_weighted,
        "expected_rate": expected_rate,
        "threshold": threshold,
        "metadata": metadata,
    }
    _write_summary(results)

    print("\n" + "=" * 60)
    print("MONITORING HEADLINES")
    print("=" * 60)
    print(f"  Verdict              : {verdict}")
    print(f"  Worst weighted PSI   : {worst_weighted:.4f}")
    print(f"  Months monitored     : {len(score_drift)}")
    print(f"  Weeks measured       : {len(period_metrics)}")
    print(f"\n  Full report: {DRIFT_SUMMARY_FILE}")

    return results


def _write_summary(results: dict) -> None:
    lines: list[str] = []
    add = lines.append

    score_drift = results["score_drift"]
    feature_drift = results["feature_drift"]
    period_metrics = results["period_metrics"]

    add("# Monitoring Summary")
    add("")
    add(
        "Generated automatically by `src/pipelines/monitoring.py`. "
        "Do not edit by hand, it is overwritten on every run."
    )
    add("")

    add(f"## Verdict: {results['verdict']}")
    add("")
    add(
        f"- Worst importance-weighted PSI across all periods: "
        f"**{results['worst_weighted_psi']:.4f}**"
    )
    add(f"- Retrain threshold: {RETRAIN_WEIGHTED_PSI}")
    add(
        f"- Operating threshold: {results['threshold']:.4f}, expected alert rate "
        f"{results['expected_rate']:.2%}"
    )
    add("")

    add("## 1. Performance on labelled held-out data")
    add("")
    if period_metrics.empty:
        add("Not enough labelled rows per week to measure.")
    else:
        add(period_metrics.round(4).to_markdown(index=False))
        add("")
        add(
            "`pr_auc_lift` is the PR-AUC divided by that week's own fraud rate. "
            "It is the column to read for a trend. Raw PR-AUC sits on a floor "
            "equal to the fraud rate, and that rate moves week to week, so raw "
            "scores from different weeks are not directly comparable."
        )
        add("")
        add(
            "These weeks are the last labelled data the model never trained on. "
            "This is the only honest performance measurement available, because "
            "the test period has no labels at all. In production you would be "
            "in the same position: weeks or months of scoring before you learn "
            "whether the scores were any good."
        )
    add("")

    add("## 2. Data drift, month by month")
    add("")
    add(
        score_drift[
            [
                "period",
                "rows",
                "alert_rate",
                "weighted_psi",
                "features_significant",
                "top_features_significant",
            ]
        ]
        .round(4)
        .to_markdown(index=False)
    )
    add("")
    add(
        "`weighted_psi` weights each feature's drift by how much the model "
        "actually relies on it. With 284 features a few will always have "
        "drifted, and drift in a feature the model ignores is not a problem."
    )
    add("")

    add("## 3. The features that moved most")
    add("")
    confident = feature_drift[~feature_drift["low_confidence"]]
    add(
        confident.nlargest(20, "psi")[
            [
                "period",
                "feature",
                "psi",
                "band",
                "rows_current",
                "missing_reference",
                "missing_current",
            ]
        ]
        .round(4)
        .to_markdown(index=False)
    )
    add("")
    low = feature_drift[feature_drift["low_confidence"]]
    add(
        f"{len(low)} feature-period combinations were measured on fewer than "
        f"{DRIFT_LOW_CONFIDENCE_ROWS:,} usable values and are excluded from this "
        "table. A PSI computed on a few hundred rows swings wildly for reasons "
        "that have nothing to do with drift. They are still in "
        "`feature_drift.csv`, flagged in the `low_confidence` column."
    )

    add("## 4. Alert volume")
    add("")
    add(
        "The threshold is fixed, so any change in alert volume comes entirely "
        "from the data moving. This is the number an operations manager feels "
        "directly, because it is how much work lands in the review queue."
    )
    add("")
    add(
        score_drift[["period", "alert_rate", "alert_rate_ratio"]]
        .round(4)
        .to_markdown(index=False)
    )
    add("")

    add("## 5. What happens next")
    add("")
    if results["verdict"] == "RETRAIN":
        add(
            "The drift is large enough to act on. Retrain on data that includes "
            "the recent period, then run the promotion gates before the new "
            "model is allowed to serve:"
        )
        add("")
        add("```powershell")
        add("python run.py --step features")
        add("python run.py --step training")
        add("python scripts/promote_model.py --version <new version> --dry-run")
        add("```")
    elif results["verdict"] == "WATCH":
        add(
            "Drift is present but below the retraining threshold. Keep "
            "monitoring, and look at whether the trend is worsening month "
            "on month or holding steady."
        )
    else:
        add(
            "No action needed. The data still resembles what the model was "
            "trained on."
        )
    add("")

    DRIFT_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {DRIFT_SUMMARY_FILE.name}")
