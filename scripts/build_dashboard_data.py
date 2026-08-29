"""
Collect every number the dashboard needs into one small JSON file.

Why a bundle rather than reading the reports directly: the dashboard runs on
a different machine, from a different folder, possibly without the dataset,
and it must open in under three seconds. Reading eight CSV files and a
Parquet table on every page load would fail all three of those. Decision
D-45 and D-57.

The output is a few hundred kilobytes, committed to the repository, so the
dashboard needs nothing except itself.

Usage:
    python scripts/build_dashboard_data.py
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.config import (  # noqa: E402
    COST_CHARGEBACK_FEE,
    COST_FALSE_ALARM_FRICTION,
    COST_REVIEW_PER_CASE,
    CV_RESULTS_FILE,
    FEATURE_DRIFT_FILE,
    FEATURE_IMPORTANCE_FILE,
    FRAUD_RECOVERY_RATE,
    MODEL_COMPARISON_FILE,
    MODEL_METADATA_FILE,
    PERIOD_METRICS_FILE,
    PROJECT_ROOT,
    PSI_SIGNIFICANT,
    RETRAIN_WEIGHTED_PSI,
    REVIEW_CAPACITY_RATE,
    SCORE_DRIFT_FILE,
    THRESHOLD_ANALYSIS_FILE,
)

OUTPUT_FILE = PROJECT_ROOT / "app" / "dashboard_data.json"

# The live API. Used once at build time to capture a real response, so the
# dashboard has something to show when the free-tier service is asleep.
API_URL = "https://ieee-cis-fraud-detection.onrender.com"

EXAMPLE_TRANSACTION = {
    "TransactionID": 3663549,
    "TransactionDT": 18403224,
    "TransactionAmt": 31.95,
    "ProductCD": "W",
    "card1": 10409,
    "card2": 111,
    "card4": "visa",
    "card6": "debit",
    "addr1": 325,
    "D1": 14,
    "D15": 0,
    "C1": 1,
    "C13": 1,
    "C14": 1,
    "P_emaildomain": "gmail.com",
    "DeviceType": "desktop",
}


def read_csv(path: Path, columns: list[str] | None = None) -> list[dict]:
    """Read a report file into plain records, tolerating a missing file."""
    if not path.exists():
        print(f"  skipping {path.name}, not found")
        return []

    frame = pd.read_csv(path)
    if columns:
        frame = frame[[column for column in columns if column in frame.columns]]

    # NaN is not valid JSON. Turning it into None keeps the file loadable.
    return json.loads(frame.to_json(orient="records"))


def capture_example_response() -> dict:
    """
    Call the live API once and keep the answer.

    The dashboard shows this when the free-tier service is asleep, so the
    scorer section always has something in it rather than an error.
    """
    try:
        import requests

        print(f"  calling {API_URL}/predict for a cached example ...")
        response = requests.post(
            f"{API_URL}/predict",
            json={"transaction": EXAMPLE_TRANSACTION, "explain": True},
            timeout=120,
        )
        response.raise_for_status()
        print("  got a live response")
        return response.json()
    except Exception as error:  # noqa: BLE001
        print(f"  could not reach the API ({error}), using a stored example")
        return {
            "transaction_id": 3663549,
            "fraud_probability": 0.015623875265457728,
            "threshold": 0.4222493056998478,
            "decision": "pass",
            "model_version": "4",
            "explanation": [
                {"feature": "C13", "value": 1.0, "contribution": 0.2821886320282231},
                {
                    "feature": "D15_mean_by_card1",
                    "value": 240.918,
                    "contribution": -0.2188581338218559,
                },
                {"feature": "V54", "value": None, "contribution": 0.1960968019708084},
                {
                    "feature": "card1_addr1_code",
                    "value": -1.0,
                    "contribution": 0.1878922435337035,
                },
            ],
        }


def main() -> None:
    print("Building the dashboard bundle ...")

    metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))

    comparison = read_csv(
        MODEL_COMPARISON_FILE,
        [
            "model",
            "pr_auc",
            "pr_auc_lift",
            "roc_auc",
            "best_savings_within_capacity",
            "best_round",
            "fit_minutes",
        ],
    )
    thresholds = read_csv(
        THRESHOLD_ANALYSIS_FILE,
        ["review_rate", "n_reviewed", "threshold", "recall", "precision", "savings"],
    )
    cv = read_csv(CV_RESULTS_FILE, ["fold", "valid_start", "valid_end", "pr_auc"])
    weekly = read_csv(PERIOD_METRICS_FILE)
    monthly = read_csv(SCORE_DRIFT_FILE)

    importance = read_csv(FEATURE_IMPORTANCE_FILE)[:20]

    # Only the worst drift, and only rows measured on enough data to trust.
    drift_records = []
    if FEATURE_DRIFT_FILE.exists():
        drift = pd.read_csv(FEATURE_DRIFT_FILE)
        if "low_confidence" in drift.columns:
            drift = drift[~drift["low_confidence"]]
        drift = drift.nlargest(25, "psi")[
            ["period", "feature", "psi", "missing_reference", "missing_current"]
        ]
        drift_records = json.loads(drift.to_json(orient="records"))

    bundle = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "api_url": API_URL,
        "repo_url": "https://github.com/Dee-ui/ieee-cis-fraud-detection",
        "model_hub_url": "https://huggingface.co/Dee-ui/ieee-cis-fraud-detector",
        "model": {
            "family": metadata["model_family"],
            "version": metadata.get("registered_version"),
            "n_features": len(metadata["feature_names"]),
            "threshold": metadata["chosen_threshold"],
            "review_rate": metadata.get("chosen_review_rate", REVIEW_CAPACITY_RATE),
            "pr_auc": metadata.get("selection_pr_auc"),
            "cv_pr_auc_mean": metadata.get("cv_pr_auc_mean"),
            "cv_pr_auc_std": metadata.get("cv_pr_auc_std"),
            "trained_on_rows": metadata.get("trained_on_rows"),
            "savings_window": metadata.get("savings_within_capacity"),
            "savings_annual": metadata.get("annualised_savings"),
        },
        # Fixed facts about the data, so the dashboard never has to open it.
        "dataset": {
            "rows": 590540,
            "frauds": 20663,
            "fraud_rate": 0.034990,
            "features_raw": 435,
            "features_final": len(metadata["feature_names"]),
            "train_start": "2017-12-01",
            "train_end": "2018-05-31",
            "test_start": "2018-07-01",
            "test_end": "2018-12-30",
            "gap_days": 30,
            "kaggle_private": 0.914018,
            "kaggle_public": 0.944058,
            "recall_by_count": 0.44587,
            "recall_by_value": 0.312,
            "mean_caught_fraud": 105.0,
            "mean_missed_fraud": 186.0,
        },
        "cost_assumptions": {
            "review_per_case": COST_REVIEW_PER_CASE,
            "chargeback_fee": COST_CHARGEBACK_FEE,
            "false_alarm_friction": COST_FALSE_ALARM_FRICTION,
            "recovery_rate": FRAUD_RECOVERY_RATE,
            "review_capacity": REVIEW_CAPACITY_RATE,
        },
        "thresholds": {
            "psi_significant": PSI_SIGNIFICANT,
            "retrain_weighted_psi": RETRAIN_WEIGHTED_PSI,
        },
        "model_comparison": comparison,
        "threshold_analysis": thresholds,
        "cv_results": cv,
        "feature_importance": importance,
        "weekly_performance": weekly,
        "monthly_drift": monthly,
        "top_drift": drift_records,
        "example_transaction": EXAMPLE_TRANSACTION,
        "example_response": capture_example_response(),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nWrote {OUTPUT_FILE.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")
    print(
        f"  {len(comparison)} models, {len(weekly)} weeks, "
        f"{len(monthly)} months, {len(importance)} features"
    )


if __name__ == "__main__":
    main()
