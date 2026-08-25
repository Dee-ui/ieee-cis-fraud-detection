"""
Promote a registered model version to production.

Usage:
  python scripts/promote_model.py --version 2 --dry-run
  python scripts/promote_model.py --version 2

Promotion moves the 'production' alias. Step 6 loads whatever that alias
points at, so deploying a new model means moving a pointer rather than
editing and redeploying code.

--dry-run runs every gate and reports, without moving anything. Use it first,
always.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import mlflow  # noqa: E402

from config.config import (  # noqa: E402
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS_PRODUCTION,
    MODEL_METADATA_FILE,
    PREPROCESSOR_FILE,
    REGISTERED_MODEL_NAME,
)
from src.monitoring.promotion import (  # noqa: E402
    all_passed,
    evaluate_gates,
    format_gates,
)


def _production_metrics(client) -> dict | None:
    """Metrics of whatever is currently in production, if anything is."""
    try:
        version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME, MODEL_ALIAS_PRODUCTION
        )
    except Exception:  # noqa: BLE001
        return None
    return client.get_run(version.run_id).data.metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model to production.")
    parser.add_argument("--version", required=True, help="Registry version number.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check every gate and report, without moving the alias.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    print("=" * 60)
    print(f"PROMOTION CHECK: {REGISTERED_MODEL_NAME} version {args.version}")
    print("=" * 60)

    version = client.get_model_version(REGISTERED_MODEL_NAME, args.version)
    run = client.get_run(version.run_id)

    metadata = None
    if MODEL_METADATA_FILE.exists():
        candidate_metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))
        # The metadata file describes the most recent training run. Only use
        # it if it really is the version being promoted, otherwise the gates
        # would be checking the wrong model.
        if str(candidate_metadata.get("registered_version")) == str(args.version):
            metadata = candidate_metadata
        else:
            print(
                f"  Note: {MODEL_METADATA_FILE.name} describes version "
                f"{candidate_metadata.get('registered_version')}, "
                f"not {args.version}. Gates needing it will fail."
            )

    transformer_features = None
    if PREPROCESSOR_FILE.exists():
        transformer_features = list(joblib.load(PREPROCESSOR_FILE).feature_names_)

    gates = evaluate_gates(
        run_tags=run.data.tags,
        run_metrics=run.data.metrics,
        metadata=metadata,
        transformer_features=transformer_features,
        production_metrics=_production_metrics(client),
    )

    print(f"\n  run id: {version.run_id}\n")
    print(format_gates(gates))

    if not all_passed(gates):
        print("\n  RESULT: promotion refused. Fix the failures above.")
        sys.exit(1)

    if args.dry_run:
        print("\n  RESULT: every gate passed. Re-run without --dry-run to promote.")
        return

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, MODEL_ALIAS_PRODUCTION, args.version
    )
    print(f"\n  RESULT: version {args.version} is now " f"'{MODEL_ALIAS_PRODUCTION}'.")
    print("  Step 6 loads whatever this alias points at.")


if __name__ == "__main__":
    main()
