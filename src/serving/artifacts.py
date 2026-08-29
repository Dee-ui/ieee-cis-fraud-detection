"""
Finding and loading the model artefacts.

Two sources, tried in order:

  1. Local files, for development and for anyone who has run the pipeline.
  2. The Hugging Face Model Hub, for the container, which ships with no
     artefacts inside it.

Keeping the model outside the image means a retrained model can be shipped
by restarting the container rather than rebuilding and redeploying it. That
is decision D-65.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib

from config.config import (
    ARTIFACT_CACHE_DIR,
    FINAL_MODEL_FILE,
    HF_MODEL_REPO,
    MODEL_METADATA_FILE,
    PREPROCESSOR_FILE,
)

ARTIFACT_FILES = [
    "feature_engineer.joblib",
    "final_model.joblib",
    "final_model_metadata.json",
]


@dataclass
class Artifacts:
    """Everything the service needs to turn a transaction into a score."""

    engineer: object
    model: object
    metadata: dict

    @property
    def feature_names(self) -> list[str]:
        return list(self.metadata["feature_names"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["chosen_threshold"])


def _local_paths_exist() -> bool:
    return all(
        path.exists()
        for path in (PREPROCESSOR_FILE, FINAL_MODEL_FILE, MODEL_METADATA_FILE)
    )


def _download_from_hub() -> Path:
    """
    Fetch the artefacts from the Model Hub into the cache folder.

    snapshot_download only fetches files that have changed, so a container
    restart with an unchanged model is fast.
    """
    from huggingface_hub import snapshot_download

    if not HF_MODEL_REPO:
        raise RuntimeError(
            "No artefacts on disk and HF_MODEL_REPO is not set. Either run the "
            "pipeline locally or set HF_MODEL_REPO to the Model Hub repository."
        )

    print(f"  Downloading artefacts from {HF_MODEL_REPO} ...")
    location = snapshot_download(
        repo_id=HF_MODEL_REPO,
        allow_patterns=ARTIFACT_FILES,
        cache_dir=str(ARTIFACT_CACHE_DIR),
    )
    return Path(location)


def load_artifacts() -> Artifacts:
    """Load from disk if possible, otherwise from the Model Hub."""
    if _local_paths_exist():
        print("  Loading artefacts from local files ...")
        engineer_path = PREPROCESSOR_FILE
        model_path = FINAL_MODEL_FILE
        metadata_path = MODEL_METADATA_FILE
    else:
        folder = _download_from_hub()
        engineer_path = folder / "feature_engineer.joblib"
        model_path = folder / "final_model.joblib"
        metadata_path = folder / "final_model_metadata.json"

    engineer = joblib.load(engineer_path)
    model = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    # The two must agree, or the model is being fed columns in the wrong
    # order and every prediction is quietly wrong. Promotion gate 6 checks
    # this too; checking again at load costs nothing and covers the case
    # where someone swapped a file by hand.
    if list(engineer.feature_names_) != list(metadata["feature_names"]):
        raise RuntimeError(
            f"Feature mismatch: the transformer produces "
            f"{len(engineer.feature_names_)} features, the model expects "
            f"{len(metadata['feature_names'])}. Do not serve this."
        )

    print(
        f"  Loaded {metadata['model_family']}, "
        f"{len(metadata['feature_names'])} features, "
        f"threshold {metadata['chosen_threshold']:.4f}"
    )

    return Artifacts(engineer=engineer, model=model, metadata=metadata)


def expected_raw_columns(engineer) -> list[str]:
    """
    Every raw column the transformer might look for.

    The service builds a frame containing all of these, blank by default,
    then fills in whatever the caller sent. That way a transaction with six
    fields and one with four hundred both produce a valid frame, and the
    transformer never trips over a missing column.
    """
    from config.config import (
        AMOUNT_COLUMN,
        M_COLUMNS,
        TIME_COLUMN,
        UID_ADDRESS_COLUMN,
        UID_CARD_COLUMN,
        UID_TIMEDELTA_COLUMN,
    )

    columns = set(engineer.base_columns_)
    columns |= set(getattr(engineer, "label_source_columns_", []))
    columns |= {
        TIME_COLUMN,
        AMOUNT_COLUMN,
        UID_CARD_COLUMN,
        UID_ADDRESS_COLUMN,
        UID_TIMEDELTA_COLUMN,
    }
    columns |= set(M_COLUMNS)

    return sorted(columns)
