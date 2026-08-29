"""
Upload the model artefacts to the Hugging Face Model Hub.

Why the Hub rather than putting them in the image: the model version becomes
independent of the code version. Retrain, upload, restart the container, and
the new model is live without rebuilding or redeploying anything. D-65.

The repository is public, so the Space that consumes it needs no credentials
at all. A deployment that needs no secret cannot leak one. D-67.

Usage:
    python scripts/publish_model.py
    python scripts/publish_model.py --private
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import (  # noqa: E402
    FINAL_MODEL_FILE,
    HF_MODEL_REPO,
    HF_TOKEN,
    MODEL_METADATA_FILE,
    PREPROCESSOR_FILE,
)


def build_model_card(metadata: dict) -> str:
    """
    The README that appears on the model page.

    Worth writing properly. It is a public page with your name on it, and it
    is often the first thing someone sees.
    """
    return f"""---
license: mit
tags:
  - fraud-detection
  - tabular-classification
  - lightgbm
---

# IEEE-CIS Fraud Detection

A {metadata['model_family']} model that scores card transactions for fraud
risk, trained on the IEEE-CIS Fraud Detection dataset.

Full project: https://github.com/Dee-ui/ieee-cis-fraud-detection

## What is in this repository

| File | Contents |
|------|----------|
| `feature_engineer.joblib` | The fitted transformer. Turns a raw transaction into {metadata['n_features']} features. |
| `final_model.joblib` | The trained model. |
| `final_model_metadata.json` | Feature list, threshold, and the scores it was measured at. |

Both files are needed. The model expects features in one exact order, which
only the transformer produces.

## How it performs

Measured on a held-out period that comes strictly after everything it was
trained on, 2018-04-20 to 2018-05-31.

| Metric | Baseline | This model |
|--------|----------|------------|
| PR-AUC | 0.0344 | {metadata.get('selection_pr_auc', 0):.4f} |
| Cross-validated PR-AUC | - | {metadata.get('cv_pr_auc_mean', 0):.4f} |

Operating threshold **{metadata['chosen_threshold']:.4f}**, chosen by a cost
model at a {metadata.get('chosen_review_rate', 0.02):.0%} manual review
capacity, not left at the default 0.5.

## Using it

```python
import joblib, json, pandas as pd
from huggingface_hub import hf_hub_download

repo = "{HF_MODEL_REPO}"
engineer = joblib.load(hf_hub_download(repo, "feature_engineer.joblib"))
model = joblib.load(hf_hub_download(repo, "final_model.joblib"))
metadata = json.load(open(hf_hub_download(repo, "final_model_metadata.json")))

# A raw transaction. Any column you leave out is treated as unknown.
transaction = pd.DataFrame([{{
    "TransactionID": 3663549, "TransactionDT": 18403224,
    "TransactionAmt": 31.95, "ProductCD": "W", "card1": 10409,
}}])

features = engineer.transform(transaction)
probability = model.predict_proba(features[metadata["feature_names"]])[:, 1][0]
print(probability, "review" if probability >= metadata["chosen_threshold"] else "pass")
```

## Limitations worth knowing

- It catches about 44.6% of fraud **by count** but only 31.2% **by value**.
  Missed frauds average $186 against $105 for caught ones. Do not estimate
  savings by multiplying recall by total fraud losses.
- Roughly 10% of its decision weight sits on features derived from a
  customer fingerprint that is unavailable for about 82% of transactions in
  the later test period. Performance on data far from the training window
  should be monitored, not assumed.
- Trained on 2017 to 2018 data. Fraud patterns move.

## Licence

MIT.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish artefacts to the Model Hub.")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private. The Space would then need a token, "
        "so public is preferred.",
    )
    args = parser.parse_args()

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set.")
        print("Put it in .env, which is git-ignored. See step6.md section 1.")
        sys.exit(1)

    if not HF_MODEL_REPO:
        print("ERROR: HF_MODEL_REPO is not set. Add it to .env, for example:")
        print("  HF_MODEL_REPO=your-username/ieee-cis-fraud-detector")
        sys.exit(1)

    for path in (PREPROCESSOR_FILE, FINAL_MODEL_FILE, MODEL_METADATA_FILE):
        if not path.exists():
            print(f"ERROR: {path} not found. Run the training stage first.")
            sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    print(f"Publishing to {HF_MODEL_REPO} ...")
    api.create_repo(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    metadata = json.loads(MODEL_METADATA_FILE.read_text(encoding="utf-8"))

    card_path = Path("README_model_card.md")
    card_path.write_text(build_model_card(metadata), encoding="utf-8")

    uploads = [
        (PREPROCESSOR_FILE, "feature_engineer.joblib"),
        (FINAL_MODEL_FILE, "final_model.joblib"),
        (MODEL_METADATA_FILE, "final_model_metadata.json"),
        (card_path, "README.md"),
    ]

    for local_path, remote_name in uploads:
        size_mb = local_path.stat().st_size / 1024**2
        print(f"  uploading {remote_name} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )

    card_path.unlink()

    print(f"\nDone: https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
