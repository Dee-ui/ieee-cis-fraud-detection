# IEEE-CIS Fraud Detection

[![CI](https://github.com/Dee-ui/ieee-cis-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Dee-ui/ieee-cis-fraud-detection/actions/workflows/ci.yml)
[![Live API](https://img.shields.io/badge/live%20demo-Render-46E3B7)](https://ieee-cis-fraud-detection.onrender.com/docs)

An end-to-end machine learning and MLOps project that detects fraudulent card
transactions, covering the full lifecycle from raw data to a monitored,
containerised, deployed service with an interactive dashboard.

**[Try the live API](https://ieee-cis-fraud-detection.onrender.com/docs)** —
send a transaction, get a fraud score and an explanation of what drove it.
No installation. (Free-tier hosting: the first request after a few minutes
of inactivity may take 30–60s to wake the service.)

> Status: in progress. Steps 1 to 6 of 7 complete.

---

## Headline result

A LightGBM model that catches **44.6% of fraudulent transactions while
reviewing 2% of all traffic**, worth roughly **$1.76M a year** in prevented
losses under a documented cost model.

Scored **0.914 ROC-AUC on the Kaggle private leaderboard** as a single model,
with no ensembling and no test-set leakage.

---

## Problem

Card fraud is rare and expensive.

Rare, because about 3.5% of transactions in this dataset are fraudulent. A model
that predicts "never fraud" is 96.5% accurate and completely useless, which is
why accuracy is not used anywhere in this project.

Expensive, because the two ways of being wrong cost different things. A missed
fraud is a direct loss. A false alarm blocks a real customer's payment. Any
useful system has to be tuned against a real review capacity rather than
optimised in the abstract.

## Dataset

IEEE-CIS Fraud Detection (Kaggle competition, data provided by Vesta
Corporation).

| Table | Rows | Columns | Contents |
|-------|------|---------|----------|
| `train_transaction` | 590,540 | 394 | Transaction level, carries the `isFraud` label |
| `train_identity` | 144,233 | 41 | Device and network signals, only for some transactions |
| `test_transaction` | 506,691 | 393 | No label |
| `test_identity` | 141,907 | 41 | No label |

The two tables join on `TransactionID`. Most columns are anonymised: 339 are
engineered features supplied by Vesta with no published meaning.

Data is not stored in this repository. See Quickstart to download it.

## What the data shows

Full findings in [`reports/eda_summary.md`](reports/eda_summary.md).

![Class balance](reports/figures/01_class_balance.png)

**The imbalance.** 20,663 frauds out of 590,540, a rate of 3.4990%, roughly one
in twenty-nine.

**Time matters more than anything else.** Training covers 2017-12-01 to
2018-05-31; the test set covers 2018-07-01 to 2018-12-30, with a deliberate 30
day gap. The test set is entirely in the future, so validation is a time-based
split, never a random one.

**Fraud concentrates in identifiable places.** Product code C runs at 11.69%
against W at 2.04%. Credit cards run at 6.68% against debit at 2.43%. Mobile
devices run at 10.17% against desktop at 6.52%.

**Identity records are almost decided by product type.** The raw figures suggest
fraud is 3.75x more likely when an identity record exists. That comparison is
confounded: product W never produces one and also has the lowest fraud rate.
Restricted to products where the flag varies, the difference is 1.39x. The
model later confirmed this, ranking the flag 270th of 284 features with a SHAP
value of exactly zero.

**The 339 anonymous V columns have hidden structure.** They fall into 15 blocks
that go blank on identical rows. Eight of the fifteen interleave through each
other's number ranges, so the structure is invisible unless you compare the
actual missing patterns. Correlation clustering inside each block reduced 337
surviving columns to 137.

## Approach

- **Metric.** PR-AUC is primary, baseline 0.035. ROC-AUC secondary. Recall at a
  fixed review rate is the business headline. Accuracy is never reported.
- **Validation.** Time-based: the last 20% of the training period.
- **Missing values are left missing.** Boosted trees learn a direction for
  blanks at every split. Filling them would assert something untrue.
- **No leakage by construction.** Every learned transformation is fitted on the
  training portion only and saved as an object, so training and serving cannot
  drift apart. A test asserts that transforming one row gives the same answer as
  transforming a batch containing it.

## Results

Validation is the last 20% of the training period by time: 2018-04-20 to
2018-05-31, 118,108 transactions containing 4,064 frauds, never seen in training.

| Model | PR-AUC | ROC-AUC | Fit time |
|-------|--------|---------|----------|
| **LightGBM** | **0.6068** | 0.9275 | 43s |
| XGBoost | 0.5991 | **0.9308** | 4m 21s |
| CatBoost | 0.5291 | 0.8937 | 14m 39s |
| Logistic regression | 0.1831 | 0.8210 | 1m 04s |
| Random baseline | 0.0344 | 0.5000 | - |

Every candidate was trained to convergence. CatBoost was given a 4,000 round
budget after it hit a 1,500 round ceiling; the extra 2,500 rounds gained
0.0009, confirming it had plateaued rather than been cut short.

Note that CatBoost, despite the worse ranking, scored slightly better on the
cost model. PR-AUC counts transactions; the cost model weights by amount. See
`docs/steps/step6.md` section 3.5.

| Metric | Baseline | This model |
|--------|----------|------------|
| PR-AUC | 0.0344 | **0.6068** (17.6x) |
| ROC-AUC | 0.500 | **0.9275** |
| Recall at 1% review rate | 1.0% | **26.6%** |
| Recall at 2% review rate | 2.0% | **44.6%** |
| Kaggle private leaderboard | 0.500 | **0.9140** |

Stability across four expanding time windows: PR-AUC **0.6334**, spread
**0.0280**.

### What it is worth

Under a cost model with five stated assumptions, documented in
[`docs/steps/step4.md`](docs/steps/step4.md), running at a 2% manual review
capacity saves **$202,013 over the 42 day validation window**, which annualises
to roughly **$1.76M a year**.

The assumptions: $4.00 per analyst review, $25.00 chargeback fee per missed
fraud, $1.00 friction per false alarm, 90% of flagged fraud actually prevented,
and a team able to review 2% of transactions. All five live in
`config/config.py`. Change one, re-run, and the figure updates.

**One caveat that matters.** The model catches 44.6% of fraud **by count** but
only 31.2% **by value**. Missed frauds average $186 against $105 for caught
ones, because a large fraudulent purchase looks much like a large legitimate
one. The cost model accounts for this correctly, but any estimate built by
multiplying the recall figure by total fraud losses would overstate the benefit
by about 43%.

These are assumptions rather than figures from a business, and the savings
estimate should be read as an order of magnitude rather than a forecast.

## How it is kept honest

- **A test suite** covering the cost model against hand arithmetic, a
  joblib round-trip of the feature transformer, and structural guards against
  leakage. Tests run on synthetic data, so they work anywhere with no dataset.
- **CI on every push**: ruff, black, and pytest on a clean machine.
- **Drift monitoring** comparing every month of the unlabelled test period
  against the training distribution, using PSI weighted by feature importance,
  because drift in a feature the model ignores is not a problem.
- **Promotion gates.** A model reaches production only by passing six checks.
  One of them exists because a quick-mode test model once registered itself.
- **A container built in CI** on every push, then started and health-checked,
  so a Dockerfile that no longer builds is caught in about a minute.
- **No credentials anywhere in the repository.** A pre-commit hook refuses any
  commit containing something shaped like an API token.

## Deployment

The model runs as a containerised FastAPI service on **Render**.

| Piece | Where |
|-------|-------|
| Live API | [ieee-cis-fraud-detection.onrender.com](https://ieee-cis-fraud-detection.onrender.com/docs) |
| Model artefacts | [Hugging Face Model Hub](https://huggingface.co/Dee-ui/ieee-cis-fraud-detector) |
| Image | `Dockerfile`, `python:3.11-slim`, non-root, about 1.3 GB |

The API and the model are hosted separately on purpose. Hugging Face's Model
Hub remains the artefact store — public, free, no credentials needed to pull
from it. The running service itself was originally built for Hugging Face
Spaces, but Spaces now requires a paid PRO subscription to host Docker-SDK
containers on free hardware (a `402 Payment Required` error when attempting
to deploy). Rather than pay for that, the API moved to Render's free tier
instead, with no code changes beyond making the container's listening port
configurable via an environment variable.

The artefacts live on the Model Hub rather than inside the image, so a
retrained model ships by restarting the container instead of rebuilding and
redeploying it.

```bash
docker build -t fraud-api .
docker run -p 8000:7860 -e HF_MODEL_REPO=Dee-ui/ieee-cis-fraud-detector fraud-api
```

## Architecture

_Diagram added in Step 7._

## Quickstart

```bash
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS or Linux
pip install -r requirements.lock.txt

# Data (requires a Kaggle account that has joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# Build everything
python run.py --step all
```

## Pipeline

| Stage | Command | Output |
|-------|---------|--------|
| Ingestion | `python run.py --step ingestion` | `data/interim/*_joined.parquet` |
| EDA | `python run.py --step eda` | `reports/eda_summary.md`, 10 charts |
| Features | `python run.py --step features` | 284 features, `models/feature_engineer.joblib` |
| Training | `python run.py --step training` | `models/final_model.joblib`, MLflow runs |
| Monitoring | `python run.py --step monitoring` | `reports/monitoring/*`, 4 charts |
| Promotion | `python scripts/promote_model.py --version N` | Moves the production alias |
| Serving | `uvicorn src.serving.app:app` | Local API at `/docs` |

Every stage reads a file and writes a file, so any one can be run on its own.

## Project structure

See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the annotated
structure, the full decision log, and current status.

## Roadmap

- [x] Step 1: Dataset acquisition, scaffold, repo, environment
- [x] Step 2: Exploratory data analysis and data understanding
- [x] Step 3: Feature engineering and preprocessing pipeline
- [x] Step 4: Model training with MLflow experiment tracking
- [x] Step 5: Testing, CI, drift monitoring, and promotion gates
- [x] Step 6: Dockerisation and deployment
- [ ] Step 7: Dashboard and portfolio packaging

## Tech stack

Python 3.11, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, MLflow, SHAP,
DVC, pytest, ruff, GitHub Actions, FastAPI, Docker, Render, Streamlit.

## Licence

MIT. See [`LICENSE`](LICENSE).
