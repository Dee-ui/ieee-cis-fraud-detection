# IEEE-CIS Fraud Detection

An end-to-end machine learning and MLOps project that detects fraudulent card
transactions, covering the full lifecycle from raw data to a monitored,
containerised, deployed service with an interactive dashboard.

> Status: in progress. Steps 1 to 3 of 7 complete.

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

The goal is a model that ranks transactions by risk well enough to be useful at
a realistic review budget, plus the engineering around it that makes it
deployable, observable, and maintainable.

## Dataset

IEEE-CIS Fraud Detection (Kaggle competition, data provided by Vesta
Corporation).

| Table | Rows | Columns | Contents |
|-------|------|---------|----------|
| `train_transaction` | 590,540 | 394 | Transaction level, carries the `isFraud` label |
| `train_identity` | 144,233 | 41 | Device and network signals, only for some transactions |
| `test_transaction` | 506,691 | 393 | No label |
| `test_identity` | 141,907 | 41 | No label |

The two tables join on `TransactionID`. Most columns are anonymised: 339 of them
are engineered features supplied by Vesta with no published meaning, and the
identity columns are similarly masked.

Data is not stored in this repository. See Quickstart to download it.

## What the data shows

Full findings in [`reports/eda_summary.md`](reports/eda_summary.md), with charts
in [`reports/figures/`](reports/figures/).

![Class balance](reports/figures/01_class_balance.png)

**The imbalance.** 20,663 fraudulent transactions out of 590,540, a rate of
3.4990%, roughly one in twenty-nine.

**Time matters more than anything else.** Training covers 2017-12-01 to
2018-05-31. The test set covers 2018-07-01 to 2018-12-30. There is a deliberate
30 day gap between them. The test set is entirely in the future, so validation
here is a time-based split, never a random one. A random split would let the
model learn from transactions that happened after the ones it is scored on.

**Fraud concentrates in identifiable places.** Product code C runs at 11.69%
against product W at 2.04%. Credit cards run at 6.68% against debit at 2.43%.
Mobile devices run at 10.17% against desktop at 6.52%. Some email domains run
above 18% while others sit below 2.5%.

**Identity records are almost decided by product type.** Only 24.4% of
transactions have a matching identity record, and the raw figures suggest fraud
is 3.75 times as likely among those that do. That comparison is confounded:
product W never produces an identity record and also has the lowest fraud rate,
while every other product almost always produces one. Restricted to the products
where the flag actually varies, the difference is closer to 1.4x.

**The 339 anonymous V columns have hidden structure.** They fall into 15 blocks
that go blank on exactly the same rows, which is the fingerprint of features
built in batches from shared source data. Eight of the fifteen blocks interleave
through each other's number ranges, so the structure is invisible unless you
compare the actual missing patterns. Correlation clustering inside each block
reduces the 339 columns substantially without discarding them arbitrarily.

## Approach

- **Metric.** PR-AUC is primary, with a baseline of 0.035 equal to the fraud
  rate. ROC-AUC is reported alongside it. Recall at a 1% manual review rate is
  the headline business figure. Accuracy is not reported.
- **Validation.** A time-based split: the last 20% of the training period by
  `TransactionDT`.
- **Missing values are left missing.** LightGBM, XGBoost, and CatBoost all learn
  a direction for blanks at every split. Filling them with an average would
  assert something untrue.
- **No leakage by construction.** Every learned transformation is fitted on the
  training portion only and saved as an object, so training and serving cannot
  drift apart.

## Architecture

_Diagram added in Step 6._

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

# 2. Environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS or Linux
pip install -r requirements.lock.txt

# 3. Data (requires a Kaggle account that has joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# 4. Build everything
python run.py --step all
```

## Pipeline

| Stage | Command | Input | Output |
|-------|---------|-------|--------|
| Ingestion | `python run.py --step ingestion` | Raw CSVs | `data/interim/*_joined.parquet` |
| EDA | `python run.py --step eda` | Joined Parquet | `reports/eda_summary.md`, charts |
| Features | `python run.py --step features` | Joined Parquet | `data/processed/*_features.parquet`, `models/feature_engineer.joblib` |

Every stage reads a file and writes a file, so any one of them can be run and
debugged on its own.

## Project structure

See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) for the annotated structure,
the full decision log, and current status.

## Roadmap

- [x] Step 1: Dataset acquisition, scaffold, repo, environment
- [x] Step 2: Exploratory data analysis and data understanding
- [x] Step 3: Feature engineering and preprocessing pipeline
- [ ] Step 4: Model training with MLflow experiment tracking
- [ ] Step 5: MLOps layer: CI/CD, testing, model registry, drift monitoring
- [ ] Step 6: Dockerisation and deployment
- [ ] Step 7: Dashboard and portfolio packaging

## Results

Validation is the last 20% of the training period by time, from 2018-04-20 to
2018-05-31: 118,108 transactions containing 4,064 frauds. The model never sees
any of it during training.

| Metric | Baseline | Best model |
|--------|----------|------------|
| PR-AUC | 0.035 | 0.60682 |
| ROC-AUC | 0.500 | 0.92751 |
| Recall at 1% review rate | 0.010 | 0.26599 |

TStability across four expanding time windows: PR-AUC TBD, spread TBD.

### What it is worth

Under a cost model with five stated assumptions, documented in
`docs/steps/step4.md`, running the model at a 2% manual review capacity is
worth roughly **$202,013 a year** in prevented fraud, net of review costs.

The assumptions: $4.00 per analyst review, $25.00 chargeback fee per missed
fraud, $1.00 friction per false alarm, 90% of flagged fraud actually prevented,
and a team able to review 2% of transactions. All five live in
`config/config.py`. Change one, re-run, and the figure updates.

These are assumptions rather than figures from a business, and the savings
estimate should be read as an order of magnitude rather than a forecast.

## Tech stack

Python 3.11, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, MLflow, SHAP,
DVC, FastAPI, Docker, GitHub Actions, Streamlit.

## Licence

MIT. See [`LICENSE`](LICENSE).
