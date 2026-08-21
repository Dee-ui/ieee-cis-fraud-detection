# PROJECT_STATE.md

**Last updated:** End of Step 4 of 7
**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Local path:** `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`

---

## 0. What this document is

The anchor for the whole project. Rewritten in full at the end of every step, never patched with a diff.

If earlier conversation is lost, this file alone is enough to pick up exactly where we stopped. It records every decision, the current state of the repository, every verified result, what is finished, what is outstanding, and what questions remain open.

---

## 1. Project at a glance

| Item | Value |
|------|-------|
| Goal | A complete, portfolio-grade fraud detection system covering the full machine learning and MLOps lifecycle |
| Dataset | IEEE-CIS Fraud Detection (Kaggle, data provided by Vesta Corporation) |
| Scope | Data pipeline, feature engineering, model training with experiment tracking, CI/CD, model registry, drift monitoring, Docker, deployment, dashboard |
| Delivery format | 7 steps, one per conversation message, each with its own markdown guide plus a refreshed copy of this file |
| Platform | Windows, VS Code, PowerShell, Python 3.11.9 |
| Machine | Intel Core Ultra 7 265H, 32 GB RAM |
| Local path | `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`, offline, outside any sync folder |
| Version control | Git and GitHub for code, DVC with a local folder remote for processed data |
| DVC remote | `C:\Users\Dauda Agbonoga\dvcstore`, verified working |
| Tracks | Technical track (this work) and a separate PM track walkthrough afterwards with the project manager |
| Audience for the final artefacts | Hiring managers and portfolio reviewers |

---

## 2. Why this dataset

Card fraud is a rare-event problem with a real cost structure on both sides: a missed fraud is a direct loss, a false alarm blocks a paying customer. Every technical decision has a business consequence you can point at, which makes it an unusually good subject for an end-to-end project.

Four properties make IEEE-CIS worth the effort:

- **Enough positive cases.** 20,663 frauds in 590,540 transactions, a rate of 3.4990%. Rare enough to be realistic, common enough to learn from.
- **Two joinable tables** with partial coverage, so there is real data engineering rather than a single flat file.
- **Mostly anonymised features.** 339 columns are Vesta-engineered with no published meaning, so structure has to be found empirically.
- **A test set 30 days in the future**, which makes time-based validation unavoidable and gives Step 5 a genuine distribution shift to detect.

---

## 3. The 7-step plan and current status

| Step | Content | Status |
|------|---------|--------|
| 1 | Dataset acquisition, folder scaffold, GitHub repo, Python environment | **Complete, verified** |
| 2 | EDA and data understanding: joins, feature families, imbalance profiling | **Complete, verified** |
| 3 | Feature engineering and preprocessing pipeline | **Complete, verified** |
| 4 | Model training with MLflow experiment tracking | **Delivered, awaiting the user's run** |
| 5 | MLOps layer: CI/CD, testing, model registry, drift monitoring | Not started |
| 6 | Dockerisation and deployment | Not started |
| 7 | Dashboard and portfolio packaging | Not started |

---

## 4. Decision log

### Step 1: foundations

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | Dataset is IEEE-CIS Fraud Detection | 20,663 positive cases, a joinable second table, and a time-separated test set. |
| D-02 | Repository named `ieee-cis-fraud-detection` | Descriptive and scannable. The local folder matches as of Step 3. |
| D-03 | Python 3.11 | Required minimum for the current Kaggle CLI, stable prebuilt Windows packages for every library needed. Confirmed 3.11.9. |
| D-04 | `venv` plus `requirements.txt`, not conda | One dependency format that Docker and GitHub Actions both consume natively. |
| D-05 | Raw data never committed to Git | Roughly 1.3 GB of CSV. Reproducibility comes from `scripts/download_data.py`. |
| D-06 | DVC deferred to Step 3 | Resolved by D-32. |
| D-07 | Paths resolved dynamically in `config/config.py` | Proven twice: once with a mismatched folder name, once when the whole project moved with zero code changes. |
| D-08 | Branch per step, merged by pull request, tagged after merge | Reviewable trail, triggers CI in Step 5, clean narrative for the PM walkthrough. |
| D-09 | Public GitHub repository | Portfolio piece. Also gives free Actions minutes. |
| D-10 | Dependencies split into runtime and dev, plus a lock file | Keeps the Step 6 Docker image lean; the lock file gives exact reproducibility. |
| D-11 | Download script shells out to the Kaggle CLI | The CLI is the documented stable contract; the Python library's interface changes across versions. |
| D-12 | Interim and processed data stored as Parquet | Far smaller, much faster, and it preserves data types. |
| D-13 | Notebooks for exploration only; anything that matters becomes a module in `src/` | Notebooks are not testable, importable, or reviewable in diffs. |

### Step 2: data understanding

| ID | Decision | Rationale |
|----|----------|-----------|
| D-14 | `run.py` created in Step 2 rather than Step 3 | Two runnable stages existed, which is where a single entry point starts earning its keep. |
| D-15 | Test set joined and saved despite having no labels | It becomes the Step 5 drift input. Starting 30 days after training ends, it shows genuine shift rather than manufactured noise. |
| D-16 | Left join transaction to identity, keep blanks, add a `has_identity` flag | All three boosters learn a direction for blanks at every split, so blanks cost nothing. Partially revised by D-31. |
| D-17 | Interim data as Parquet with category dtypes preserved | Type work is done once, not on every read. |
| D-18 | `TransactionAmt` stays `float64`; `TransactionID` and `TransactionDT` become `int32`; everything else shrunk | Verified numerically: `float32` is exact for whole numbers only below 16,777,216, and test `TransactionDT` reaches 34,214,345. `float32` also turns 31937.39 into 31937.390625, and the cents are a fraud signal. |
| D-19 | Reference date 30 November 2017 for display only | The competition never published a start date. `TransactionDT` is only used as an ordering and a duration. |
| D-20 | PR-AUC primary, ROC-AUC secondary, recall at a fixed review rate as the business headline, accuracy never | A do-nothing model scores 96.5% accuracy. ROC-AUC is insensitive to false positives against 569,877 negatives. |
| D-21 | Validation is a time-based split, last 20% by `TransactionDT` | The real test set is 30 days in the future. A random split leaks three ways: repeated cards, `D` columns encoding elapsed time, and fraud arriving in bursts. |
| D-22 | Feature families assigned by rule, unmapped columns reported loudly | Zero unmapped columns in the actual run. |

### Step 3: feature engineering

| ID | Decision | Rationale |
|----|----------|-----------|
| D-23 | Feature engineering is a fitted object saved with joblib, built on scikit-learn's `BaseEstimator` and `TransformerMixin` | A single transaction arriving at a web service cannot recompute frequency counts or group averages. They must have been saved. |
| D-24 | The transformer is fitted only on the first 80% of the training period | Counting frequencies across validation rows means each validation row helped compute its own feature. Costs a little accuracy, buys a trustworthy number. |
| D-25 | Encodings learned from training rows only, never train and test combined | Combining is common in competition write-ups but impossible in production. |
| D-26 | `TransactionDT`, `TransactionID`, and any absolute day counter excluded from features | Test values sit entirely above training values. Trees cannot split outside their trained range. Only cyclical time features survive. |
| D-27 | Near-constant columns dropped at a 99% dominance threshold, with a rescue rule | Vindicated dramatically. See Section 5.5. |
| D-28 | V columns reduced by correlation clustering inside each of the 15 blocks at 0.75 | Blocks identify shared sources; correlation identifies near-duplicates within them. |
| D-29 | The `uid` fingerprint is used only for grouping and counting, never as a feature | Given directly, the model memorises individual customers. |
| D-30 | Text columns become integers with a stored mapping; blank gets its own code; unseen maps to -1 | Without a stored mapping the same word gets a different number next month and the model silently breaks. |
| D-31 | `has_identity` kept, with the confound recorded | The headline 3.75x is mostly a `ProductCD` effect. Within non-W products the lift is 1.39x. Expected to rank low in Step 4. |
| D-32 | DVC with a local folder remote at `C:\Users\Dauda Agbonoga\dvcstore` | No account, no internet, no cost. Required removing `data/processed/*` from `.gitignore`. Verified: `dvc pull` restored a deleted file. |
| D-33 | The Streamlit dashboard draws from two distinct sources and reads precomputed artifacts | A static model profile, a live operations view, and a drift view that deliberately overlays the two. |

### Step 4: model training

| ID | Decision | Rationale |
|----|----------|-----------|
| D-34 | A cost model with five explicitly stated assumptions, stored in config, drives the threshold choice | Turns an abstract metric into money. Assumptions in config means challenging one and re-running takes minutes. |
| D-35 | Costs weighted by the actual transaction amount, not a flat penalty per fraud | A missed $2,000 fraud is not a missed $20 fraud. Amount weighting makes the optimum naturally favour expensive fraud. |
| D-36 | The uid features get a pre-registered ablation with a 0.005 PR-AUC threshold set before the result is seen | Those six features are blank on 82% of test rows. A rule chosen after the fact is a justification, not a rule. |
| D-37 | Five candidates: dummy, logistic regression, LightGBM, XGBoost, CatBoost | The dummy gives the true measured floor. Logistic regression makes the boosters earn their complexity. |
| D-38 | No class weighting and no resampling | We need ordering, and weighting shifts probabilities without reliably improving order. It also makes scores uninterpretable, which hurts the Step 7 dashboard. |
| D-39 | Category codes treated as ordinary numbers, not declared categorical to the boosters | Frequency counts already supply the same information in a form that cannot overfit. Native categorical handling on a 1,786-value column tends to memorise. |
| D-40 | Early stopping on validation PR-AUC; cross-validation runs afterwards with the round count fixed | Early stopping inside every fold makes each fold optimistic about itself. Fixing the count keeps CV an honest stability check. |
| D-41 | The final model is retrained on all labelled data with the round count scaled by the row ratio, about 1.25x | Validation picks the settings; the shipped model should still see every labelled row. |
| D-42 | MLflow tracking URI built with `.as_posix()` | Windows backslashes are unreliable inside a SQLAlchemy database URL. |
| D-43 | The chosen model is registered under an alias, not a stage | MLflow 3 deprecated stages. Step 5 promotes by moving the alias; Step 6 loads whatever it points at. |
| D-44 | Step 6 deploys to Hugging Face Spaces using the Docker SDK | Answers Q-04. Genuinely free, runs the real Docker image so Step 6 is not decoration, public clickable URL with no cold start, recognised venue, and FastAPI's `/docs` gives an interactive demo. Render free tier is the backup, at the cost of a spin-down delay. |
| D-45 | The Step 7 dashboard is built for a hiring manager reading it cold in under two minutes | Answers Q-12. Loads in under three seconds from precomputed files, no unexplained jargon, leads with the money, one interactive scorer, and visible MLOps signals. The full EDA stays in `reports/`. |
| D-46 | A Kaggle late submission is produced | Answers Q-05. Free, and it gives one externally verified number. Scored on ROC-AUC, which is not our primary metric, and that mismatch is reported honestly. |

---

## 5. Verified results

### 5.1 Raw data, Step 1

| File | Size | Rows | Columns |
|------|------|------|---------|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 |
| `sample_submission.csv` | 5.8 MB | 506,691 | 2 |

Fraud rate **3.4990%**, 20,663 of 590,540. `TransactionID` unique in both training tables. 144,233 transactions have an identity record, **24.4%**. Train uses `id_` prefixes, test uses `id-`.

### 5.2 Ingestion, Step 2

| Split | Rows | Columns | Memory before | Memory after | Reduction | Parquet |
|-------|------|---------|---------------|--------------|-----------|---------|
| train | 590,540 | 435 | 2,567.7 MB | 927.2 MB | 63.9% | 80.3 MB |
| test | 506,691 | 434 | 2,214.5 MB | 795.2 MB | 64.1% | 69.8 MB |

Runtime 3 minutes 7 seconds. Train dtypes: 398 `float32`, 31 `category`, 2 `int8`, 2 `int32`, 1 `float64`, 1 `int16`.

Parquet came out about four times smaller than estimated. The table is unusually compressible: mostly `float32` with many blanks, which Parquet stores as a compact bitmap, and long runs of repeated values.

### 5.3 EDA, Step 2

**Time coverage**

| Split | First | Last | Span |
|-------|-------|------|------|
| train | 2017-12-01 | 2018-05-31 | 182.0 days |
| test | 2018-07-01 | 2018-12-30 | 183.0 days |

Gap: **30.0 days**.

**Identity coverage and the confound**

| Group | Transactions | Fraud rate |
|-------|--------------|------------|
| No identity record | 446,307 | 2.0939% |
| Has identity record | 144,233 | 7.8470% |

**The 3.75x figure must never be quoted without its caveat.** Identity coverage is almost decided by `ProductCD`: W is 0% covered, C is 90.8%, H, R, and S are 99.6% each. W also has the lowest fraud rate (2.04%) and is 439,670 of 590,540 rows. Restricted to non-W products: 7.85% with identity (144,233 rows) against 5.67% without (6,637 rows), a lift of **1.39x**. See D-31.

Test identity coverage is **28.0%** against training's 24.4%, a real 3.6 point shift. This is one of the two worked drift examples for Step 5.

**Fraud rate by category**

| Column | Highest | Lowest |
|--------|---------|--------|
| ProductCD | C at 11.69% | W at 2.04% |
| card4 | discover at 7.73% | american express at 2.87% |
| card6 | credit at 6.68% | debit at 2.43% |
| DeviceType | mobile at 10.17% | missing at 2.10% |
| P_emaildomain | mail.com at 18.96% | aol.com at 2.18% |

**Missing data.** 53 columns with none at all. 12 above 90%. The nine emptiest are identity columns between 99.12% and 99.20%. `dist2` 93.63%, `D7` 93.41%. The C family has zero missing across all 14.

### 5.4 The V column blocks, Step 2

All 339 V columns fall into 15 blocks sharing an identical missing pattern.

| Block | Columns | Missing | Number range | Shape | Kept after reduction |
|-------|---------|---------|--------------|-------|----------------------|
| 1 | 46 | 77.91% | V217 to V278 | interleaved | 13 |
| 2 | 43 | 0.05% | V95 to V137 | solid run | 24 (from 42 after pruning) |
| 3 | 32 | 0.00% | V279 to V321 | interleaved | 15 (from 31) |
| 4 | 31 | 76.36% | V167 to V216 | interleaved | 8 |
| 5 | 23 | 12.88% | V12 to V34 | solid run | 8 |
| 6 | 22 | 13.06% | V53 to V74 | solid run | 8 |
| 7 | 20 | 15.10% | V75 to V94 | solid run | 8 |
| 8 | 19 | 76.32% | V169 to V210 | interleaved | 10 |
| 9 | 18 | 28.61% | V35 to V52 | solid run | 8 |
| 10 | 18 | 86.12% | V138 to V163 | interleaved | 6 |
| 11 | 18 | 86.05% | V322 to V339 | solid run | 7 |
| 12 | 16 | 76.05% | V220 to V272 | interleaved | 6 |
| 13 | 11 | 47.29% | V1 to V11 | solid run | 7 |
| 14 | 11 | 86.12% | V143 to V166 | interleaved | 2 |
| 15 | 11 | 0.21% | V281 to V315 | interleaved | 7 |

Two structural facts:

**Blocks 10 and 14 both sit at 86.12% missing but are different blocks.** Same count of blanks, different rows. Grouping by missing count would have merged them; hashing the actual pattern separated them.

**Eight of fifteen blocks are interleaved**, so their V numbers weave through each other. Blocks 4 and 8 thread through V167 to V216; blocks 10 and 14 through V138 to V166. Chopping by number range would cut across the real groupings.

The reduction rate varies widely by block, from 18% kept in block 14 to 57% in block 2. A flat rule would have over-cut one and under-cut the other.

### 5.5 Feature engineering, Step 3

Runtime 2 minutes 25 seconds. All four verification checks passed.

| Stage | Result |
|-------|--------|
| Input columns | 435 |
| Passthrough (not features) | 3 |
| Candidates assessed | 432 |
| Dropped: single value | 0 |
| Dropped: near-constant | 2 (V107, V305) |
| Rescued | 22 |
| Survivors | 430 |
| V columns | 337 reduced to 137 |
| Base columns | 230, of which 199 numeric and 31 text |
| **Final features** | **284** |
| Transformer file | 28.0 MB |
| Train Parquet | 84.4 MB |
| Test Parquet | 68.5 MB |
| Unseen lookups in test | 6.81% |

**Feature composition:** 199 base_numeric, 38 category_code, 18 aggregate, 18 frequency, 3 derived_amount, 3 derived_screen, 2 derived_time, 2 derived_match, 1 derived_email.

**The rescue rule was strongly vindicated.** Only 2 columns were dropped; 22 were rescued:

| Column | Dominant | Share | Rare rows | Fraud rate among rare | Lift |
|--------|----------|-------|-----------|----------------------|------|
| V111 | 1.0 | 99.71% | 1,370 | **46.35%** | 13.2x |
| V113 | 1.0 | 99.65% | 1,645 | 39.51% | 11.2x |
| V117 | 1.0 | 99.88% | 578 | 31.14% | 8.9x |
| V112 | 1.0 | 99.49% | 2,431 | 29.25% | 8.3x |
| V119 | 1.0 | 99.87% | 638 | 28.68% | 8.2x |
| V108 | 1.0 | 99.52% | 2,283 | 28.03% | 8.0x |
| V118, V114, V110, V120, V122, V121 | 1.0 | 99.1 to 99.9% | 759 to 4,163 | 7.3 to 22.7% | 2.1 to 6.5x |
| id_24, id_08, id_07, id_21, id_27, id_23, id_22, id_25, id_26 | blank | 99.10 to 99.17% | 3,914 to 4,273 | 7.80 to 8.05% | ~2.2x |
| C3 | 0.0 | 99.60% | 1,872 | **0.053%** | **0.015x** |

V111 would have been deleted by any blanket "drop 99% constant" rule, yet nearly half its 1,370 rare rows are fraud. C3 only survived because the rescue triggers in both directions, on rare values that are unusually **safe** as well as unusually risky. The nine identity columns were flagged and then rescued, so my Step 3 prediction that they would be dropped was wrong.

V107 (189 rare rows) and V305 (16) were dropped for having too few rare rows to judge, which is the right reason.

**The split**

| Portion | Rows | Frauds | Fraud rate | Period |
|---------|------|--------|------------|--------|
| train | 472,432 | 16,599 | 3.5135% | 2017-12-01 to 2018-04-20 |
| valid | 118,108 | 4,064 | 3.4409% | 2018-04-20 to 2018-05-31 |

Boundary at TransactionDT 12,192,854. The validation window is about 41 days.

### 5.6 The uid problem, found in Step 3, addressed in Step 4

Exactly **6 of 284 features** have a train-to-test missingness gap above 20 points, and all six are the uid aggregates:

| Feature | Missing in train | Missing in test | Gap |
|---------|------------------|-----------------|-----|
| `TransactionAmt_mean_by_uid` | 11.30% | 81.94% | 70.6 |
| `TransactionAmt_ratio_to_uid_mean` | 11.30% | 81.94% | 70.6 |
| `D15_mean_by_uid` | 20.63% | 82.20% | 61.6 |
| `TransactionAmt_std_by_uid` | 30.54% | 84.96% | 54.4 |
| `D15_std_by_uid` | 36.42% | 85.23% | 48.8 |
| `D15_ratio_to_uid_mean` | 39.72% | 84.01% | 44.3 |

The seventh largest gap is 5.96 points, so the problem is sharply bounded to one family. Cause: 82% of test rows carry a uid fingerprint that never appeared in the training portion, because test runs six months later and most customers are new or their fingerprint shifted.

Step 4 addresses this with a pre-registered ablation (D-36): retrain the winner without those six, and drop them if the cost is under 0.005 PR-AUC.

Features moving the other way, becoming **less** missing in test, are the identity-linked ones such as D13 and the V35 to V52 block, at about 13 to 14 points. That is the 24.4% to 28.0% identity coverage rise showing up in the feature table.

---

## 6. Current repository structure

```
ieee-cis-fraud-detection/
│
├── .dvc/ , .dvcignore                  # DVC config                (Step 3)
├── .github/workflows/                  # empty                     (Step 5)
├── .vscode/settings.json
├── app/                                # empty                     (Step 7)
│
├── config/
│   ├── __init__.py
│   └── config.py                       # extended in Steps 2, 3, 4
│
├── data/
│   ├── raw/                            # git-ignored, 5 CSVs, 1.29 GB
│   ├── interim/                        # git-ignored
│   │   ├── train_joined.parquet        # 590,540 x 435,  80.3 MB
│   │   └── test_joined.parquet         # 506,691 x 434,  69.8 MB
│   ├── processed/                      # DVC-tracked
│   │   ├── train_features.parquet      # 84.4 MB
│   │   ├── test_features.parquet       # 68.5 MB
│   │   ├── kaggle_submission.csv                                   # Step 4
│   │   └── *.dvc pointer files         # committed to git
│   └── external/                       # empty
│
├── docker/                             # empty                     (Step 6)
│
├── docs/
│   ├── PROJECT_STATE.md                # this file
│   ├── steps/step1.md ... step4.md
│   └── decisions/                      # empty
│
├── models/                             # git-ignored except metadata
│   ├── feature_engineer.joblib         # 28.0 MB                   (Step 3)
│   ├── final_model.joblib                                          # Step 4
│   └── final_model_metadata.json       # committed                 (Step 4)
│
├── notebooks/                          # empty
│
├── reports/
│   ├── data_inventory.md                                           # Step 1
│   ├── eda_summary.md                                              # Step 2
│   ├── column_profile.csv , missing_profile.csv                    # Step 2
│   ├── v_column_missing_groups.csv                                 # Step 2
│   ├── feature_summary.md , feature_manifest.csv                   # Step 3
│   ├── dropped_columns.csv , v_column_reduction.csv                # Step 3
│   ├── training_summary.md , model_comparison.csv                  # Step 4
│   ├── threshold_analysis.csv , cost_curve.csv                     # Step 4
│   ├── cv_results.csv , feature_importance.csv                     # Step 4
│   ├── figures/                        # 15 PNG charts
│   └── explainability/                 # 3 SHAP charts             (Step 4)
│
├── scripts/
│   ├── download_data.py
│   └── verify_data.py
│
├── src/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py                                             # Step 3
│   ├── models/
│   │   ├── __init__.py
│   │   └── candidates.py                                           # Step 4
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── ingestion.py , eda.py                                   # Step 2
│   │   ├── features.py                                             # Step 3
│   │   └── training.py                                             # Step 4
│   ├── serving/__init__.py             # modules added             (Step 6)
│   ├── monitoring/__init__.py          # modules added             (Step 5)
│   └── utils/
│       ├── __init__.py
│       ├── memory_utils.py , ingestion_utils.py , eda_utils.py     # Step 2
│       ├── column_selection.py , feature_utils.py                  # Step 3
│       └── metrics.py , mlflow_utils.py , model_plots.py           # Step 4
│
├── tests/__init__.py                   # tests added               (Step 5)
│
├── .env.example , .gitignore , LICENSE , README.md
├── requirements.txt , requirements-dev.txt , requirements.lock.txt
├── mlflow.db                           # git-ignored               (Step 4)
└── run.py                              # ingestion, eda, features, training
```

---

## 7. Files and what each one does

### Step 1

| File | Purpose |
|------|---------|
| `.gitignore` | Blocks raw and interim data, models, secrets, `.venv`, MLflow artifacts, CatBoost logs, editor noise. The `data/processed/*` lines were removed in Step 3 so DVC could manage that folder. |
| `.env.example` | Template listing needed secrets, with no values. |
| `.vscode/settings.json` | Points VS Code at `.venv`, enables pytest, sets the project root as an import path. |
| `README.md` | Rewritten in Step 3, results filled in Step 4. |
| `requirements*.txt` | Runtime deps, dev tooling, and the exact-version lock file. |
| `scripts/download_data.py` | Kaggle CLI download with fallback flag syntax, extraction, size reporting. |
| `scripts/verify_data.py` | Five integrity checks. Uses `usecols` so it never loads all 394 columns. |

### Step 2

| File | Purpose |
|------|---------|
| `src/utils/memory_utils.py` | `optimise_dtypes` with a `PROTECTED_DTYPES` map for the four columns where shrinking would corrupt data. |
| `src/utils/ingestion_utils.py` | Load, standardise `id-` to `id_`, mark, left join with `validate="one_to_one"`, validate, save Parquet. |
| `src/pipelines/ingestion.py` | One code path for train and test, with `SPLIT_SETTINGS` holding expected shapes. |
| `src/utils/eda_utils.py` | Family assignment, column profiling, `missing_pattern_groups` (md5 over each column's blank mask), category fraud rates, ten chart functions. |
| `src/pipelines/eda.py` | Runs every analysis, writes four reports and ten charts. Patched in Step 3 to carry the `ProductCD` caveat. |
| `run.py` | Entry point. `--step ingestion\|eda\|features\|training\|all`, plus `--split`, `--nrows`, `--quick`, `--models`. |

### Step 3

| File | Purpose |
|------|---------|
| `src/utils/column_selection.py` | `top_value_share`, `find_constant_columns`, `assess_near_constant_columns` (the two-directional rescue rule), `load_v_groups`, `cluster_by_correlation`, `choose_representative`, `reduce_v_columns`. |
| `src/utils/feature_utils.py` | `as_label_series` (the single place numbers become text, so train and test cannot diverge), time, amount, email, screen, device, match, and uid builders. |
| `src/features/engineer.py` | `FraudFeatureEngineer`. `_transform_frame` is shared by `fit` and `transform` so they cannot drift. The feature list is fixed by running a real transform during `fit`. Concat warning fixed in Step 4. |
| `src/pipelines/features.py` | Load, split by time, fit on the training portion only, transform both, four verification checks, save two Parquet files, the joblib transformer, four reports. |

### Step 4

| File | Purpose |
|------|---------|
| `config/config.py` | Now also holds the five cost assumptions, MLflow registry names, training settings, `UID_FEATURE_MARKERS`, `UID_ABLATION_TOLERANCE`, and the `.as_posix()` tracking URI fix. |
| `src/utils/metrics.py` | `ranking_metrics`, `review_rate_metrics`, `cost_curve` (exact, via cumulative sums over score-sorted rows), `best_operating_point`, `evaluate`, `downsample_curve`. |
| `src/utils/mlflow_utils.py` | `configure_mlflow`, `log_model_compatibly` (inspects the signature to handle the MLflow 3 `name` versus `artifact_path` change), safe param and metric logging. |
| `src/utils/model_plots.py` | Five charts: model comparison, precision-recall curves, cost curve, score distribution, CV stability. |
| `src/models/candidates.py` | `Candidate` dataclass, per-library fit adapters, `build_candidates`, `rebuild_for_refit`, `expanding_window_splits`. |
| `src/pipelines/training.py` | Eight phases: load and split, train candidates, pick a winner, run the uid ablation, cross-validate, threshold and cost analysis, SHAP, retrain and register and score the test set. |

---

## 8. Environment

| Item | Value |
|------|-------|
| Python | **3.11.9** |
| Environment | `.venv` in the project root, git-ignored |
| Activate | `.\.venv\Scripts\Activate.ps1` |
| If blocked | `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` |
| Rebuild exactly | `pip install -r requirements.lock.txt` |

### 8.1 Confirmed library versions

| Library | Version | Notes that shaped the code |
|---------|---------|----------------------------|
| pandas | 2.3.3 | `observed=True` on every category groupby. Empty-frame concat is deprecated, fixed in Step 4. |
| numpy | 2.4.6 | numpy 2.x, so `np.NaN` and `np.float_` do not exist |
| pyarrow | 24.0.0 | Parquet engine, preserves category dtypes |
| scipy | 1.17.1 | |
| scikit-learn | 1.9.0 | `BaseEstimator`, `TransformerMixin`, `Pipeline`, metrics |
| lightgbm | 4.7.0 | Early stopping via callbacks, `eval_metric="average_precision"` |
| xgboost | 3.2.0 | Early stopping in the constructor, `eval_metric="aucpr"` |
| catboost | 1.2.10 | Early stopping in `fit`, `eval_metric="PRAUC"`, `allow_writing_files=False` |
| imbalanced-learn | 0.14.2 | Installed but deliberately unused, see D-38 |
| mlflow | 3.15.1 | **MLflow 3.** Aliases not stages; `log_model` uses `name` not `artifact_path`; the URI needs forward slashes |
| shap | 0.51.0 | `TreeExplainer`, beeswarm, bar, waterfall |
| matplotlib | 3.11.1 | `plt.cm.get_cmap` removed in 3.9, so unused |
| seaborn | 0.13.2 | |
| plotly | 6.9.0 | Step 7 |
| fastapi | 0.141.1 | Step 6 |
| uvicorn | 0.52.3 | Step 6 |
| streamlit | 1.61.1 | Step 7 |
| joblib | 1.5.3 | Saves the transformer and the model |
| pytest | 9.1.1 | Step 5 |
| ruff | 0.16.3 | Step 5 |
| black | 26.5.1 | Step 5 |
| pre-commit | 4.6.2 | Step 5 |
| kaggle | 2.2.4 | Positional competition argument |
| dvc | 3.55 or newer | Added in Step 3, verified working |

---

## 9. The cost model

Introduced in Step 4 (D-34, D-35). **These are stated assumptions, not figures from a business.** All five live in `config/config.py`.

| Assumption | Value | Reasoning |
|------------|-------|-----------|
| Analyst review | $4.00 per case | Fully loaded analyst at about $60k a year is roughly $29 an hour; a five minute review is $2.40; rounded up for supervision and customer calls |
| Chargeback fee | $25.00 per missed fraud | Card networks charge $15 to $40 per dispute on top of the clawback |
| False alarm friction | $1.00 | Expected value of holding and releasing a legitimate customer. The softest number and the first to replace |
| Fraud recovered when caught | 90% | Reviews take time, some are judged wrongly, some transactions have settled |
| Review capacity | 2% of transactions | About one in fifty, roughly one analyst's full shift at this volume |

**The arithmetic.** For a given threshold: missed fraud costs the amount plus the fee; caught fraud costs a review plus the 10% not recovered; a false alarm costs a review plus friction; a correct pass costs nothing. The comparison point is doing nothing, where every fraud is missed.

Costs are weighted by the real transaction amount, so the optimal threshold naturally leans towards catching expensive fraud. A flat per-fraud penalty cannot express that.

The curve is computed exactly at every possible threshold using cumulative sums over score-sorted rows, not sampled on a grid.

**Framing for the PM track:** present the annual saving as an order of magnitude under stated assumptions, with sensitivity to each one, never as a forecast.

---

## 10. Conventions in force

**Code**
- All paths come from `config/config.py`. No module builds its own.
- One random seed, `RANDOM_SEED = 42`.
- Each pipeline stage reads a file and writes a file.
- `src/pipelines/` orders stages, `src/features/` builds features, `src/models/` defines candidates, `src/utils/` supports all of them.
- Anything learned from data is learned from training rows only, then applied unchanged.
- Every groupby over a category column passes `observed=True`.
- Matplotlib uses `Agg`, set before pyplot is imported.
- Decision rules that use results are written down before the results are seen.

**Git**
- Branch naming `step-NN-short-description`; commits `type: message`; squash-merge by pull request; tag `v0.N.0-stepN`
- `main` must always work

**Documentation**
- No em dashes; plain vocabulary; explanation before every code block
- Every file created is stated with full contents and its reason
- When an earlier claim turns out to be wrong, it is corrected openly in the next step rather than quietly amended. So far: the Parquet size estimate, the memory reduction range, the 3.75x identity finding, and the prediction that the nine `id_` columns would be dropped.

---

## 11. Completed

### Steps 1 to 3, verified
- [x] Data downloaded and verified, repo live, environment locked, tagged `v0.1.0-step1`
- [x] Ingestion and EDA run, 15 V blocks found, ten charts, tagged `v0.2.0-step2`
- [x] Project moved offline, `.venv` rebuilt, verification re-run
- [x] Feature stage run, 284 features, all four checks passed
- [x] DVC set up with a local remote; `dvc pull` restored a deleted file
- [x] README rewritten as a standalone document
- [x] `ProductCD` confound identified, quantified, and patched into the EDA report generator
- [x] Tagged `v0.3.0-step3`

### Step 4, delivered
- [x] Cost model designed with five documented assumptions
- [x] `config/config.py` extended; MLflow URI fixed with `.as_posix()`
- [x] `src/features/engineer.py` concat warning fixed
- [x] `src/utils/metrics.py`, `mlflow_utils.py`, `model_plots.py` created
- [x] `src/models/` package and `candidates.py` created
- [x] `src/pipelines/training.py` created
- [x] `run.py` updated with `training`, `--quick`, `--models`
- [x] Q-04, Q-05, Q-12 answered as D-44, D-46, D-45
- [ ] Training run completed by the user
- [ ] Kaggle late submission uploaded and scored
- [ ] README results filled in
- [ ] Branch merged and tagged `v0.4.0-step4`

---

## 12. Pending

**Immediately next (Step 5)**
- pytest suite: metrics and cost model unit tests, a joblib round-trip test for the transformer, and a test that fails if leakage is reintroduced
- GitHub Actions running tests, ruff, and black on every pull request, with a README badge
- `pre-commit` hooks
- MLflow registry used properly: promoting `candidate` to `production` by moving an alias, and the conditions required before a promotion
- Drift monitoring built on the two real shifts already found: identity coverage 24.4% to 28.0%, and the uid family 11% to 82% missing
- Population Stability Index and Kolmogorov-Smirnov, explained from scratch, with meaningful thresholds
- Scoring the test set month by month to show whether performance decays with distance from training
- A retraining trigger: the firing condition and what happens next
- Monitoring outputs written in the shape the Step 7 dashboard needs, per D-33 and D-45

**Later**
- Step 6: Dockerfile, FastAPI service loading the transformer and the model, deployment to Hugging Face Spaces per D-44, artifacts served from the HF Model Hub
- Step 7: Streamlit dashboard per D-33 and D-45, architecture diagram, final README, portfolio packaging

---

## 13. Open questions

| # | Question | Needed by | Status |
|---|----------|-----------|--------|
| Q-01 | Library versions | Step 2 | **Answered.** Section 8.1. |
| Q-02 | RAM | Step 2 | **Answered.** 32 GB. |
| Q-03 | DVC remote | Step 3 | **Answered.** Local folder at `C:\Users\Dauda Agbonoga\dvcstore`. D-32. |
| Q-04 | Deployment target | Step 6 | **Answered.** Hugging Face Spaces, Docker SDK. D-44. Render is the backup. |
| Q-05 | Kaggle late submission | Step 4 | **Answered.** Yes, it is free. D-46. |
| Q-06 | Streamlit or React | Step 7 | **Answered.** Streamlit. |
| Q-07 | Business cost figures | Step 4 | **Answered by construction.** No real figures exist, so a five-assumption model was built and documented. See Section 9. Replace any assumption with a real number when one becomes available. |
| Q-08 | Project location | Step 3 | **Answered.** Moved offline. |
| Q-09 | Folder rename | Step 3 | **Answered.** Done. |
| Q-10 | Python patch version | Step 3 | **Answered.** 3.11.9. |
| Q-11 | V block structure | Step 3 | **Answered.** 15 blocks, Section 5.4. |
| Q-12 | Dashboard audience | Step 7 | **Answered.** Portfolio and hiring managers. D-45. |
| Q-13 | Does the uid ablation drop or keep the six features? | Step 5 | Open until the Step 4 run finishes. The pre-registered rule (D-36) decides it, not judgement after the fact. Either outcome changes what Step 5 monitors. |
| Q-14 | Which model wins, and does `has_identity` rank low as D-31 predicts? | Step 5 | Open until the run finishes. If it ranks high, the D-31 analysis was wrong and needs revisiting. |
| Q-15 | Does a Hugging Face account exist, and is the CLI set up? | Step 6 | Open. Free to create. Needed before deployment, not before Step 5. |

---

## 14. How to resume from nothing

```powershell
# 1. Clone and enter
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

# 2. Recreate the environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt

# 3. Rebuild the raw data (needs a Kaggle account that joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# 4. Rebuild everything downstream
python run.py --step all

# 5. Or, if the DVC remote survived, pull the processed data directly
dvc pull

# 6. Read the current state
code docs/PROJECT_STATE.md
```

If the repository is also lost, `docs/steps/step1.md` through `step4.md` rebuild everything from scratch.

---

## 15. Glossary

| Term | Plain meaning |
|------|---------------|
| Parquet | A file format storing tables column by column. Smaller and faster than CSV, and it remembers data types |
| Left join | Keep every row from the left table, attach matching data from the right where it exists, leave blanks otherwise |
| Category dtype | Stores each distinct text value once, plus a small number per row pointing at it |
| Class imbalance | When one outcome is far rarer than the other, here 3.5% fraud |
| Accuracy | Share of predictions correct. Useless here: always predicting "not fraud" scores 96.5% |
| Precision | Of the transactions you flagged, the share that really were fraud |
| Recall | Of all the fraud that occurred, the share you caught |
| PR-AUC | Precision-Recall Area Under Curve. Primary metric. Baseline equals the fraud rate |
| ROC-AUC | Probability a random fraud scores above a random legitimate transaction. Baseline 0.5. The Kaggle metric |
| Time-based split | Train on earlier data, validate on later data. Imitates predicting the future |
| Data leakage | Information unavailable at prediction time influencing training, producing a score you cannot reproduce in production |
| Confounded comparison | A difference between two groups that is really a difference in what those groups contain. See D-31 |
| Frequency encoding | Replacing a category with how often it appeared in training. Turns rarity into a number |
| Aggregate feature | A row's value compared against the average for its group |
| Fitted transformer | An object that learns from training data, stores what it learned, and applies it later |
| Training and serving skew | Model trained on one set of transformations and fed another in production. Nothing errors; the predictions are just wrong |
| Ablation | Removing one part of a system on purpose to measure what it was contributing |
| Pre-registered decision | A rule written down before the result is seen, so the conclusion cannot be fitted to the data |
| Early stopping | Halting training when the validation metric stops improving, to avoid memorising the training set |
| Expanding-window CV | Folds where each trains on more history than the last and is scored on the period straight after |
| Class weighting | Telling the model rare cases count for more. Deliberately not used here, see D-38 |
| MLflow run | One training attempt with its settings, metrics, and files recorded |
| MLflow alias | A movable pointer to a model version, such as `candidate` or `production`. Replaces the deprecated stages |
| SHAP | Explains how much each feature pushed one prediction away from the average |
| DVC | Versions large data files alongside code, keeping a fingerprint in Git and the data elsewhere |
| Drift | When live data slowly stops resembling training data, so the model quietly gets worse |
| PSI | Population Stability Index. Measures how far a distribution has moved. Step 5 |
| Model registry | A catalogue of trained model versions with a record of which is live |
| CI/CD | Automated checks and deployment on every code change |

---

*End of PROJECT_STATE.md. Next: Step 5, the MLOps layer.*
