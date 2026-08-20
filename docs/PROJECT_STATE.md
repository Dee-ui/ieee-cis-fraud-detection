# PROJECT_STATE.md

**Last updated:** End of Step 3 of 7
**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Local path:** `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`

---

## 0. What this document is

This is the anchor for the whole project. It is rewritten in full at the end of every step, never patched with a diff.

If earlier conversation is lost, this file alone is enough to pick up exactly where we stopped. It records every decision made, the current state of the repository, what is finished, what is outstanding, and what questions are still open.

---

## 1. Project at a glance

| Item | Value |
|------|-------|
| Goal | A complete, portfolio-grade fraud detection system covering the full machine learning and MLOps lifecycle |
| Dataset | IEEE-CIS Fraud Detection (Kaggle, data provided by Vesta Corporation) |
| Scope | Data pipeline, feature engineering, model training with experiment tracking, CI/CD, model registry, drift monitoring, Docker, deployment, dashboard |
| Delivery format | 7 steps, one per conversation message, each with its own markdown guide plus a refreshed copy of this file |
| Platform | Windows, VS Code, PowerShell, Python 3.11.9 |
| Machine | Intel Core Ultra 7 265H, 32 GB RAM. Enough to hold the full joined table in memory, so no chunked reading is needed anywhere |
| Local path | `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`, offline and outside any sync folder |
| Version control | Git, public GitHub repository, plus DVC for processed data |
| Tracks | Technical track (this work) and a separate PM track walkthrough that happens afterwards with the project manager. Documentation quality matters for both. |

---

## 2. Why this dataset

Card fraud is a rare-event problem, which makes it a genuinely hard machine learning problem rather than a tutorial one. It also has a real cost structure on both sides: a missed fraud is a direct loss, and a false alarm blocks a paying customer. That combination makes it an unusually good subject for an end-to-end project, because every technical decision has a business consequence you can point at.

The IEEE-CIS dataset in particular has four properties that make it worth the effort:

- **Enough positive cases to learn from.** 20,663 fraudulent transactions out of 590,540, a rate of 3.4990%. Rare enough to be realistic, common enough that a model has something to work with.
- **Two joinable tables.** Transaction data and identity data, linked on `TransactionID`, with only partial coverage. That is real data engineering rather than a single flat file.
- **A mix of named and anonymised features.** 339 of the columns are engineered features supplied by Vesta with no published meaning. You cannot reason your way through them, so you have to find structure empirically, which is what actually happens in industry.
- **A test set 30 days in the future.** The competition split by time on purpose. That makes the time-based validation lesson unavoidable rather than optional, and it gives Step 5 a genuine distribution shift to detect rather than one that has to be manufactured.

---

## 3. The 7-step plan and current status

| Step | Content | Status |
|------|---------|--------|
| 1 | Dataset acquisition, folder scaffold, GitHub repo, Python environment | **Complete, verified** |
| 2 | EDA and data understanding: table joins, feature groups, imbalance profiling | **Complete, verified** |
| 3 | Feature engineering and preprocessing pipeline | **Delivered, awaiting the user's run** |
| 4 | Model training with MLflow experiment tracking | Not started |
| 5 | MLOps layer: CI/CD, testing, model registry, drift monitoring | Not started |
| 6 | Dockerisation and deployment | Not started |
| 7 | Advanced dashboard and final documentation or portfolio packaging | Not started |

---

## 4. Decision log

### Step 1

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | Dataset is IEEE-CIS Fraud Detection | 590,540 rows at 3.4990% fraud gives 20,663 positive cases, plus a joinable second table and a time-separated test set. |
| D-02 | Repository named `ieee-cis-fraud-detection` | Descriptive and scannable for reviewers. The local folder now matches, as of Step 3. |
| D-03 | Python 3.11 | Required minimum for the current Kaggle CLI, and has stable prebuilt Windows packages for LightGBM, XGBoost, CatBoost, and SHAP. Confirmed as 3.11.9. |
| D-04 | `venv` plus `requirements.txt`, not conda | One dependency format that Docker (Step 6) and GitHub Actions (Step 5) both consume natively. |
| D-05 | Raw data is never committed to Git | Roughly 1.3 GB of CSV. GitHub rejects files over 100 MB. Reproducibility comes from `scripts/download_data.py` instead. |
| D-06 | DVC deferred to Step 3 | Resolved. See D-32. |
| D-07 | Paths resolved dynamically in `config/config.py` | `Path(__file__).resolve().parents[1]` works on any machine, inside Docker, and in CI. Proven twice: once when the folder name did not match the repository, and again when the whole project was moved in Step 3 with zero code changes. |
| D-08 | Branch per step, merged into `main` by pull request, tagged after merge | Gives a reviewable trail, triggers CI in Step 5, and produces a clean narrative for the PM walkthrough. |
| D-09 | Public GitHub repository | It is a portfolio piece. Public also gives free Actions minutes and branch rulesets. |
| D-10 | Dependencies split into `requirements.txt` and `requirements-dev.txt` | Runtime needs stay separate from development tooling, so the Docker image in Step 6 stays lean. `requirements.lock.txt` gives exact reproducibility. |
| D-11 | Download script shells out to the Kaggle CLI rather than importing the Kaggle Python library | The library's internal interface changes across versions; the CLI is the documented, stable contract. |
| D-12 | Interim and processed data stored as Parquet, not CSV | Far smaller, much faster to load, and it preserves data types, which CSV loses entirely. |
| D-13 | Notebooks are for exploration only; anything that matters is rewritten as a module in `src/` | Notebooks are not testable, not importable, and not reviewable in diffs. |

### Step 2

| ID | Decision | Rationale |
|----|----------|-----------|
| D-14 | `run.py` created in Step 2 rather than Step 3 | Two runnable stages existed, which is the point where a single entry point stops being overhead and becomes useful. |
| D-15 | Test set joined and saved alongside train, despite having no labels | It becomes the drift monitoring input in Step 5. It starts 30 days after training ends and shows genuine distribution shift, which is far better than perturbing training data artificially. |
| D-16 | Left join transaction to identity, keep blanks as blanks, add a `has_identity` flag | LightGBM, XGBoost, and CatBoost all learn a direction for missing values at every split, so blanks cost nothing. Dropping 41 identity columns would throw away signal; splitting into two models would halve the data each sees. Partially revised by D-31. |
| D-17 | Interim data stored as Parquet with category dtypes preserved | Type information survives a save and load, so the optimisation work is done once. |
| D-18 | `TransactionAmt` stays `float64`. `TransactionID` and `TransactionDT` become `int32`. Everything else shrunk to the smallest exact type | Verified numerically: `float32` stores whole numbers exactly only below 16,777,216, and the test `TransactionDT` reaches 34,214,345, so `float32` would silently round it to 34,214,344. `float32` also turns a `TransactionAmt` of 31937.39 into 31937.390625, and the cents portion is a fraud signal that Step 3 extracts. |
| D-19 | `TransactionDT` displayed against a reference date of 30 November 2017, for readability only | The competition never published a real start date. This convention places the first transaction on 1 December 2017. `TransactionDT` is only ever used as an ordering and an elapsed duration. |
| D-20 | PR-AUC primary, ROC-AUC secondary, recall at a fixed review rate as the business headline. Accuracy not used at all | A do-nothing model scores 96.5% accuracy here. ROC-AUC is insensitive to false positives when the negative class is 569,877 rows. PR-AUC has a meaningful baseline equal to the fraud rate, 0.035. |
| D-21 | Validation is a time-based split, never random. Last 20% of the training period by `TransactionDT` | The real test set is 30 days in the future. A random split leaks three ways here: the same card appears on both sides, `D` columns encode elapsed time from earlier events, and fraud arrives in bursts that a shuffle splits across both sides. |
| D-22 | Feature families assigned by rule, with unmapped columns reported loudly | With 435 columns, a silently miscategorised column is easy to miss. The run reported zero unmapped columns. |

### Step 3

| ID | Decision | Rationale |
|----|----------|-----------|
| D-23 | Feature engineering is a fitted object saved with joblib, not a script that edits data in place | In Step 6 a single transaction arrives at a web service. Frequency counts, category codes, and group averages cannot be worked out from one row, so they must have been saved. A script cannot be loaded; an object can. Implemented as a scikit-learn `BaseEstimator` and `TransformerMixin` so it drops into a `Pipeline`. |
| D-24 | The transformer is fitted only on the first 80% of the training period, never on the validation portion or on test | Counting card frequencies across validation rows means the number attached to each validation row was partly computed from that row. The validation score then flatters the model. Costs a little final accuracy, buys a trustworthy number. |
| D-25 | Encodings learned from training rows only, never from train and test combined | Combining them is common in competition write-ups because the test set is available. It is not available in production, where you score one transaction with no knowledge of future ones. |
| D-26 | `TransactionDT`, `TransactionID`, and any absolute day counter are excluded from the feature set | Test `TransactionDT` values sit entirely above training values, with no overlap. Tree models cannot split outside the range they were trained on, so such a column looks useful in training and does nothing afterwards. Only cyclical time features (hour, day of week) survive. |
| D-27 | Columns where one value covers 99% or more of rows are dropped, with a rescue rule | A column that is 99.2% blank carries almost nothing, but "almost" is not "nothing" when the target occurs 3.5% of the time. Each candidate is checked against the fraud rate among its rare values and rescued if there are at least 500 such rows and the lift is at least 2x in either direction. Every decision is written to `reports/dropped_columns.csv` with its evidence. |
| D-28 | V columns reduced by correlation clustering inside each of the 15 blocks, at a threshold of 0.75, keeping the column with the most distinct values per cluster | The blocks identify columns from a shared source; correlation identifies near-duplicates within them. Correlation is computed only on rows where the block is present. Absolute correlation is used, because a perfectly inverted column carries the same information and a tree can flip a sign for free. |
| D-29 | The `uid` customer fingerprint is used only for grouping and counting, never as a feature in its own right | Given the fingerprint directly, the model memorises individual customers. That scores well in validation, where the same customers appear on both sides, and is worth nothing on customers it has never met. |
| D-30 | Every text column becomes an integer with a stored mapping. Blank gets its own code. Unseen values map to -1 | Without a stored mapping the same word gets a different number next month and the model silently breaks. Blank is treated as real information rather than a hole. |
| D-31 | `has_identity` is kept, but the confound behind it is recorded | The headline "fraud is 3.75x as likely with an identity record" is mostly a `ProductCD` effect: product W never has an identity record and has the lowest fraud rate, while every other product almost always does. Within non-W products the lift is 1.39x. The flag is one column and costs nothing, but is expected to rank low. If it ranks high in Step 4, that is a warning sign. |
| D-32 | DVC introduced in Step 3 with a local folder remote at `C:\Users\Dauda Agbonoga\dvcstore` | Closes Q-03. A local folder needs no account, no internet, and no cost, which suits a deliberately offline machine. Swapping to cloud storage later is one command and changes nothing else. Required removing `data/processed/*` from `.gitignore`, since DVC refuses to track already-ignored paths and manages its own ignore rules. |
| D-33 | The Streamlit dashboard will draw from two distinct sources and use precomputed artifacts, not the raw tables | Three areas: a static model profile built at training time, a live operations view from scored data, and a drift view that deliberately overlays the two. Charts read small precomputed aggregate files so pages load fast. Detail in `docs/steps/step3.md` Section 20. |

---

## 5. Verified results so far

### 5.1 Raw data, verified in Step 1

| File | Size | Rows | Columns |
|------|------|------|---------|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 |
| `sample_submission.csv` | 5.8 MB | 506,691 | 2 |

- Fraud rate **3.4990%**: 20,663 fraudulent out of 590,540, roughly 1 in 28
- `TransactionID` unique in both training tables, zero duplicates
- 144,233 transactions have an identity record, **24.4%**
- `train_identity` uses `id_` prefixes, `test_identity` uses `id-`

### 5.2 Ingestion, verified in Step 2

| Split | Rows | Columns | Memory before | Memory after | Reduction | Parquet |
|-------|------|---------|---------------|--------------|-----------|---------|
| train | 590,540 | 435 | 2,567.7 MB | 927.2 MB | 63.9% | 80.3 MB |
| test | 506,691 | 434 | 2,214.5 MB | 795.2 MB | 64.1% | 69.8 MB |

Train renamed zero identity columns, test renamed 38. Runtime 3 minutes 7 seconds. Column type breakdown for train: 398 `float32`, 31 `category`, 2 `int8`, 2 `int32`, 1 `float64`, 1 `int16`.

Note: the Parquet files came out roughly four times smaller than estimated. The table is unusually compressible because most columns are `float32` with a high share of blanks, which Parquet stores as a compact bitmap, and the remaining values sit in long repeated runs that compress well.

### 5.3 EDA, verified in Step 2

**Time coverage**

| Split | First | Last | Span |
|-------|-------|------|------|
| train | 2017-12-01 | 2018-05-31 | 182.0 days |
| test | 2018-07-01 | 2018-12-30 | 183.0 days |

Gap between them: **30.0 days**.

**Identity coverage**

| Group | Transactions | Fraud rate |
|-------|--------------|------------|
| No identity record | 446,307 | 2.0939% |
| Has identity record | 144,233 | 7.8470% |

**The 3.75x figure is confounded and must not be quoted without the caveat.** Identity coverage is almost entirely decided by `ProductCD`: W is 0% covered, C is 90.8%, and H, R, and S are each 99.6%. W also has the lowest fraud rate (2.04%) and makes up 439,670 of the 590,540 transactions. Restricted to non-W products, where the flag actually varies, the comparison is 7.85% (has identity, 144,233 rows) against 5.67% (no identity, 6,637 rows), a lift of **1.39x**. See D-31.

Test identity coverage is **28.0%** against training's 24.4%, a genuine 3.6 point shift across the 30 day gap. This is the worked drift example for Step 5.

**Fraud rate by category**

| Column | Highest | Lowest |
|--------|---------|--------|
| ProductCD | C at 11.69% | W at 2.04% |
| card4 | discover at 7.73% | american express at 2.87% |
| card6 | credit at 6.68% | debit at 2.43% |
| DeviceType | mobile at 10.17% | missing at 2.10% |
| P_emaildomain | mail.com at 18.96% | aol.com at 2.18% |

**Missing data.** 53 columns have no missing values at all. 12 columns are more than 90% missing. The nine emptiest are identity columns between 99.12% and 99.20% missing. `dist2` is 93.63% and `D7` is 93.41%. The C family has zero missing across all 14 columns.

**Feature families in the joined table**

| Family | Columns | Mean missing |
|--------|---------|--------------|
| vesta_V | 339 | 43.04% |
| identity_id | 38 | 84.82% |
| timedelta_D | 15 | 58.15% |
| counting_C | 14 | 0.00% |
| match_M | 9 | 49.92% |
| card | 6 | 0.51% |
| address, device, distance, email | 2 each | 11.13%, 78.04%, 76.64%, 46.37% |
| amount, product, identifier, time, target, engineered | 1 each | 0.00% |

### 5.4 The V column block structure, confirmed

All 339 V columns fall into 15 blocks that share an identical missing pattern.

| Block | Columns | Missing | Number range | Shape |
|-------|---------|---------|--------------|-------|
| 1 | 46 | 77.91% | V217 to V278 | interleaved |
| 2 | 43 | 0.05% | V95 to V137 | solid run |
| 3 | 32 | 0.00% | V279 to V321 | interleaved |
| 4 | 31 | 76.36% | V167 to V216 | interleaved |
| 5 | 23 | 12.88% | V12 to V34 | solid run |
| 6 | 22 | 13.06% | V53 to V74 | solid run |
| 7 | 20 | 15.10% | V75 to V94 | solid run |
| 8 | 19 | 76.32% | V169 to V210 | interleaved |
| 9 | 18 | 28.61% | V35 to V52 | solid run |
| 10 | 18 | 86.12% | V138 to V163 | interleaved |
| 11 | 18 | 86.05% | V322 to V339 | solid run |
| 12 | 16 | 76.05% | V220 to V272 | interleaved |
| 13 | 11 | 47.29% | V1 to V11 | solid run |
| 14 | 11 | 86.12% | V143 to V166 | interleaved |
| 15 | 11 | 0.21% | V281 to V315 | interleaved |

Two structural facts that matter:

**Blocks 10 and 14 both sit at 86.12% missing but are different blocks.** Same number of blanks, different rows. Grouping by missing count rather than by missing pattern would have merged them. The md5 hashing in `missing_pattern_groups` compares the actual pattern, which is why it separated them.

**Eight of the fifteen blocks are interleaved**, meaning their V numbers weave through each other rather than sitting in clean runs. Blocks 4 and 8 are threaded together across V167 to V216; blocks 10 and 14 across V138 to V166. Reducing the V columns by chopping the number range, which is the intuitive approach, would cut across the real groupings.

---

## 6. Current repository structure

Folders marked with a step number exist but are empty, waiting for that step.

```
ieee-cis-fraud-detection/
│
├── .dvc/                               # DVC config                (Step 3)
├── .dvcignore                                                      # Step 3
│
├── .github/
│   └── workflows/                      # empty                     (Step 5)
│
├── .vscode/
│   └── settings.json
│
├── app/                                # empty                     (Step 7)
│
├── config/
│   ├── __init__.py
│   └── config.py                       # extended in Steps 2 and 3
│
├── data/
│   ├── raw/                            # git-ignored, 5 CSVs, 1.29 GB
│   ├── interim/                        # git-ignored
│   │   ├── train_joined.parquet        # 590,540 x 435,  80.3 MB
│   │   └── test_joined.parquet         # 506,691 x 434,  69.8 MB
│   ├── processed/                      # DVC-tracked               (Step 3)
│   │   ├── train_features.parquet
│   │   ├── test_features.parquet
│   │   ├── train_features.parquet.dvc  # committed to git
│   │   └── test_features.parquet.dvc   # committed to git
│   └── external/                       # empty
│
├── docker/                             # empty                     (Step 6)
│
├── docs/
│   ├── PROJECT_STATE.md                # this file
│   ├── steps/
│   │   ├── step1.md
│   │   ├── step2.md
│   │   └── step3.md
│   └── decisions/                      # empty
│
├── models/                             # git-ignored
│   └── feature_engineer.joblib                                     # Step 3
│
├── notebooks/                          # empty
│
├── reports/
│   ├── data_inventory.md                                           # Step 1
│   ├── eda_summary.md                                              # Step 2
│   ├── column_profile.csv                                          # Step 2
│   ├── missing_profile.csv                                         # Step 2
│   ├── v_column_missing_groups.csv                                 # Step 2
│   ├── feature_summary.md                                          # Step 3
│   ├── feature_manifest.csv                                        # Step 3
│   ├── dropped_columns.csv                                         # Step 3
│   ├── v_column_reduction.csv                                      # Step 3
│   ├── figures/                        # 10 PNG charts             (Step 2)
│   └── explainability/                 # empty                     (Step 4)
│
├── scripts/
│   ├── download_data.py
│   └── verify_data.py
│
├── src/
│   ├── __init__.py
│   ├── features/                                                   # Step 3
│   │   ├── __init__.py
│   │   └── engineer.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── ingestion.py                                            # Step 2
│   │   ├── eda.py                                                  # Step 2
│   │   └── features.py                                             # Step 3
│   ├── serving/__init__.py             # modules added             (Step 6)
│   ├── monitoring/__init__.py          # modules added             (Step 5)
│   └── utils/
│       ├── __init__.py
│       ├── memory_utils.py                                         # Step 2
│       ├── ingestion_utils.py                                      # Step 2
│       ├── eda_utils.py                                            # Step 2
│       ├── column_selection.py                                     # Step 3
│       └── feature_utils.py                                        # Step 3
│
├── tests/
│   └── __init__.py                     # tests added               (Step 5)
│
├── .env.example
├── .gitignore
├── LICENSE                             # MIT
├── README.md                           # rewritten in Step 3
├── requirements.txt
├── requirements-dev.txt
├── requirements.lock.txt
└── run.py                              # ingestion, eda, features
```

---

## 7. Files and what each one does

### Step 1

| File | Purpose |
|------|---------|
| `.gitignore` | Blocks raw and interim data, models, secrets, the virtual environment, MLflow artifacts, CatBoost logs, and editor noise. The `data/processed/*` lines were removed in Step 3 so DVC could take over that folder. |
| `.env.example` | Template listing which secrets are needed, with no values. Safe to commit. |
| `.vscode/settings.json` | Points VS Code at `.venv`, enables pytest, sets the project root as an import path, formats on save. |
| `README.md` | Rewritten in Step 3. Problem, dataset, findings, approach, quickstart, pipeline table, roadmap, results, tech stack. |
| `requirements.txt` / `requirements-dev.txt` / `requirements.lock.txt` | Runtime deps, dev tooling, and the exact-version lock file that rebuilds an identical environment. |
| `config/config.py` | Extended in Steps 2 and 3. See below. |
| `scripts/download_data.py` | Checks the Kaggle CLI, skips if files exist unless `--force`, downloads via subprocess with a fallback to older flag syntax, extracts, deletes the zip, reports sizes. |
| `scripts/verify_data.py` | Five checks: file presence, row and column counts, fraud rate, key uniqueness and identity coverage, and the `id_` versus `id-` naming difference. Uses `usecols` so it never loads all 394 columns. |

### Step 2

| File | Purpose |
|------|---------|
| `src/utils/memory_utils.py` | `optimise_dtypes` shrinks every column to its smallest safe type, with a `PROTECTED_DTYPES` map for the four columns where shrinking would corrupt data. |
| `src/utils/ingestion_utils.py` | `load_csv`, `standardise_identity_columns`, `add_identity_marker`, `join_transaction_identity` (left join with `validate="one_to_one"`), `validate_join`, `save_parquet`. |
| `src/pipelines/ingestion.py` | Load, standardise, join, validate, optimise, save. Train and test go through one code path with a `SPLIT_SETTINGS` dictionary holding expected shapes. |
| `src/utils/eda_utils.py` | Family assignment by regex, column profiling, `missing_pattern_groups` (md5-hashes each column's blank mask), `fraud_rate_by_category`, `derive_time_frame`, and ten chart functions. Sets the matplotlib `Agg` backend before importing pyplot. |
| `src/pipelines/eda.py` | Runs every analysis, writes four report files and ten charts, and auto-generates `reports/eda_summary.md` from computed results. Patched in Step 3 to carry the `ProductCD` caveat. |
| `run.py` | Single entry point. `--step ingestion\|eda\|features\|all`, `--split train\|test\|both`, `--nrows N`. |

### Step 3

| File | Purpose |
|------|---------|
| `config/config.py` | Now also holds Step 3 output paths, `SPLIT_COLUMN`, `PASSTHROUGH_COLUMNS`, the pruning thresholds (`NEAR_CONSTANT_THRESHOLD`, `RESCUE_MIN_RARE_ROWS`, `RESCUE_MIN_FRAUD_LIFT`, `V_CORRELATION_THRESHOLD`), `MISSING_LABEL`, `UNSEEN_CATEGORY_CODE`, `FREQUENCY_ENCODE_COLUMNS` (18 entries), `AGGREGATION_SPECS` (6 pairs), and the uid source columns. |
| `src/utils/column_selection.py` | `top_value_share`, `find_constant_columns`, `assess_near_constant_columns` (the rescue rule), `load_v_groups`, `cluster_by_correlation`, `choose_representative`, `reduce_v_columns`. |
| `src/utils/feature_utils.py` | `as_label_series` (the single place numbers become text, so train and test cannot diverge), `combine_labels`, `build_time_features`, `build_amount_features`, `split_email_domain`, `build_screen_features`, `first_token`, `build_match_features`, `build_uid`. |
| `src/features/engineer.py` | `FraudFeatureEngineer`, a scikit-learn transformer holding every learned map: frequency shares, category codes, and group aggregates. Its `_transform_frame` is shared by `fit` and `transform`, so the two cannot drift apart. The feature list is fixed by actually running a transform during `fit` rather than being predicted separately. |
| `src/pipelines/features.py` | Load, split by time, fit on the training portion only, transform train and test, run four verification checks, save two Parquet files, the joblib transformer, and four reports. |

---

## 8. Environment

| Item | Value |
|------|-------|
| Python | **3.11.9** |
| Environment | `.venv` in the project root, git-ignored, rebuilt from scratch after the Step 3 move |
| Activate | `.\.venv\Scripts\Activate.ps1` |
| If activation is blocked | `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` |
| Rebuild exactly | `pip install -r requirements.lock.txt` |

### 8.1 Confirmed library versions

| Library | Version | Notes that shaped the code |
|---------|---------|----------------------------|
| pandas | 2.3.3 | `observed=True` passed explicitly on every category groupby |
| numpy | 2.4.6 | numpy 2.x, so `np.NaN` and `np.float_` do not exist |
| pyarrow | 24.0.0 | Parquet engine, preserves category dtypes across save and load |
| scipy | 1.17.1 | |
| scikit-learn | 1.9.0 | `BaseEstimator` and `TransformerMixin` for the feature engineer |
| lightgbm | 4.7.0 | Native missing value handling, which underpins D-16 |
| xgboost | 3.2.0 | Same |
| catboost | 1.2.10 | Same |
| imbalanced-learn | 0.14.2 | Step 4 if resampling is trialled |
| mlflow | 3.15.1 | **MLflow 3, not 2.** Step 4 code must target the 3.x API |
| shap | 0.51.0 | Step 4 |
| matplotlib | 3.11.1 | `plt.cm.get_cmap` removed in 3.9, so no chart code uses it |
| seaborn | 0.13.2 | |
| plotly | 6.9.0 | Step 7 |
| fastapi | 0.141.1 | Step 6 |
| uvicorn | 0.52.3 | Step 6 |
| streamlit | 1.61.1 | Step 7 |
| joblib | 1.5.3 | Saves the fitted transformer |
| pytest | 9.1.1 | Step 5 |
| ruff | 0.16.3 | Step 5 |
| black | 26.5.1 | Step 5 |
| pre-commit | 4.6.2 | Step 5 |
| kaggle | 2.2.4 | Current CLI, positional competition argument |
| dvc | 3.55 or newer | Added in Step 3 |

---

## 9. Conventions in force

**Code**
- All paths come from `config/config.py`. No module builds its own paths.
- One random seed, `RANDOM_SEED = 42`, used everywhere.
- Each pipeline stage reads a file and writes a file, so stages run and debug independently.
- `src/pipelines/` says what happens in what order. `src/features/` builds features. `src/utils/` says how individual things are done.
- Code that matters lives in `src/`, not in notebooks.
- Every function gets a docstring. Non-obvious lines get an inline comment.
- Every groupby over a category column passes `observed=True`.
- Matplotlib uses the `Agg` backend, set before pyplot is imported.
- Anything learned from data is learned from training rows only, then applied unchanged.

**Git**
- Branch naming: `step-NN-short-description`
- Commit style: `type: message`, using `feat`, `fix`, `docs`, `build`, `chore`, `test`, `refactor`
- One branch per step, squash-merged into `main` by pull request, then tagged `v0.N.0-stepN`
- `main` must always be in a working state

**Documentation**
- No em dashes
- Plain vocabulary, with advanced ideas explained rather than assumed
- An explanation before every code block, and comments inside the code
- Every file created gets stated, with full contents and the reason it exists
- When an earlier claim turns out to be wrong, it gets corrected openly in the next step rather than quietly amended

---

## 10. Completed

### Step 1, verified
- [x] Kaggle competition joined, credentials configured, all five files downloaded and verified
- [x] Full folder scaffold, `.gitignore`, `README.md`, `LICENSE`, `.env.example`, VS Code settings
- [x] Git initialised, public repo live, branch merged, tagged `v0.1.0-step1`
- [x] Python 3.11.9 environment built and locked
- [x] `config/config.py` with dynamic path resolution
- [x] Both data scripts written and run successfully

### Step 2, verified
- [x] `config/config.py` extended with feature families and EDA settings
- [x] Six code files created: memory, ingestion, EDA utilities and pipelines, plus `run.py`
- [x] Ingestion run: both splits, correct shapes, 63.9% and 64.1% memory reduction
- [x] EDA run: 15 V blocks found, no unmapped columns, ten charts, summary report
- [x] Metric decision (D-20) and split decision (D-21) made and documented
- [x] Branch merged, tagged `v0.2.0-step2`

### Step 3, delivered
- [x] Project moved out of OneDrive to `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`, `.venv` rebuilt, verification re-run successfully
- [x] `config/config.py` extended with Step 3 settings
- [x] `src/utils/column_selection.py` created
- [x] `src/utils/feature_utils.py` created
- [x] `src/features/` package and `engineer.py` created
- [x] `src/pipelines/features.py` created
- [x] `run.py` updated with the `features` stage
- [x] `README.md` rewritten as a standalone document
- [x] The `ProductCD` confound identified, quantified, and patched into the EDA report generator
- [ ] Feature stage run completed by the user
- [ ] DVC set up and verified by the user
- [ ] Branch merged and tagged `v0.3.0-step3`

---

## 11. Pending

**Immediately next (Step 4)**
- Load the processed features, read the `split` column rather than recomputing the split
- A baseline model first, so later numbers have something to be measured against
- LightGBM, XGBoost, and CatBoost trained and compared on identical splits
- MLflow from scratch, targeting the 3.x API: runs, parameters, metrics, artifacts, and the UI
- PR-AUC, ROC-AUC, and recall at 1% and 5% review rates
- Threshold selection tied to review capacity rather than left at 0.5
- Time-aware cross-validation with expanding windows
- SHAP explainability, globally and for a single prediction
- Check whether `has_identity` ranks low as D-31 predicts
- Register one model as the Step 5 candidate

**Later steps**
- Step 5: pytest suite, GitHub Actions CI, MLflow model registry, drift monitoring using the test set as genuine future data, retraining trigger. Monitoring outputs shaped so the Step 7 dashboard can read them directly.
- Step 6: Dockerfile, docker-compose, FastAPI service loading `feature_engineer.joblib`, deployment target
- Step 7: Streamlit dashboard per D-33, architecture diagram, README results, portfolio packaging

---

## 12. Open questions

| # | Question | Needed by | Status |
|---|----------|-----------|--------|
| Q-01 | Exact installed library versions | Step 2 | **Answered.** Section 8.1. |
| Q-02 | How much RAM | Step 2 | **Answered.** 32 GB, Intel Core Ultra 7 265H. |
| Q-03 | Where DVC stores data | Step 3 | **Answered.** Local folder remote at `C:\Users\Dauda Agbonoga\dvcstore`. See D-32. |
| Q-04 | Deployment target for Step 6 | Step 6 | Open. Render or Railway free tier, Hugging Face Spaces, a cloud provider, or local Docker only. |
| Q-05 | Kaggle late submission wanted? | Step 4 | Open. The competition is closed but late submissions still score, which would give an externally verified number for the README. Roughly twenty minutes of work. |
| Q-06 | Streamlit or React for the dashboard | Step 7 | Assumed Streamlit, confirmed by the user in Step 3. Effectively closed. |
| Q-07 | Business cost figures from the PM: cost of a missed fraud, cost of a false alarm | Step 4 | Open, and worth chasing. With them, threshold selection becomes a cost optimisation with a currency answer rather than a statistical exercise. |
| Q-08 | Project location | Step 3 | **Answered.** Moved offline to `Documents\Projects\ieee-cis-fraud-detection`. |
| Q-09 | Folder rename to match the repository | Step 3 | **Answered.** Done as part of the move. |
| Q-10 | Exact Python patch version | Step 3 | **Answered.** 3.11.9. |
| Q-11 | The V column block structure | Step 3 | **Answered.** 15 blocks, full detail in Section 5.4. |
| Q-12 | Who is the Step 7 dashboard for: a fraud analyst working a review queue, or a manager watching overall performance? | Step 7 | Open. The two lead to genuinely different layouts. An analyst needs a queue, per-transaction explanations, and fast filtering. A manager needs trends, totals, and drift alerts. |

---

## 13. How to resume from nothing

If everything is lost except the GitHub repository:

```powershell
# 1. Clone and enter
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

# 2. Recreate the environment with the exact same versions
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt

# 3. Rebuild the raw data (needs a Kaggle account that has joined the competition)
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

If the repository is also lost, `docs/steps/step1.md` through `step3.md` rebuild everything from scratch.

---

## 14. Glossary

| Term | Plain meaning |
|------|---------------|
| Virtual environment | A private copy of Python belonging to one project. Contains absolute paths, so it cannot be moved, only rebuilt |
| Parquet | A file format that stores tables column by column. Smaller and faster than CSV, and it remembers data types |
| Left join | Keep every row from the left table, attach matching data from the right where it exists, leave blanks where it does not |
| dtype | The data type of a column, such as int8 or float32 |
| Category dtype | Stores each distinct text value once, plus a small number per row pointing at it |
| Class imbalance | When one outcome is far rarer than the other, here 3.5% fraud |
| Accuracy | Share of predictions that were correct. Useless here, because always predicting "not fraud" scores 96.5% |
| Precision | Of the transactions you flagged, the share that were actually fraud |
| Recall | Of all the fraud that occurred, the share you caught |
| PR-AUC | Precision-Recall Area Under Curve. Primary metric. Baseline equals the fraud rate, 0.035 |
| ROC-AUC | Probability a randomly chosen fraud scores higher than a randomly chosen legitimate transaction. Baseline 0.5 |
| Time-based split | Train on earlier data, validate on later data. Imitates the real situation, where you always predict the future |
| Data leakage | When information unavailable at prediction time influences training, producing a validation score you cannot reproduce in production |
| Confounded comparison | A difference between two groups that is really a difference in what those groups are made of. See D-31 |
| Frequency encoding | Replacing a category with how often it appeared in training. Turns rarity into a number the model can split on |
| Aggregate feature | A row's value compared against the average for its group, for example this amount against the usual amount for this card |
| Fitted transformer | An object that learns from training data, stores what it learned, and applies it later. The defence against training and serving skew |
| Training and serving skew | When the model is trained on one set of transformations and fed another in production. Nothing errors, the predictions are just wrong |
| MLflow | Records every training run: settings used, metrics produced, and the model file |
| SHAP | Explains which features pushed a single prediction up or down |
| DVC | Versions large data files alongside the code that produced them, storing a small fingerprint in Git and the data elsewhere |
| Drift | When live data slowly stops resembling training data, so the model quietly gets worse |
| CI/CD | Automated checks and deployment that run on every code change |
| Model registry | A catalogue of trained model versions with a record of which one is live |

---

*End of PROJECT_STATE.md. Next: Step 4, model training and MLflow experiment tracking.*
