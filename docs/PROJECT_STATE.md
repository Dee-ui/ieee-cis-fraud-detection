# PROJECT_STATE.md

**Last updated:** End of Step 5 of 7
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
| Delivery format | 7 steps, one per conversation message, each with its own markdown guide plus a refreshed copy of this file |
| Platform | Windows, VS Code, PowerShell, Python 3.11.9 |
| Machine | Intel Core Ultra 7 265H, 32 GB RAM |
| Local path | `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`, offline, outside any sync folder |
| Version control | Git and GitHub for code, DVC with a local folder remote for processed data |
| DVC remote | `C:\Users\Dauda Agbonoga\dvcstore`, verified working |
| Experiment tracking | MLflow 3.15.1, SQLite backend at `mlflow.db` |
| Registered model | `ieee-cis-fraud-detector`, version 2 with alias `candidate` |
| Tracks | Technical track (this work) and a separate PM track walkthrough afterwards |
| Audience for final artefacts | Hiring managers and portfolio reviewers |

**Headline result so far:** LightGBM catching 44.6% of fraud cases at a 2% review rate, validation PR-AUC 0.6068 against a 0.0344 baseline, Kaggle private leaderboard 0.9140, worth roughly $1.76M a year under the documented cost model.

---

## 2. Why this dataset

Card fraud is a rare-event problem with a real cost structure on both sides: a missed fraud is a direct loss, a false alarm blocks a paying customer. Every technical decision has a business consequence you can point at.

Four properties make IEEE-CIS worth the effort:

- **Enough positive cases.** 20,663 frauds in 590,540 transactions, 3.4990%.
- **Two joinable tables** with partial coverage, so there is real data engineering.
- **Mostly anonymised features.** 339 columns with no published meaning, so structure has to be found empirically.
- **A test set 30 days in the future**, which makes time-based validation unavoidable and gives Step 5 a genuine distribution shift to detect rather than one that had to be manufactured.

---

## 3. The 7-step plan and current status

| Step | Content | Status |
|------|---------|--------|
| 1 | Dataset acquisition, folder scaffold, GitHub repo, Python environment | **Complete, verified** |
| 2 | EDA and data understanding: joins, feature families, imbalance profiling | **Complete, verified** |
| 3 | Feature engineering and preprocessing pipeline | **Complete, verified** |
| 4 | Model training with MLflow experiment tracking | **Complete, verified** |
| 5 | MLOps layer: tests, CI, drift monitoring, promotion gates | **Delivered, awaiting the user's run** |
| 6 | Dockerisation and deployment to Hugging Face Spaces | Not started |
| 7 | Streamlit dashboard and portfolio packaging | Not started |

---

## 4. Decision log

### Step 1: foundations

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | Dataset is IEEE-CIS Fraud Detection | 20,663 positive cases, a joinable second table, a time-separated test set. |
| D-02 | Repository named `ieee-cis-fraud-detection` | Descriptive. The local folder matches as of Step 3. |
| D-03 | Python 3.11 | Kaggle CLI minimum; stable prebuilt Windows wheels for every library. Confirmed 3.11.9. |
| D-04 | `venv` plus `requirements.txt`, not conda | One dependency format Docker and GitHub Actions both consume. |
| D-05 | Raw data never committed to Git | 1.3 GB of CSV. Reproducibility comes from `scripts/download_data.py`. |
| D-06 | DVC deferred to Step 3 | Resolved by D-32. |
| D-07 | Paths resolved dynamically in `config/config.py` | Proven twice: a mismatched folder name, then a whole-project move with zero code changes. |
| D-08 | Branch per step, merged by pull request, tagged after merge | Reviewable trail, triggers CI, clean narrative for the PM walkthrough. |
| D-09 | Public GitHub repository | Portfolio piece, plus free Actions minutes. |
| D-10 | Dependencies split into runtime and dev, plus a lock file | Keeps the Docker image lean; the lock file gives exact reproducibility. |
| D-11 | Download script shells out to the Kaggle CLI | The CLI is the documented stable contract. |
| D-12 | Interim and processed data as Parquet | Far smaller, much faster, preserves data types. |
| D-13 | Notebooks for exploration only | Notebooks are not testable, importable, or reviewable in diffs. |

### Step 2: data understanding

| ID | Decision | Rationale |
|----|----------|-----------|
| D-14 | `run.py` created in Step 2 rather than Step 3 | Two runnable stages existed. |
| D-15 | Test set joined and saved despite having no labels | It became the Step 5 drift input, showing genuine shift. |
| D-16 | Left join transaction to identity, keep blanks, add a `has_identity` flag | Boosters learn a direction for blanks, so blanks cost nothing. Partially revised by D-31. |
| D-17 | Interim data as Parquet with category dtypes preserved | Type work done once, not on every read. |
| D-18 | `TransactionAmt` stays `float64`; `TransactionID` and `TransactionDT` become `int32` | Verified numerically: `float32` is exact for integers only below 16,777,216 and test `TransactionDT` reaches 34,214,345. `float32` also turns 31937.39 into 31937.390625, and the cents are a fraud signal. |
| D-19 | Reference date 30 November 2017 for display only | The competition never published a start date. |
| D-20 | PR-AUC primary, ROC-AUC secondary, recall at a fixed review rate as the business headline, accuracy never | A do-nothing model scores 96.5% accuracy. Decided before any numbers existed, which is why the LightGBM/XGBoost tie-break in Step 4 was uncontroversial. |
| D-21 | Validation is a time-based split, last 20% by `TransactionDT` | The real test set is 30 days in the future. A random split leaks three ways. |
| D-22 | Feature families assigned by rule, unmapped columns reported loudly | Zero unmapped columns in the actual run. |

### Step 3: feature engineering

| ID | Decision | Rationale |
|----|----------|-----------|
| D-23 | Feature engineering is a fitted object saved with joblib, built on `BaseEstimator` and `TransformerMixin` | A single transaction at a web service cannot recompute frequency counts or group averages. |
| D-24 | The transformer is fitted only on the first 80% of the training period | Otherwise each validation row helps compute its own feature. |
| D-25 | Encodings learned from training rows only, never train and test combined | Common in competition write-ups, impossible in production. |
| D-26 | `TransactionDT`, `TransactionID`, and any absolute day counter excluded from features | Test values sit entirely above training values; trees cannot split outside their trained range. |
| D-27 | Near-constant columns dropped at a 99% dominance threshold, with a two-directional rescue rule | Dramatically vindicated. See Section 5.5. |
| D-28 | V columns reduced by correlation clustering inside each of the 15 blocks at 0.75 | 337 to 137, with reduction rates varying from 18% to 57% per block. |
| D-29 | The `uid` fingerprint used only for grouping and counting, never as a feature | Given directly, the model memorises individual customers. |
| D-30 | Text columns become integers with a stored mapping; blank gets its own code; unseen maps to -1 | Without a stored mapping the same word gets a different number next month. |
| D-31 | `has_identity` kept, with the confound recorded | **Confirmed in Step 4:** ranked 270th of 284 with SHAP exactly 0.0. |
| D-32 | DVC with a local folder remote at `C:\Users\Dauda Agbonoga\dvcstore` | No account, no internet, no cost. `dvc pull` verified. |
| D-33 | The Streamlit dashboard draws from two distinct sources and reads precomputed artifacts | Delivered in Step 5 as `dashboard_data.json`. |

### Step 4: model training

| ID | Decision | Rationale |
|----|----------|-----------|
| D-34 | A cost model with five stated assumptions, stored in config, drives the threshold | Turns an abstract metric into money. |
| D-35 | Costs weighted by the actual transaction amount, not a flat penalty | This is what revealed the count-versus-value gap in Section 5.7. |
| D-36 | The uid features get a pre-registered ablation with a 0.005 PR-AUC threshold set in advance | Result: 0.01289, so **kept**. The rule decided, not hindsight. |
| D-37 | Five candidates: dummy, logistic regression, LightGBM, XGBoost, CatBoost | The dummy gave a measured floor that validated the metric code exactly. |
| D-38 | No class weighting and no resampling | We need ordering; weighting shifts probabilities without reliably improving order and makes scores uninterpretable. |
| D-39 | Category codes treated as ordinary numbers | Frequency counts already supply the same information in a form that cannot overfit. |
| D-40 | Early stopping on validation PR-AUC; cross-validation afterwards with the round count fixed | Keeps CV an honest stability check. |
| D-41 | The final model retrained on all labelled data with rounds scaled by the row ratio | 617 rounds scaled 1.25x to 771. |
| D-42 | MLflow tracking URI built with `.as_posix()` | Windows backslashes are unreliable in a SQLAlchemy URL. |
| D-43 | The chosen model registered under an alias, not a stage | MLflow 3 deprecated stages. |
| D-44 | Step 6 deploys to Hugging Face Spaces using the Docker SDK | Free, runs the real Docker image, public clickable URL with no cold start, and FastAPI's `/docs` gives an interactive demo. Render is the backup. |
| D-45 | The Step 7 dashboard is built for a hiring manager reading it cold in under two minutes | Loads in under three seconds from precomputed files, leads with the money, one interactive scorer, visible MLOps signals. |
| D-46 | A Kaggle late submission is produced | Free external validation. Scored on ROC-AUC, which is not our primary metric, and the mismatch is reported honestly. |

### Step 5: the MLOps layer

| ID | Decision | Rationale |
|----|----------|-----------|
| D-47 | Every model run is tagged `run_mode`, and quick runs cannot be registered | Registry version 1 is a 150-round quick-mode test model that registered itself. Nothing downstream could tell it from a real one. |
| D-48 | Model schemas declare integer columns as floats | JSON has one number type, so the Step 6 API would send floats where the schema demands integers and be rejected inside a container. |
| D-49 | Feature importance records max absolute SHAP alongside the mean | Mean importance systematically hides rare-but-decisive features. V111 ranked 259th on mean despite a 46% fraud rate on its rare rows. See Section 5.6. |
| D-50 | Tests run on synthetic data only, never on the real dataset | The dataset is 1.3 GB and not in the repository. Tests that cannot run in CI do not get run. |
| D-51 | A row-independence test is the primary leakage guard | Transforming one row must equal transforming a batch containing it. Catches anyone reintroducing a groupby in `transform`, and is exactly what Step 6 needs. |
| D-52 | CI installs a light dependency set, not the full environment | The full environment is ~2.5 GB. Tests only need pandas, numpy, scipy, scikit-learn, pyarrow. |
| D-53 | Drift measured with PSI as primary, KS as secondary, missingness alongside | PSI catches distribution collapse, which a missingness check misses entirely. `uid_freq` is the worked example: it never goes blank, it just becomes 0.0 for 82% of rows. |
| D-54 | The KS statistic is used, never the KS p-value | On 100,000 rows every difference is significant, so the p-value would flag all 284 features every month. |
| D-55 | Feature drift is weighted by SHAP importance before becoming a verdict | With 284 features a few will always have drifted. A raw count fires constantly and gets ignored. |
| D-56 | Promotion from `candidate` to `production` runs through six gates and is a separate deliberate command | Training says "here is a candidate". Promotion says "this is fit to serve". Gate 1 alone would have stopped version 1. |
| D-57 | The monitoring stage writes a small `dashboard_data.json` | Per D-45 the dashboard cannot compute anything from the 590,540 row table at page load. |
| D-58 | Recall is always reported by count **and** by value | The model catches 44.6% of fraud cases but only 31.2% of fraud money. Quoting the count alone overstates the benefit by about 43%. |

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

Fraud rate **3.4990%**, 20,663 of 590,540. 144,233 transactions have an identity record, **24.4%**. Train uses `id_` prefixes, test uses `id-`.

### 5.2 Ingestion, Step 2

| Split | Rows | Columns | Memory before | Memory after | Reduction | Parquet |
|-------|------|---------|---------------|--------------|-----------|---------|
| train | 590,540 | 435 | 2,567.7 MB | 927.2 MB | 63.9% | 80.3 MB |
| test | 506,691 | 434 | 2,214.5 MB | 795.2 MB | 64.1% | 69.8 MB |

Runtime 3m 07s. Train dtypes: 398 `float32`, 31 `category`, 2 `int8`, 2 `int32`, 1 `float64`, 1 `int16`.

### 5.3 EDA, Step 2

**Time coverage:** train 2017-12-01 to 2018-05-31 (182 days), test 2018-07-01 to 2018-12-30 (183 days), gap **30 days**.

**Identity coverage and the confound**

| Group | Transactions | Fraud rate |
|-------|--------------|------------|
| No identity record | 446,307 | 2.0939% |
| Has identity record | 144,233 | 7.8470% |

**The 3.75x figure must never be quoted without its caveat.** Coverage is almost decided by `ProductCD`: W is 0%, C is 90.8%, H/R/S are 99.6%. W also has the lowest fraud rate (2.04%) and is 439,670 of 590,540 rows. Restricted to non-W products: 7.85% against 5.67%, a lift of **1.39x**. The model settled this in Step 4 by ranking `has_identity` 270th of 284 with SHAP exactly 0.0.

Test identity coverage is **28.0%** against training's 24.4%.

**Fraud rate by category**

| Column | Highest | Lowest |
|--------|---------|--------|
| ProductCD | C at 11.69% | W at 2.04% |
| card4 | discover at 7.73% | american express at 2.87% |
| card6 | credit at 6.68% | debit at 2.43% |
| DeviceType | mobile at 10.17% | missing at 2.10% |
| P_emaildomain | mail.com at 18.96% | aol.com at 2.18% |

**Missing data.** 53 columns with none. 12 above 90%. The nine emptiest are identity columns between 99.12% and 99.20%. The C family has zero missing across all 14.

### 5.4 The V column blocks, Step 2

All 339 V columns fall into 15 blocks sharing an identical missing pattern.

| Block | Columns | Missing | Range | Shape | Kept |
|-------|---------|---------|-------|-------|------|
| 1 | 46 | 77.91% | V217-V278 | interleaved | 13 |
| 2 | 43 | 0.05% | V95-V137 | solid | 24 (of 42 surviving) |
| 3 | 32 | 0.00% | V279-V321 | interleaved | 15 (of 31) |
| 4 | 31 | 76.36% | V167-V216 | interleaved | 8 |
| 5 | 23 | 12.88% | V12-V34 | solid | 8 |
| 6 | 22 | 13.06% | V53-V74 | solid | 8 |
| 7 | 20 | 15.10% | V75-V94 | solid | 8 |
| 8 | 19 | 76.32% | V169-V210 | interleaved | 10 |
| 9 | 18 | 28.61% | V35-V52 | solid | 8 |
| 10 | 18 | 86.12% | V138-V163 | interleaved | 6 |
| 11 | 18 | 86.05% | V322-V339 | solid | 7 |
| 12 | 16 | 76.05% | V220-V272 | interleaved | 6 |
| 13 | 11 | 47.29% | V1-V11 | solid | 7 |
| 14 | 11 | 86.12% | V143-V166 | interleaved | 2 |
| 15 | 11 | 0.21% | V281-V315 | interleaved | 7 |

**Blocks 10 and 14 both sit at 86.12% missing but are different blocks.** Grouping by missing count would have merged them; hashing the actual pattern separated them.

**Eight of fifteen blocks interleave**, so chopping by number range would cut across the real groupings.

Reduction varied from 18% kept (block 14) to 57% (block 2). A flat rule would have over-cut one and under-cut the other.

### 5.5 Feature engineering, Step 3

Runtime 2m 25s. All four verification checks passed.

| Stage | Result |
|-------|--------|
| Input columns | 435 |
| Candidates assessed | 432 |
| Dropped: single value | 0 |
| Dropped: near-constant | 2 (V107, V305) |
| Rescued | 22 |
| V columns | 337 reduced to 137 |
| **Final features** | **284** |
| Transformer file | 28.0 MB |
| Train / test Parquet | 84.4 MB / 68.5 MB |
| Unseen lookups in test | 6.81% |

**Composition:** 199 base_numeric, 38 category_code, 18 aggregate, 18 frequency, 3 derived_amount, 3 derived_screen, 2 derived_time, 2 derived_match, 1 derived_email.

**The rescue rule was strongly vindicated.** Only 2 dropped, 22 rescued:

| Column | Dominant | Share | Rare rows | Fraud rate among rare | Lift |
|--------|----------|-------|-----------|----------------------|------|
| V111 | 1.0 | 99.71% | 1,370 | **46.35%** | 13.2x |
| V113 | 1.0 | 99.65% | 1,645 | 39.51% | 11.2x |
| V117 | 1.0 | 99.88% | 578 | 31.14% | 8.9x |
| V112 | 1.0 | 99.49% | 2,431 | 29.25% | 8.3x |
| V108 | 1.0 | 99.52% | 2,283 | 28.03% | 8.0x |
| V118, V114, V110, V119, V120, V122, V121 | 1.0 | 99.1-99.9% | 638-4,163 | 7.3-28.7% | 2.1-8.2x |
| Nine `id_` columns | blank | 99.10-99.17% | 3,914-4,273 | 7.80-8.05% | ~2.2x |
| C3 | 0.0 | 99.60% | 1,872 | **0.053%** | **0.015x** |

V111 would have been deleted by any blanket rule. C3 survived only because the rescue triggers in both directions, on rare values that are unusually **safe** as well as unusually risky. The nine identity columns were flagged and rescued, so my Step 3 prediction that they would be dropped was wrong.

**The split**

| Portion | Rows | Frauds | Fraud rate | Period |
|---------|------|--------|------------|--------|
| train | 472,432 | 16,599 | 3.5135% | 2017-12-01 to 2018-04-20 |
| valid | 118,108 | 4,064 | 3.4409% | 2018-04-20 to 2018-05-31 |

Boundary at TransactionDT 12,192,854. Validation window 42 days.

### 5.6 Model training, Step 4

Runtime 21m 32s.

| Model | PR-AUC | Lift | ROC-AUC | Rounds | Minutes |
|-------|--------|------|---------|--------|---------|
| **lightgbm** | **0.60682** | 17.6x | 0.92751 | 617 | 0.71 |
| xgboost | 0.59907 | 17.4x | **0.93079** | 1,193 | 4.35 |
| catboost | 0.52819 | 15.4x | 0.89368 | 1,500 | 7.14 |
| logistic_regression | 0.18309 | 5.3x | 0.82095 | n/a | 1.07 |
| dummy | 0.03441 | 1.0x | 0.50000 | n/a | 0.03 |

**The dummy check passed exactly.** PR-AUC 0.03441 equals the validation fraud rate and ROC-AUC is exactly 0.50000, which proves the metric code is sound.

**The two metrics disagreed.** LightGBM won PR-AUC, XGBoost won ROC-AUC. D-20 had already fixed PR-AUC as the tie-breaker, so this was not a judgement call. One consequence worth stating: Kaggle scores ROC-AUC, so XGBoost would likely have scored slightly higher on the leaderboard.

**CatBoost never converged.** Best iteration 1,499 of 1,500. Its score understates it and the comparison is not clean. Optional fix: raise `MAX_BOOSTING_ROUNDS` to 4,000 and run `--models catboost`.

**Cross-validation, expanding windows**

| Fold | Train rows | Validation period | PR-AUC |
|------|-----------|-------------------|--------|
| 1 | 118,108 | 2017-12-26 to 2018-02-02 | 0.61833 |
| 2 | 236,216 | 2018-02-02 to 2018-03-11 | 0.63763 |
| 3 | 354,324 | 2018-03-11 to 2018-04-20 | **0.67082** |
| 4 | 472,432 | 2018-04-20 to 2018-05-31 | **0.60682** |

Mean 0.63340, spread 0.02800. **Fold 4 has the most training data and the second-worst score.** That is not a data-volume effect, it is a period-difficulty effect, and it is the strongest single motivation for the monitoring built in Step 5.

**The uid ablation:** with 0.60682, without 0.59393, difference +0.01289. Above the 0.005 pre-registered tolerance, so **kept**.

**Cost analysis, 42 day validation window**

| Operating point | Review rate | Recall (count) | Savings |
|-----------------|-------------|----------------|---------|
| Doing nothing | 0% | 0% | baseline cost $711,534 |
| Cheapest overall | 18.86% | 86.0% | $444,996 |
| Cheapest within 2% capacity | 2.00% | 44.6% | $202,013 |

Annualised at the within-capacity point: **$1,760,894**. Chosen threshold **0.4222**.

| Review rate | Threshold | Recall | Precision | Savings |
|-------------|-----------|--------|-----------|---------|
| 0.5% | 0.95653 | 13.8% | 94.6% | $57,414 |
| 1.0% | 0.83433 | 26.6% | 91.5% | $114,501 |
| 2.0% | 0.42142 | 44.6% | 76.7% | $202,013 |
| 5.0% | 0.09626 | 64.3% | 44.3% | $339,362 |

**Kaggle:** public 0.944058, private **0.914018**. Single model, no ensembling, no test-set leakage.

**Registry:** version 2, alias `candidate`. MLflow run `68850ae7c1264e80ba87229fa54ed899`. Final model 5.3 MB.

**Top SHAP features:** C13 (0.290), C14 (0.133), TransactionAmt_ratio_to_addr1_mean (0.121), C1 (0.117), V70 (0.115), D15_std_by_uid (0.107), D15_mean_by_uid (0.104), uid_freq (0.103), card1_freq (0.100), TransactionAmt_mean_by_card1 (0.097).

### 5.7 Findings from Step 4 that change how the model is described

**`has_identity` ranked 270th of 284 with SHAP exactly 0.0.** D-31 predicted it would rank low and it ranked at the very bottom, used zero times. The confound analysis was correct.

**The rescued V columns ranked low on mean SHAP, and that is a measurement artefact, not a verdict.** V111 ranked 259th with mean SHAP 0.000091. But V111 holds 1.0 on 99.71% of rows, so in a 5,000 row SHAP sample only about 14 rows have it non-1.0. A huge effect averaged over 14 rows and nothing over 4,986 is near zero. **Mean absolute SHAP systematically hides rare-but-decisive features**, which in fraud detection is an entire category of the most valuable signals. D-49 adds max absolute SHAP for future runs. C3, at rank 108 with 0.0110, is the rescue rule paying off in a way mean importance *can* see.

**The uid family carries 9.9% of total SHAP mass**, with four of the top twenty features (`D15_std_by_uid` 6th, `D15_mean_by_uid` 7th, `uid_freq` 8th, `TransactionAmt_mean_by_uid` 14th). Those features are blank on roughly 82% of test rows. This is the number one production risk in the project.

**The uid family is 7 features, not 6.** The marker rule also caught `uid_freq`, which the Step 3 manifest analysis missed because `uid_freq` does not go *missing*. An unseen uid returns a frequency of 0.0, not a blank. A feature that collapses onto one constant value looks perfectly healthy in a missingness check while carrying almost no information. This is why PSI, not a missingness check, is the primary drift signal (D-53).

**The model catches cheap fraud and misses expensive fraud.** Derived from the reported cost figures:

| Measure | Value |
|---------|-------|
| Total fraud value, 42 days | $609,934 |
| Mean fraud amount | $150.08 |
| Recall by **count** | 44.6% |
| Recall by **value** | **31.2%** |
| Mean amount of a caught fraud | **$105** |
| Mean amount of a missed fraud | **$186** |

Missed frauds are on average 77% larger than caught ones, because a large fraudulent purchase resembles a large legitimate one. The cost model handles this correctly, which is why the saving is $1.76M rather than the $2.5M a count-based estimate would give. But quoting recall alone overstates the benefit by about 43%. Hence D-58.

---

## 6. Current repository structure

```
ieee-cis-fraud-detection/
│
├── .dvc/ , .dvcignore                  # DVC config                (Step 3)
├── .github/workflows/ci.yml            # ruff, black, pytest       (Step 5)
├── .pre-commit-config.yaml                                         # Step 5
├── .vscode/settings.json
├── app/                                # empty                     (Step 7)
│
├── config/
│   ├── __init__.py
│   └── config.py                       # extended in Steps 2,3,4,5
│
├── data/
│   ├── raw/                            # git-ignored, 1.29 GB
│   ├── interim/                        # git-ignored
│   │   ├── train_joined.parquet        # 590,540 x 435,  80.3 MB
│   │   └── test_joined.parquet         # 506,691 x 434,  69.8 MB
│   ├── processed/                      # DVC-tracked
│   │   ├── train_features.parquet      # 84.4 MB
│   │   ├── test_features.parquet       # 68.5 MB
│   │   ├── kaggle_submission.csv       # 14.6 MB
│   │   └── *.dvc pointer files         # committed to git
│   └── external/                       # empty
│
├── docker/                             # empty                     (Step 6)
│
├── docs/
│   ├── PROJECT_STATE.md                # this file
│   ├── steps/step1.md ... step5.md
│   └── decisions/                      # empty
│
├── models/                             # git-ignored except metadata
│   ├── feature_engineer.joblib         # 28.0 MB                   (Step 3)
│   ├── final_model.joblib              #  5.3 MB                   (Step 4)
│   ├── selection_model.joblib          # train-portion only        (Step 5)
│   └── final_model_metadata.json       # committed                 (Step 4)
│
├── notebooks/                          # empty
│
├── reports/
│   ├── data_inventory.md                                           # Step 1
│   ├── eda_summary.md , column_profile.csv , missing_profile.csv   # Step 2
│   ├── v_column_missing_groups.csv                                 # Step 2
│   ├── feature_summary.md , feature_manifest.csv                   # Step 3
│   ├── dropped_columns.csv , v_column_reduction.csv                # Step 3
│   ├── training_summary.md , model_comparison.csv                  # Step 4
│   ├── threshold_analysis.csv , cost_curve.csv                     # Step 4
│   ├── cv_results.csv , feature_importance.csv                     # Step 4
│   ├── monitoring/                                                 # Step 5
│   │   ├── feature_drift.csv , score_drift.csv
│   │   ├── period_metrics.csv , drift_summary.md
│   │   └── dashboard_data.json         # feeds the Step 7 dashboard
│   ├── figures/                        # 19 PNG charts
│   └── explainability/                 # 3 SHAP charts             (Step 4)
│
├── scripts/
│   ├── download_data.py , verify_data.py                           # Step 1
│   └── promote_model.py                                            # Step 5
│
├── src/
│   ├── __init__.py
│   ├── features/engineer.py                                        # Step 3
│   ├── models/candidates.py                                        # Step 4
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift.py                                                # Step 5
│   │   └── promotion.py                                            # Step 5
│   ├── pipelines/
│   │   ├── ingestion.py , eda.py                                   # Step 2
│   │   ├── features.py                                             # Step 3
│   │   ├── training.py                                             # Step 4
│   │   └── monitoring.py                                           # Step 5
│   ├── serving/__init__.py             # modules added             (Step 6)
│   └── utils/
│       ├── memory_utils.py , ingestion_utils.py , eda_utils.py     # Step 2
│       ├── column_selection.py , feature_utils.py                  # Step 3
│       ├── metrics.py , mlflow_utils.py , model_plots.py           # Step 4
│       └── monitoring_plots.py                                     # Step 5
│
├── tests/
│   ├── __init__.py , conftest.py                                   # Step 5
│   ├── test_metrics.py , test_feature_engineer.py                  # Step 5
│   └── test_leakage.py , test_drift.py                             # Step 5
│
├── .env.example , .gitignore , LICENSE , README.md
├── pyproject.toml                      # ruff, black, pytest       (Step 5)
├── requirements.txt , requirements-dev.txt , requirements.lock.txt
├── requirements-ci.txt                                             # Step 5
├── mlflow.db                           # git-ignored               (Step 4)
└── run.py                              # 5 stages
```

---

## 7. Files and what each one does

### Step 1

| File | Purpose |
|------|---------|
| `.gitignore` | Blocks raw and interim data, models, secrets, `.venv`, MLflow artifacts, editor noise. The `data/processed/*` lines were removed in Step 3 for DVC. |
| `scripts/download_data.py` | Kaggle CLI download with fallback flag syntax, extraction, size reporting. |
| `scripts/verify_data.py` | Five integrity checks, using `usecols` so it never loads all 394 columns. |

### Step 2

| File | Purpose |
|------|---------|
| `src/utils/memory_utils.py` | `optimise_dtypes` with `PROTECTED_DTYPES` for the four columns where shrinking corrupts data. |
| `src/utils/ingestion_utils.py` | Load, standardise `id-` to `id_`, mark, left join with `validate="one_to_one"`, save Parquet. |
| `src/pipelines/ingestion.py` | One code path for train and test via `SPLIT_SETTINGS`. |
| `src/utils/eda_utils.py` | Family assignment, profiling, `missing_pattern_groups` (md5 over each column's blank mask), ten charts. |
| `src/pipelines/eda.py` | Four reports, ten charts. Patched in Step 3 to carry the `ProductCD` caveat. |

### Step 3

| File | Purpose |
|------|---------|
| `src/utils/column_selection.py` | `assess_near_constant_columns` (the two-directional rescue rule), `cluster_by_correlation`, `reduce_v_columns`. |
| `src/utils/feature_utils.py` | `as_label_series` (the single place numbers become text), time, amount, email, screen, device, match, uid builders. |
| `src/features/engineer.py` | `FraudFeatureEngineer`. `_transform_frame` shared by `fit` and `transform`. Feature list fixed by running a real transform during `fit`. |
| `src/pipelines/features.py` | Split by time, fit on the training portion only, four verification checks. |

### Step 4

| File | Purpose |
|------|---------|
| `src/utils/metrics.py` | `ranking_metrics`, `review_rate_metrics`, `cost_curve` (exact, via cumulative sums), `best_operating_point`. |
| `src/utils/mlflow_utils.py` | `configure_mlflow`, `log_model_compatibly` handling both the `name`/`artifact_path` change and the skops trusted-types requirement. |
| `src/utils/model_plots.py` | Five charts: model comparison, PR curves, cost curve, score distribution, CV stability. |
| `src/models/candidates.py` | `Candidate` dataclass, per-library fit adapters, `rebuild_for_refit`, `expanding_window_splits`. |
| `src/pipelines/training.py` | Eight phases from load to registry and submission. |

### Step 5

| File | Purpose |
|------|---------|
| `pyproject.toml` | ruff, black, pytest, and coverage settings in one place. |
| `requirements-ci.txt` | The light dependency set CI installs, about 200 MB instead of 2.5 GB. |
| `tests/conftest.py` | Synthetic fixtures mirroring the joined table's shape. |
| `tests/test_metrics.py` | The cost model against hand arithmetic, plus the constant-score floor check. |
| `tests/test_feature_engineer.py` | Joblib round-trip and the row-independence guard. |
| `tests/test_leakage.py` | Time-ordered split, unseen-value handling, no feature tracking the clock. |
| `tests/test_drift.py` | PSI catches a collapse onto one value that a missingness check misses. |
| `src/monitoring/drift.py` | PSI, KS statistic, missingness, `weighted_drift_score`. |
| `src/monitoring/promotion.py` | The six gates, returning all results rather than stopping at the first failure. |
| `src/utils/monitoring_plots.py` | Four charts: weekly performance, drift grid, score drift, alert rate. |
| `src/pipelines/monitoring.py` | Weekly labelled performance plus monthly unlabelled drift, and `dashboard_data.json`. |
| `scripts/promote_model.py` | The promotion command, with `--dry-run`. |
| `.github/workflows/ci.yml` | ruff, black, pytest on every push and pull request. |
| `.pre-commit-config.yaml` | The same checks locally at commit time. |

---

## 8. Environment

| Item | Value |
|------|-------|
| Python | **3.11.9** |
| Environment | `.venv` in the project root, git-ignored |
| Activate | `.\.venv\Scripts\Activate.ps1` |
| Rebuild exactly | `pip install -r requirements.lock.txt` |

### 8.1 Confirmed library versions

| Library | Version | Notes that shaped the code |
|---------|---------|----------------------------|
| pandas | 2.3.3 | `observed=True` on category groupbys; empty-frame concat deprecated, fixed in Step 4 |
| numpy | 2.4.6 | numpy 2.x, so `np.NaN` and `np.float_` do not exist |
| pyarrow | 24.0.0 | Parquet engine, preserves category dtypes |
| scipy | 1.17.1 | `ks_2samp` for drift |
| scikit-learn | 1.9.0 | `BaseEstimator`, `TransformerMixin`, `Pipeline`, metrics |
| lightgbm | 4.7.0 | **The chosen model.** `eval_set` deprecated in favour of `eval_X`/`eval_y`, handled by inspection |
| xgboost | 3.2.0 | Early stopping in the constructor, `eval_metric="aucpr"` |
| catboost | 1.2.10 | `eval_metric="PRAUC"`, `allow_writing_files=False`. Did not converge in 1,500 rounds |
| imbalanced-learn | 0.14.2 | Installed but deliberately unused, D-38 |
| mlflow | 3.15.1 | Aliases not stages; `log_model` uses `name`; URI needs forward slashes; **sklearn flavor saves via skops and needs `skops_trusted_types=["numpy.dtype"]`** |
| shap | 0.51.0 | `TreeExplainer`. Mean absolute SHAP hides rare features, see D-49 |
| matplotlib | 3.11.1 | `plt.cm.get_cmap` removed in 3.9, so unused |
| seaborn | 0.13.2 | |
| plotly | 6.9.0 | Step 7 |
| fastapi | 0.141.1 | Step 6 |
| uvicorn | 0.52.3 | Step 6 |
| streamlit | 1.61.1 | Step 7 |
| joblib | 1.5.3 | Saves the transformer and models |
| pytest | 9.1.1 | Step 5 |
| ruff | 0.16.3 | Step 5 |
| black | 26.5.1 | Step 5 |
| pre-commit | 4.6.2 | Step 5 |
| kaggle | 2.2.4 | Positional competition argument |
| dvc | 3.55 or newer | Local folder remote, verified |

---

## 9. The cost model

Introduced in Step 4 (D-34, D-35). **These are stated assumptions, not figures from a business.** All five live in `config/config.py`.

| Assumption | Value | Reasoning |
|------------|-------|-----------|
| Analyst review | $4.00 per case | Fully loaded analyst at ~$60k/year is ~$29/hour; a five minute review is $2.40; rounded up for supervision and customer calls |
| Chargeback fee | $25.00 per missed fraud | Card networks charge $15 to $40 per dispute on top of the clawback |
| False alarm friction | $1.00 | Expected value of holding and releasing a legitimate customer. The softest number, the first to replace |
| Fraud recovered when caught | 90% | Reviews take time, some are judged wrongly, some transactions have settled |
| Review capacity | 2% of transactions | About one in fifty, roughly one analyst's full shift at this volume |

Missed fraud costs the amount plus the fee; caught fraud costs a review plus the 10% not recovered; a false alarm costs a review plus friction; a correct pass costs nothing. The baseline is doing nothing, where every fraud is missed.

Costs are weighted by the real transaction amount. The curve is computed exactly at every possible threshold using cumulative sums over score-sorted rows.

**Framing for the PM track:** present the annual saving as an order of magnitude under stated assumptions, with sensitivity to each one, never as a forecast. Always pair the count-based recall with the value-based recall (D-58).

---

## 10. Conventions in force

**Code**
- All paths come from `config/config.py`. No module builds its own.
- One random seed, `RANDOM_SEED = 42`.
- Each pipeline stage reads a file and writes a file.
- `src/pipelines/` orders stages, `src/features/` builds features, `src/models/` defines candidates, `src/monitoring/` watches, `src/utils/` supports all of them.
- Anything learned from data is learned from training rows only, then applied unchanged.
- Every groupby over a category column passes `observed=True`.
- Matplotlib uses `Agg`, set before pyplot is imported.
- Decision rules that use results are written down before the results are seen.
- When a library is mid-transition, inspect what is actually installed rather than assuming. Used three times: MLflow `name`/`artifact_path`, LightGBM `eval_set`/`eval_X`, and skops trusted types.

**Git**
- Branch `step-NN-short-description`; commits `type: message`; squash-merge by pull request; tag `v0.N.0-stepN`
- `main` must always work, and CI must be green before merging
- Formatting changes go in their own commit, never mixed with a feature

**Documentation**
- No em dashes; plain vocabulary; explanation before every code block
- Every file created is stated with full contents and its reason
- When an earlier claim turns out to be wrong, it is corrected openly in the next step rather than quietly amended

**Corrections made so far**
1. Parquet size estimate was 4x too high (Step 3)
2. Memory reduction range was slightly optimistic (Step 3)
3. The 3.75x identity finding was confounded by `ProductCD` (Step 3)
4. The nine `id_` columns were predicted to be dropped; all nine were rescued (Step 4)
5. The uid family is 7 features, not 6 (Step 5)
6. Mean absolute SHAP was the wrong tool for judging rare-but-decisive features (Step 5)
7. The README stated the 42-day saving as an annual figure (Step 5)

---

## 11. Completed

### Steps 1 to 4, verified
- [x] Data downloaded and verified, repo live, environment locked, tagged `v0.1.0-step1`
- [x] Ingestion and EDA run, 15 V blocks found, tagged `v0.2.0-step2`
- [x] Project moved offline, `.venv` rebuilt, verification re-run
- [x] Feature stage run, 284 features, all four checks passed, tagged `v0.3.0-step3`
- [x] DVC set up; `dvc pull` restored a deleted file
- [x] Training run: LightGBM wins at 0.60682, registered as version 2
- [x] uid ablation ran against its pre-registered rule; result: keep
- [x] Kaggle late submission: private 0.914018
- [x] `has_identity` confirmed at rank 270 of 284, validating D-31

### Step 5, delivered
- [x] Four fixes specified: LightGBM eval argument, integer schema, quick-run registry guard, max SHAP
- [x] The user's skops fix reviewed and endorsed
- [x] `pyproject.toml` and `requirements-ci.txt` specified
- [x] Five test files specified, including the row-independence leakage guard
- [x] `src/monitoring/drift.py` and `promotion.py` specified
- [x] `src/pipelines/monitoring.py` and `src/utils/monitoring_plots.py` specified
- [x] `scripts/promote_model.py` with six gates specified
- [x] GitHub Actions and pre-commit specified
- [x] README rewritten, including the $202,013 annual-figure error
- [x] The count-versus-value finding derived and documented
- [ ] Tests run by the user
- [ ] Monitoring stage run by the user
- [ ] Promotion gates run against versions 1 and 2
- [ ] CI green
- [ ] Branch merged and tagged `v0.5.0-step5`

---

## 12. Pending

**Immediately next (Step 6)**
- Dockerfile, with layer ordering explained for build speed
- FastAPI service: `/health`, `/predict`, `/predict/batch`, and the automatic `/docs` page
- Loading the model from the registry alias, so deploying means moving a pointer
- Pydantic request validation
- Single-row scoring, already proved safe by the row-independence test
- `docker compose` for the service plus MLflow locally
- Artifacts published to the Hugging Face Model Hub so the container downloads them at startup rather than baking 33 MB into the image
- Deployment to Hugging Face Spaces, Docker SDK, per D-44
- CI extended to build the image on every push
- A response time budget

**Step 7**
- Streamlit dashboard per D-33 and D-45, reading `dashboard_data.json`
- Five sections: headline and business impact, model performance, a live transaction scorer, drift monitoring, how it was built
- Architecture diagram, final README, portfolio packaging

---

## 13. Open questions

| # | Question | Needed by | Status |
|---|----------|-----------|--------|
| Q-01 | Library versions | Step 2 | **Answered.** Section 8.1. |
| Q-02 | RAM | Step 2 | **Answered.** 32 GB. |
| Q-03 | DVC remote | Step 3 | **Answered.** Local folder, verified. |
| Q-04 | Deployment target | Step 6 | **Answered.** Hugging Face Spaces, Docker SDK. D-44. |
| Q-05 | Kaggle late submission | Step 4 | **Answered.** Done: private 0.914018. |
| Q-06 | Streamlit or React | Step 7 | **Answered.** Streamlit. |
| Q-07 | Business cost figures | Step 4 | **Answered by construction.** Five documented assumptions. Section 9. |
| Q-08 | Project location | Step 3 | **Answered.** Moved offline. |
| Q-09 | Folder rename | Step 3 | **Answered.** Done. |
| Q-10 | Python patch version | Step 3 | **Answered.** 3.11.9. |
| Q-11 | V block structure | Step 3 | **Answered.** 15 blocks, Section 5.4. |
| Q-12 | Dashboard audience | Step 7 | **Answered.** Portfolio and hiring managers. D-45. |
| Q-13 | uid ablation outcome | Step 5 | **Answered.** Kept: 0.01289 against a 0.005 tolerance. The family carries 9.9% of SHAP mass and is blank on 82% of test rows, so it is the primary monitoring target. |
| Q-14 | Winner, and does `has_identity` rank low? | Step 5 | **Answered.** LightGBM. `has_identity` ranked 270 of 284 with SHAP exactly 0.0, confirming D-31. |
| Q-15 | Hugging Face account and token | Step 6 | **Open and now needed.** Free at huggingface.co. A write token is required from Settings, Access Tokens. |
| Q-16 | Should the model be retrained with amount-weighted examples? | Future | Open. It catches 44.6% of fraud by count but 31.2% by value. Weighting by amount would push it towards the money, probably lowering PR-AUC by count while raising recall by value. Worth measuring, outside the 7-step scope. |
| Q-17 | Should CatBoost be re-run with a larger round budget? | Optional | Open. It stopped at iteration 1,499 of 1,500 still improving, so its 0.52819 understates it. About fifteen minutes to settle. |

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

# 3. Confirm the code is sound before trusting anything it produces
pytest

# 4. Rebuild the raw data (needs a Kaggle account that joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# 5. Rebuild everything downstream
python run.py --step all

# 6. Or, if the DVC remote survived, pull the processed data directly
dvc pull

# 7. Read the current state
code docs/PROJECT_STATE.md
```

If the repository is also lost, `docs/steps/step1.md` through `step5.md` rebuild everything from scratch.

---

## 15. Glossary

| Term | Plain meaning |
|------|---------------|
| Parquet | A file format storing tables column by column. Smaller and faster than CSV, and it remembers data types |
| Left join | Keep every row from the left table, attach matching data from the right where it exists, leave blanks otherwise |
| Class imbalance | When one outcome is far rarer than the other, here 3.5% fraud |
| Accuracy | Share of predictions correct. Useless here: always predicting "not fraud" scores 96.5% |
| Precision | Of the transactions you flagged, the share that really were fraud |
| Recall | Of all the fraud that occurred, the share you caught. Report by count **and** by value, D-58 |
| PR-AUC | Precision-Recall Area Under Curve. Primary metric. Baseline equals the fraud rate |
| ROC-AUC | Probability a random fraud scores above a random legitimate transaction. Baseline 0.5. The Kaggle metric |
| Time-based split | Train on earlier data, validate on later data. Imitates predicting the future |
| Data leakage | Information unavailable at prediction time influencing training, producing a score you cannot reproduce in production |
| Confounded comparison | A difference between two groups that is really a difference in what those groups contain. See D-31 |
| Frequency encoding | Replacing a category with how often it appeared in training. Unseen values get 0, which is why `uid_freq` collapses on test data |
| Aggregate feature | A row's value compared against the average for its group |
| Fitted transformer | An object that learns from training data, stores what it learned, and applies it later |
| Training and serving skew | Model trained on one set of transformations and fed another in production. Nothing errors; the predictions are just wrong |
| Ablation | Removing part of a system on purpose to measure what it was contributing |
| Pre-registered decision | A rule written down before the result is seen, so the conclusion cannot be fitted to the data |
| Early stopping | Halting training when the validation metric stops improving |
| Expanding-window CV | Folds where each trains on more history than the last and is scored on the period straight after |
| MLflow run | One training attempt with its settings, metrics, and files recorded |
| MLflow alias | A movable pointer to a model version, such as `candidate` or `production`. Replaces the deprecated stages |
| SHAP | Explains how much each feature pushed one prediction away from the average. **Mean absolute SHAP hides rare-but-decisive features**, so max absolute SHAP is recorded too |
| DVC | Versions large data files alongside code, keeping a fingerprint in Git and the data elsewhere |
| Drift | When live data slowly stops resembling training data, so the model quietly gets worse |
| PSI | Population Stability Index. Buckets the reference distribution, then measures how much the new data's shares moved. Under 0.10 stable, over 0.25 investigate |
| KS statistic | The largest gap between two cumulative distribution curves, from 0 to 1. The p-value is ignored: at 100,000 rows everything is significant |
| Importance-weighted drift | PSI weighted by SHAP importance, so drift in a feature the model ignores contributes nothing |
| Promotion gate | A check a model must pass before its alias is allowed to move to production |
| Unit test | A small piece of code that runs your real code with a known input and checks the answer |
| Fixture | A pytest function that builds something tests need, provided fresh to each test |
| CI | Automated checks that run on a clean machine every time code is pushed |
| Linter | A tool that reads code without running it and flags likely mistakes. We use ruff |
| Formatter | A tool that rewrites code into one consistent style so nobody argues about it. We use black |
| skops | The library MLflow 3 uses to save scikit-learn models safely. It checks types against an allow-list on load |

---

*End of PROJECT_STATE.md. Next: Step 6, Dockerisation and deployment.*
