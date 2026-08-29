# PROJECT_STATE.md

**Last updated:** End of Step 6 of 7
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
| Registered model | `ieee-cis-fraud-detector` version 4, alias `production` |
| Deployment | Hugging Face Spaces, Docker SDK; artefacts on the HF Model Hub |
| Secrets | `.env` only, blocked at commit time by `scripts/check_no_secrets.py` |
| Audience for final artefacts | Hiring managers and portfolio reviewers |

**Headline result:** LightGBM catching 44.6% of fraud cases (31.2% by value) at a 2% review rate. Validation PR-AUC 0.6068 against a 0.0344 baseline. Kaggle private leaderboard 0.9140. Worth roughly $1.76M a year under the documented cost model.

---

## 2. Why this dataset

Card fraud is a rare-event problem with a real cost structure on both sides: a missed fraud is a direct loss, a false alarm blocks a paying customer. Every technical decision has a business consequence you can point at.

- **Enough positive cases.** 20,663 frauds in 590,540 transactions, 3.4990%.
- **Two joinable tables** with partial coverage, so there is real data engineering.
- **Mostly anonymised features.** 339 columns with no published meaning, so structure has to be found empirically.
- **A test set 30 days in the future**, giving Step 5 a genuine distribution shift to detect rather than a manufactured one.

---

## 3. The 7-step plan and current status

| Step | Content | Status |
|------|---------|--------|
| 1 | Dataset acquisition, folder scaffold, GitHub repo, Python environment | **Complete, verified** |
| 2 | EDA and data understanding | **Complete, verified** |
| 3 | Feature engineering and preprocessing pipeline | **Complete, verified** |
| 4 | Model training with MLflow experiment tracking | **Complete, verified** |
| 5 | MLOps layer: tests, CI, drift monitoring, promotion gates | **Complete, one correction pending, see Section 5.9** |
| 6 | Dockerisation and deployment to Hugging Face | **Delivered, awaiting the user's run** |
| 7 | Streamlit dashboard and portfolio packaging | Not started |

---

## 4. Decision log

### Step 1: foundations

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | Dataset is IEEE-CIS Fraud Detection | 20,663 positive cases, a joinable second table, a time-separated test set. |
| D-02 | Repository named `ieee-cis-fraud-detection` | Local folder matches as of Step 3. |
| D-03 | Python 3.11 | Kaggle CLI minimum; stable Windows wheels. Confirmed 3.11.9. |
| D-04 | `venv` plus `requirements.txt`, not conda | One dependency format Docker and GitHub Actions both consume. |
| D-05 | Raw data never committed to Git | 1.3 GB of CSV. Reproducibility from `scripts/download_data.py`. |
| D-06 | DVC deferred to Step 3 | Resolved by D-32. |
| D-07 | Paths resolved dynamically in `config/config.py` | Proven three times: a mismatched folder name, a whole-project move, and now inside a container. |
| D-08 | Branch per step, merged by pull request, tagged after merge | Reviewable trail, triggers CI. |
| D-09 | Public GitHub repository | Portfolio piece, free Actions minutes. |
| D-10 | Dependencies split into runtime and dev, plus a lock file | Later extended with `requirements-ci.txt` and `requirements-serve.txt`. |
| D-11 | Download script shells out to the Kaggle CLI | The CLI is the documented stable contract. |
| D-12 | Interim and processed data as Parquet | Smaller, faster, preserves data types. |
| D-13 | Notebooks for exploration only | Not testable, importable, or reviewable in diffs. |

### Step 2: data understanding

| ID | Decision | Rationale |
|----|----------|-----------|
| D-14 | `run.py` created in Step 2 rather than Step 3 | Two runnable stages existed. |
| D-15 | Test set joined and saved despite having no labels | Became the Step 5 drift input. |
| D-16 | Left join transaction to identity, keep blanks, add `has_identity` | Boosters learn a direction for blanks. Partially revised by D-31. |
| D-17 | Interim data as Parquet with category dtypes preserved | Type work done once. |
| D-18 | `TransactionAmt` stays `float64`; `TransactionID` and `TransactionDT` become `int32` | `float32` is exact for integers only below 16,777,216 and test `TransactionDT` reaches 34,214,345. `float32` also turns 31937.39 into 31937.390625, and the cents are a fraud signal. |
| D-19 | Reference date 30 November 2017 for display only | The competition never published a start date. |
| D-20 | PR-AUC primary, ROC-AUC secondary, recall at a fixed review rate as the business headline, accuracy never | Decided before any numbers existed. Vindicated repeatedly: it settled the LightGBM/XGBoost tie-break without argument, and Step 5's weekly data showed ROC-AUC spanning 0.023 while PR-AUC spanned 0.170 over the same weeks. |
| D-21 | Validation is a time-based split, last 20% by `TransactionDT` | The real test set is 30 days in the future. |
| D-22 | Feature families assigned by rule, unmapped columns reported loudly | Zero unmapped columns in the actual run. |

### Step 3: feature engineering

| ID | Decision | Rationale |
|----|----------|-----------|
| D-23 | Feature engineering is a fitted object saved with joblib, on `BaseEstimator` and `TransformerMixin` | A single transaction at a web service cannot recompute frequency counts. Now proven in the deployed container. |
| D-24 | The transformer is fitted only on the first 80% of the training period | Otherwise each validation row helps compute its own feature. |
| D-25 | Encodings learned from training rows only, never train and test combined | Common in competition write-ups, impossible in production. |
| D-26 | `TransactionDT`, `TransactionID`, and any absolute day counter excluded from features | Test values sit entirely above training values; trees cannot split outside their trained range. |
| D-27 | Near-constant columns dropped at a 99% dominance threshold, with a two-directional rescue rule | Dramatically vindicated. See Section 5.5. |
| D-28 | V columns reduced by correlation clustering inside each of the 15 blocks at 0.75 | 337 to 137, with per-block rates from 18% to 57%. |
| D-29 | The `uid` fingerprint used only for grouping and counting, never as a feature | Given directly, the model memorises individual customers. |
| D-30 | Text columns become integers with a stored mapping; blank gets its own code; unseen maps to -1 | Without a stored mapping the same word gets a different number next month. |
| D-31 | `has_identity` kept, with the confound recorded | **Confirmed:** ranked 270th of 284 with SHAP exactly 0.0. |
| D-32 | DVC with a local folder remote | No account, no internet, no cost. `dvc pull` verified. |
| D-33 | The Streamlit dashboard draws from two sources and reads precomputed artifacts | Delivered as `dashboard_data.json`. |

### Step 4: model training

| ID | Decision | Rationale |
|----|----------|-----------|
| D-34 | A cost model with five stated assumptions, stored in config | Turns an abstract metric into money. |
| D-35 | Costs weighted by the actual transaction amount | Revealed the count-versus-value gap and the CatBoost disagreement. |
| D-36 | Pre-registered uid ablation with a 0.005 PR-AUC threshold set in advance | LightGBM: 0.01289, so kept. CatBoost: below tolerance, so dropped. The same rule gave different answers for different models, which is the rule working. |
| D-37 | Five candidates: dummy, logistic regression, LightGBM, XGBoost, CatBoost | The dummy gave a measured floor that validated the metric code exactly. |
| D-38 | No class weighting and no resampling | We need ordering; weighting shifts probabilities without improving order. |
| D-39 | Category codes treated as ordinary numbers | Frequency counts already supply the information in a form that cannot overfit. |
| D-40 | Early stopping on validation PR-AUC; CV afterwards with the round count fixed | Keeps CV an honest stability check. |
| D-41 | Final model retrained on all labelled data with rounds scaled by the row ratio | 617 scaled 1.25x to 771. |
| D-42 | MLflow tracking URI built with `.as_posix()` | Windows backslashes are unreliable in a SQLAlchemy URL. |
| D-43 | The chosen model registered under an alias, not a stage | MLflow 3 deprecated stages. |
| D-44 | Deploy to Hugging Face Spaces using the Docker SDK | Free, runs the real image, public clickable URL, `/docs` gives an interactive demo. |
| D-45 | The dashboard is built for a hiring manager reading it cold in under two minutes | Loads from precomputed files, leads with the money, one interactive scorer. |
| D-46 | A Kaggle late submission is produced | External validation: private 0.914018. |

### Step 5: the MLOps layer

| ID | Decision | Rationale |
|----|----------|-----------|
| D-47 | Every run tagged `run_mode`; quick runs cannot be registered | Version 1 was a 150-round test model that registered itself. |
| D-48 | Model schemas declare integer columns as floats | JSON has one number type; the API would otherwise be rejected by schema enforcement. |
| D-49 | Feature importance records max absolute SHAP alongside the mean | Mean importance hides rare-but-decisive features. V111 ranked 259th on mean despite a 46% fraud rate on its rare rows. |
| D-50 | Tests run on synthetic data only | The dataset is not in the repository; tests that cannot run in CI do not get run. |
| D-51 | A row-independence test is the primary leakage guard | Also exactly the property the Step 6 API depends on. |
| D-52 | CI installs a light dependency set | ~200 MB instead of 2.5 GB. CI runs in 45 seconds. |
| D-53 | Drift measured with PSI primary, KS secondary, missingness alongside | **Confirmed by `id_31_freq`:** missingness 0.0% in every month while PSI climbed 0.19 to 0.69. A missingness check would have reported nothing. |
| D-54 | The KS statistic is used, never the p-value | At 100,000 rows every difference is significant. |
| D-55 | Feature drift weighted by SHAP importance before becoming a verdict | A raw count fires constantly and gets ignored. |
| D-56 | Promotion runs through six gates and is a separate deliberate command | Gate 1 refused version 1; gates 5 and 6 refused version 2 when the artefacts on disk did not match. |
| D-57 | The monitoring stage writes a small `dashboard_data.json` | 7 KB, so the Step 7 dashboard loads instantly. |
| D-58 | Recall reported by count **and** by value | 44.6% by count against 31.2% by value. |

### Step 6: deployment

| ID | Decision | Rationale |
|----|----------|-----------|
| D-59 | Secrets live only in `.env` and platform secret stores, with a pre-commit hook blocking token-shaped strings | A write token can modify or delete anything in the account. Prevention beats remembering. |
| D-60 | `--experiment` mode: a subset training run never overwrites production artefacts or touches the registry | A single-candidate run wins by default. This is what caused the version 3 incident. |
| D-61 | The selection model stores a fingerprint of its run and is rebuilt when stale | Without it, monitoring silently reported on CatBoost while LightGBM was in production. |
| D-62 | Weekly performance reported and plotted as lift over each period's own baseline | Raw scores looked flat at a 2% decline; lift showed 21%. |
| D-63 | Drift records usable row counts and flags low-confidence values | A PSI of 7.15 on 720 rows is noise. `DRIFT_MIN_ROWS` raised to 1000. |
| D-64 | The API accepts a **raw transaction**, not the 284 engineered features | A caller has a transaction. Making them compute features would put the transformation in two places. |
| D-65 | Artefacts published to the Model Hub and downloaded at container start, not baked into the image | A retrained model ships by restarting the container, not by rebuilding it. |
| D-66 | Image built from `python:3.11-slim`, non-root user, requirements copied before source | Small base, matches how Spaces runs containers, and a code change does not reinstall dependencies. |
| D-67 | The model repository is public, so the Space needs no secrets | A deployment that needs no credential cannot leak one. |
| D-68 | CI builds the image on every push, starts it, and health-checks it, without pushing it | A broken Dockerfile is caught in about a minute. |
| D-69 | Every candidate is trained to convergence, and savings are reported alongside PR-AUC for all of them | CatBoost at 4,000 rounds gained +0.0009. And it is worth more money despite a worse PR-AUC, which should be surfaced rather than hidden. |

---

## 5. Verified results

### 5.1 Raw data, Step 1

| File | Size | Rows | Columns |
|------|------|------|---------|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 |

Fraud rate **3.4990%**, 20,663 of 590,540. 144,233 have an identity record, **24.4%**.

### 5.2 Ingestion, Step 2

| Split | Rows | Columns | Memory before | After | Reduction | Parquet |
|-------|------|---------|---------------|-------|-----------|---------|
| train | 590,540 | 435 | 2,567.7 MB | 927.2 MB | 63.9% | 80.3 MB |
| test | 506,691 | 434 | 2,214.5 MB | 795.2 MB | 64.1% | 69.8 MB |

Runtime 3m 07s.

### 5.3 EDA, Step 2

Train 2017-12-01 to 2018-05-31 (182 days), test 2018-07-01 to 2018-12-30 (183 days), gap **30 days**.

| Group | Transactions | Fraud rate |
|-------|--------------|------------|
| No identity record | 446,307 | 2.0939% |
| Has identity record | 144,233 | 7.8470% |

**The 3.75x figure must never be quoted without its caveat.** Coverage is almost decided by `ProductCD`: W 0%, C 90.8%, H/R/S 99.6%. W has the lowest fraud rate (2.04%) and is 439,670 rows. Restricted to non-W: 7.85% against 5.67%, a lift of **1.39x**. Settled in Step 4 when the model ranked `has_identity` 270th of 284 with SHAP exactly 0.0.

Test identity coverage **28.0%** against training's 24.4%.

| Column | Highest | Lowest |
|--------|---------|--------|
| ProductCD | C at 11.69% | W at 2.04% |
| card6 | credit at 6.68% | debit at 2.43% |
| DeviceType | mobile at 10.17% | missing at 2.10% |
| P_emaildomain | mail.com at 18.96% | aol.com at 2.18% |

### 5.4 The V column blocks, Step 2

All 339 V columns in 15 blocks sharing an identical missing pattern. **Blocks 10 and 14 both sit at 86.12% missing but are different blocks**, so grouping by missing count would have merged them. **Eight of fifteen interleave** through each other's number ranges, so chopping by number range would cut across the real groupings. Reduction ranged from 18% kept (block 14) to 57% (block 2).

### 5.5 Feature engineering, Step 3

Runtime 2m 25s. 435 columns to **284 features**. V columns 337 to 137. Only 2 dropped (V107, V305), 22 rescued.

| Column | Dominant | Share | Rare rows | Fraud rate among rare | Lift |
|--------|----------|-------|-----------|----------------------|------|
| V111 | 1.0 | 99.71% | 1,370 | **46.35%** | 13.2x |
| V113 | 1.0 | 99.65% | 1,645 | 39.51% | 11.2x |
| V112, V108, V117, V119 | 1.0 | 99.5-99.9% | 578-2,431 | 28-31% | 8.0-8.9x |
| Nine `id_` columns | blank | 99.10-99.17% | 3,914-4,273 | ~7.9% | ~2.2x |
| C3 | 0.0 | 99.60% | 1,872 | **0.053%** | **0.015x** |

V111 would have been deleted by any blanket rule. C3 survived only because the rescue triggers in both directions.

**The split:** train 472,432 rows / 16,599 frauds / 3.5135%; valid 118,108 / 4,064 / 3.4409%. Boundary at TransactionDT 12,192,854. Validation window 42 days.

### 5.6 Model training, Step 4, reproduced exactly in Step 5

| Model | PR-AUC | ROC-AUC | Rounds | Time | Savings (42 days) |
|-------|--------|---------|--------|------|-------------------|
| **lightgbm** | **0.60682** | 0.92751 | 617 | 43s | $202,033 |
| xgboost | 0.59907 | **0.93079** | 1,193 | 4m 21s | - |
| catboost | 0.52910 | 0.89368 | ~3,970 of 4,000 | 14m 39s | **$218,263** |
| logistic_regression | 0.18309 | 0.82095 | n/a | 1m 04s | - |
| dummy | 0.03441 | 0.50000 | n/a | 2s | $0 |

**The dummy check passed exactly**: PR-AUC equals the validation fraud rate, ROC-AUC exactly 0.5. That proves the metric code is sound.

**Cross-validation:** 0.61833, 0.63763, **0.67082**, 0.60682. Mean 0.63340, spread 0.02800. Fold 4 has the most training data and the second-worst score, which is a period-difficulty effect and the strongest motivation for monitoring.

**Cost, 42 days:** doing nothing costs $711,534. Within 2% capacity: 44.6% recall, $202,033 saved, **annualised $1,760,894**. Threshold **0.4222493056998478**.

| Review rate | Threshold | Recall | Precision | Savings |
|-------------|-----------|--------|-----------|---------|
| 0.5% | 0.95653 | 13.8% | 94.6% | $57,414 |
| 1.0% | 0.83433 | 26.6% | 91.5% | $114,501 |
| 2.0% | 0.42142 | 44.6% | 76.7% | $202,013 |
| 5.0% | 0.09626 | 64.3% | 44.3% | $339,362 |

**Kaggle:** public 0.944058, private **0.914018**.

**Reproducibility, proven by accident.** Version 4, retrained days after version 2, produced PR-AUC 0.60682, CV spread 0.02800, and threshold 0.4222493056998478 — identical to sixteen decimal places. One seed, no shuffling, no randomness in the feature pipeline. Two independent runs, byte-identical results.

**Top SHAP features:** C13 (0.290), C14 (0.133), TransactionAmt_ratio_to_addr1_mean (0.121), C1 (0.117), V70 (0.115), D15_std_by_uid (0.107), D15_mean_by_uid (0.104), uid_freq (0.103), card1_freq (0.100).

### 5.7 Findings that change how the model is described

**`has_identity` ranked 270th of 284 with SHAP exactly 0.0.** D-31 confirmed.

**Mean SHAP hides rare-but-decisive features.** V111 ranked 259th with mean 0.000091, because it is non-constant on 0.29% of rows, so a 5,000-row sample contains about 14 of them. The rescue rule was right; the measurement was wrong. D-49 records max SHAP from now on.

**The uid family carries 9.9% of total SHAP mass**, four of the top twenty, and is blank on ~82% of test rows. The primary production risk.

**The uid family is 7 features, not 6.** `uid_freq` does not go *missing*; an unseen uid returns 0.0. A feature collapsing onto a constant looks healthy in a missingness check.

**The model catches cheap fraud and misses expensive fraud.**

| Measure | Value |
|---------|-------|
| Total fraud value, 42 days | $609,934 |
| Mean fraud amount | $150.08 |
| Recall by **count** | 44.6% |
| Recall by **value** | **31.2%** |
| Mean caught fraud | **$105** |
| Mean missed fraud | **$186** |

Quoting recall alone overstates the benefit by about 43%.

**PR-AUC and money disagree about which model is better.** CatBoost scores 0.52910 against LightGBM's 0.60682 but saves $218,263 against $202,033, about 8% more, annualising to $1,902,351 against $1,760,894. PR-AUC counts transactions; the cost model weights by amount. Different feature sets (277 against 284) and one 42-day window, so it is a signal to investigate rather than a settled result. LightGBM is kept for training speed (43s against 14m 39s, which matters for the retraining loop) and better cross-validated stability. Q-16 (amount-weighted training) is the experiment that would resolve it.

### 5.8 Step 5 verification

**Tests:** 23 passed in 5.32 seconds. **CI:** green in 45 seconds.

**Promotion gates, all behaving correctly:**

| Version | Model | Outcome |
|---------|-------|---------|
| 1 | LightGBM, quick mode, 150 rounds | **Refused** at gate 1, `run_mode` missing |
| 2 | LightGBM, full | **Refused** at gates 5 and 6, metadata on disk described a different version |
| 3 | CatBoost, 4,000 rounds, uid dropped, 277 features | Not promoted |
| 4 | LightGBM, full, 284 features | **Passed all six**, alias `production` |

### 5.9 Step 5 monitoring: a correction is pending

**The Step 5 monitoring run measured the wrong model.** When it ran, `final_model_metadata.json` described CatBoost, because the CatBoost re-run had overwritten it. So `_get_selection_model` rebuilt a CatBoost and measured everything with it.

Evidence: the drift report covers **277 features not 284**, the threshold is **0.3609 not 0.4222**, and **zero uid features appear anywhere**.

Every number in `drift_summary.md`, `feature_drift.csv`, `score_drift.csv`, and `period_metrics.csv` from that run describes CatBoost, not the production model. D-60 and D-61 prevent a repeat; the fix is to delete `models/selection_model.joblib` and re-run.

**The CatBoost figures, kept for reference until the re-run:**

Weekly held-out performance, seven weeks:

| Week | Rows | Fraud rate | PR-AUC | Lift |
|------|------|-----------|--------|------|
| 04-16 to 04-22 | 7,029 | 4.21% | 0.6697 | 15.90 |
| 04-23 to 04-29 | 18,652 | 2.98% | 0.5367 | 18.04 |
| 04-30 to 05-06 | 22,071 | 3.09% | 0.6403 | **20.75** |
| 05-07 to 05-13 | 20,726 | 3.14% | 0.5517 | 17.59 |
| 05-14 to 05-20 | 20,332 | 3.53% | 0.4995 | 14.16 |
| 05-21 to 05-27 | 19,010 | 4.00% | 0.6275 | 15.70 |
| 05-28 to 06-03 | 10,288 | 3.94% | 0.5068 | **12.87** |

**Raw PR-AUC looked flat, a 2% decline. Lift showed a 21% decline**, because the fraud rate rose from 2.98% to 4.00% and PR-AUC's floor rose with it. D-62 fixes the reporting.

**ROC-AUC spanned 0.9004 to 0.9236 (0.023) while PR-AUC spanned 0.4995 to 0.6697 (0.170)** over the same weeks. Same model, same data. ROC-AUC is nearly blind to what PR-AUC sees clearly.

Monthly drift, verdict WATCH, worst weighted PSI 0.0668:

| Month | Rows | Alert rate | Against expected | Weighted PSI |
|-------|------|-----------|------------------|--------------|
| 2018-07 | 78,430 | 2.92% | 1.47x | 0.0386 |
| 2018-08 | 77,094 | 2.57% | 1.29x | 0.0434 |
| 2018-09 | 71,288 | 2.46% | 1.23x | 0.0385 |
| 2018-10 | 80,677 | 2.02% | 1.01x | 0.0419 |
| 2018-11 | 82,804 | 1.67% | 0.83x | 0.0319 |
| 2018-12 | 116,398 | 2.00% | 1.00x | 0.0668 |

**July is 1.76 times busier than November** at a fixed threshold. Nothing about the model changed; that is entirely the data moving. December also carries 1.49 times the average monthly volume.

**PSI caught exactly what it was built for.** `id_31_freq`, the browser frequency feature, had **0.0% missing in every single month** while PSI climbed 0.189, 0.186, 0.217, 0.303, 0.323, **0.691** and KS climbed to 0.402. Browsers release new versions, so by December many browser strings were unseen in training and the frequency lookup returned 0. A missingness check would have reported nothing wrong in all six months. This is the D-53 rationale confirmed on real data.

**One false alarm identified.** `id_21` showed PSI 7.15, the largest number in the report, on a column 99.1% blank in both periods. That is roughly 720 usable values across ten buckets. The number is measurement noise, not drift, which is why D-63 raises the floor and records row counts. `id_13`, by contrast, swings genuinely from 57% to 83% missing on hundreds of thousands of rows and deserves attention.

---

## 6. Current repository structure

```
ieee-cis-fraud-detection/
│
├── .dvc/ , .dvcignore                                              # Step 3
├── .github/workflows/ci.yml            # quality + docker jobs      Step 5,6
├── .pre-commit-config.yaml             # + secret blocking          Step 5,6
├── .dockerignore                                                   # Step 6
├── Dockerfile                          # at root, Spaces needs it   Step 6
├── .vscode/settings.json
├── app/                                # empty                     (Step 7)
│
├── config/config.py                    # extended in Steps 2,3,4,5,6
│
├── data/
│   ├── raw/                            # git-ignored, 1.29 GB
│   ├── interim/                        # git-ignored, 2 Parquet files
│   ├── processed/                      # DVC-tracked
│   │   ├── train_features.parquet      # 84.4 MB
│   │   ├── test_features.parquet       # 68.5 MB
│   │   └── kaggle_submission.csv       # 14.6 MB
│   └── external/
│
├── deploy/space/README.md              # Space config YAML         (Step 6)
├── docker/docker-compose.yml                                       # Step 6
│
├── docs/
│   ├── PROJECT_STATE.md                # this file
│   └── steps/step1.md ... step6.md
│
├── models/                             # git-ignored except metadata
│   ├── feature_engineer.joblib         # 28.0 MB
│   ├── final_model.joblib              #  5.3 MB
│   ├── selection_model.joblib          # fingerprinted             (Step 6)
│   └── final_model_metadata.json       # committed
│
├── reports/
│   ├── eda_summary.md , feature_summary.md , training_summary.md
│   ├── column_profile.csv , missing_profile.csv
│   ├── v_column_missing_groups.csv , v_column_reduction.csv
│   ├── feature_manifest.csv , dropped_columns.csv
│   ├── model_comparison.csv , threshold_analysis.csv
│   ├── cost_curve.csv , cv_results.csv , feature_importance.csv
│   ├── monitoring/
│   │   ├── feature_drift.csv , score_drift.csv , period_metrics.csv
│   │   ├── drift_summary.md
│   │   └── dashboard_data.json         # 7 KB, feeds Step 7
│   ├── figures/                        # 19 PNG charts
│   └── explainability/                 # 3 SHAP charts
│
├── scripts/
│   ├── download_data.py , verify_data.py                           # Step 1
│   ├── promote_model.py                                            # Step 5
│   ├── check_no_secrets.py                                         # Step 6
│   ├── publish_model.py                                            # Step 6
│   └── deploy_space.py                                             # Step 6
│
├── src/
│   ├── features/engineer.py                                        # Step 3
│   ├── models/candidates.py                                        # Step 4
│   ├── monitoring/drift.py , promotion.py                          # Step 5
│   ├── pipelines/
│   │   ├── ingestion.py , eda.py , features.py                     # Steps 2,3
│   │   ├── training.py                                             # Step 4
│   │   └── monitoring.py                                           # Step 5
│   ├── serving/
│   │   ├── schemas.py , artifacts.py , scoring.py , app.py         # Step 6
│   └── utils/
│       ├── memory_utils.py , ingestion_utils.py , eda_utils.py     # Step 2
│       ├── column_selection.py , feature_utils.py                  # Step 3
│       ├── metrics.py , mlflow_utils.py , model_plots.py           # Step 4
│       └── monitoring_plots.py                                     # Step 5
│
├── tests/
│   ├── conftest.py , test_metrics.py , test_feature_engineer.py    # Step 5
│   ├── test_leakage.py , test_drift.py                             # Step 5
│   └── test_serving.py                                             # Step 6
│
├── .env                                # git-ignored, holds HF_TOKEN
├── .env.example , .gitignore , LICENSE , README.md
├── pyproject.toml
├── requirements.txt , requirements-dev.txt , requirements.lock.txt
├── requirements-ci.txt                                             # Step 5
├── requirements-serve.txt                                          # Step 6
├── mlflow.db                           # git-ignored
└── run.py                              # 5 stages
```

---

## 7. Environment

| Item | Value |
|------|-------|
| Python | **3.11.9** |
| Environment | `.venv`, git-ignored |
| Rebuild exactly | `pip install -r requirements.lock.txt` |

### 7.1 Confirmed library versions

| Library | Version | Notes that shaped the code |
|---------|---------|----------------------------|
| pandas | 2.3.3 | `observed=True` on category groupbys; empty-frame concat deprecated |
| numpy | 2.4.6 | numpy 2.x, so `np.NaN` and `np.float_` do not exist |
| pyarrow | 24.0.0 | Parquet engine, preserves category dtypes |
| scipy | 1.17.1 | `ks_2samp` for drift |
| scikit-learn | 1.9.0 | `BaseEstimator`, `TransformerMixin`, `Pipeline` |
| lightgbm | 4.7.0 | **The production model.** `eval_set` deprecated for `eval_X`/`eval_y`, handled by inspection. Needs `libgomp1` in a slim container |
| xgboost | 3.2.0 | Early stopping in the constructor |
| catboost | 1.2.10 | Plateaued at 0.529 with a 4,000 round budget |
| mlflow | 3.15.1 | Aliases not stages; `log_model` uses `name`; URI needs forward slashes; sklearn flavor saves via skops and needs `skops_trusted_types=["numpy.dtype"]` |
| shap | 0.51.0 | Mean absolute SHAP hides rare features, see D-49 |
| fastapi | 0.141.1 | The service. `/docs` generated from the Pydantic schemas |
| uvicorn | 0.52.3 | Serves on port 7860 in the container |
| pydantic | 2.13.4 | Request validation |
| huggingface_hub | 0.25+ | Model Hub upload and download, Space deployment |
| streamlit | 1.61.1 | Step 7 |
| pytest | 9.1.1 | 23 tests, 5.32s |
| ruff, black, pre-commit | 0.16.3, 26.5.1, 4.6.2 | CI green in 45s |
| dvc | 3.55+ | Local folder remote, verified |

---

## 8. The cost model

**These are stated assumptions, not figures from a business.** All five live in `config/config.py`.

| Assumption | Value | Reasoning |
|------------|-------|-----------|
| Analyst review | $4.00 per case | ~$60k/year fully loaded is ~$29/hour; a five minute review is $2.40; rounded up |
| Chargeback fee | $25.00 per missed fraud | Card networks charge $15 to $40 per dispute |
| False alarm friction | $1.00 | The softest number, the first to replace |
| Fraud recovered when caught | 90% | Reviews take time, some are judged wrongly |
| Review capacity | 2% of transactions | Roughly one analyst's full shift at this volume |

Missed fraud costs the amount plus the fee; caught fraud costs a review plus the 10% not recovered; a false alarm costs a review plus friction. Costs are weighted by the real transaction amount, computed exactly at every threshold via cumulative sums.

**Framing for the PM track:** an order of magnitude under stated assumptions, with sensitivity to each, never a forecast. Always pair count-based recall with value-based recall (D-58). Report savings alongside PR-AUC and say so when they disagree (D-69).

---

## 9. Conventions in force

**Code**
- All paths come from `config/config.py`. No module builds its own.
- One random seed, `RANDOM_SEED = 42`. Proven to give byte-identical reruns.
- Each pipeline stage reads a file and writes a file.
- `pipelines/` orders stages, `features/` builds features, `models/` defines candidates, `monitoring/` watches, `serving/` answers requests, `utils/` supports all of them.
- Anything learned from data is learned from training rows only, then applied unchanged.
- Matplotlib uses `Agg`, set before pyplot is imported.
- Decision rules that use results are written down before the results are seen.
- When a library is mid-transition, inspect what is installed rather than assuming. Used four times: MLflow `name`/`artifact_path`, LightGBM `eval_set`/`eval_X`, skops trusted types, and `skops_trusted_types` being accepted at all.
- Secrets never appear in a source file, a notebook, a terminal command, or a chat. `.env` only.

**Git**
- Branch `step-NN-short-description`; commits `type: message`; squash-merge by pull request; tag `v0.N.0-stepN`
- `main` must always work, and CI must be green before merging
- Formatting changes go in their own commit

**Documentation**
- No em dashes; plain vocabulary; explanation before every code block
- Every file created is stated with full contents and its reason
- When an earlier claim turns out to be wrong, it is corrected openly in the next step

**Corrections made so far**
1. Parquet size estimate was 4x too high (Step 3)
2. Memory reduction range was slightly optimistic (Step 3)
3. The 3.75x identity finding was confounded by `ProductCD` (Step 3)
4. The nine `id_` columns were predicted to be dropped; all nine were rescued (Step 4)
5. The uid family is 7 features, not 6 (Step 5)
6. Mean absolute SHAP was the wrong tool for judging rare-but-decisive features (Step 5)
7. The README stated the 42-day saving as an annual figure (Step 5)
8. The `--models catboost` command I gave was destructive and overwrote production artefacts (Step 6)
9. Raw weekly PR-AUC hid a 21% decline because its floor moves with the fraud rate (Step 6)
10. `DRIFT_MIN_ROWS = 500` was too low and produced a PSI of 7.15 that was pure noise (Step 6)

---

## 10. Completed

### Steps 1 to 4, verified
- [x] Data downloaded and verified, repo live, environment locked, tagged `v0.1.0-step1`
- [x] Ingestion and EDA run, 15 V blocks found, tagged `v0.2.0-step2`
- [x] Feature stage run, 284 features, all checks passed, tagged `v0.3.0-step3`
- [x] Training run: LightGBM 0.60682, Kaggle private 0.914018, tagged `v0.4.0-step4`
- [x] uid ablation ran against its pre-registered rule

### Step 5, verified
- [x] 23 tests passing, CI green in 45 seconds
- [x] Monitoring stage ran, six months of drift, seven weeks of performance
- [x] Promotion gates refused versions 1 and 2, passed version 4
- [x] Version 4 promoted to `production`
- [x] `id_31_freq` confirmed D-53 on real data
- [ ] Monitoring re-run against the correct model, see Section 5.9

### Step 6, delivered
- [x] Token exposure identified; revocation and secret handling specified
- [x] `scripts/check_no_secrets.py` and the pre-commit hook specified
- [x] Six pre-work fixes specified (D-60 to D-63, D-69, plus the monitoring re-run)
- [x] `src/serving/` four modules specified
- [x] `Dockerfile`, `.dockerignore`, `docker-compose.yml` specified
- [x] `scripts/publish_model.py` and `scripts/deploy_space.py` specified
- [x] CI extended with a Docker build and health check
- [x] README updates specified
- [ ] Old token revoked by the user
- [ ] Pre-work fixes applied and monitoring re-run
- [ ] Service tested locally and in a container
- [ ] Artefacts published to the Model Hub
- [ ] Space deployed and reachable
- [ ] Branch merged and tagged `v0.6.0-step6`

---

## 11. Pending

**Step 7**
- Streamlit dashboard per D-33 and D-45, reading `dashboard_data.json`
- Five sections: headline and business impact, model performance, a live scorer calling the deployed API, drift monitoring, how it was built
- The count-versus-value distinction presented so nobody misreads the recall figure
- Architecture diagram covering Kaggle to deployed container
- The dashboard deployed as a second Space
- Final README with every link verified
- Portfolio walkthrough script and the PM track pack

---

## 12. Open questions

| # | Question | Needed by | Status |
|---|----------|-----------|--------|
| Q-01 to Q-14 | Earlier questions | - | **All answered.** See Sections 5 and 8. |
| Q-15 | Hugging Face account and token | Step 6 | **Answered.** Account created. The first token was exposed in conversation and must be revoked; a replacement goes in `.env` only. |
| Q-16 | Should the model be retrained with amount-weighted examples? | Future | **Open and now the highest-value experiment left.** It catches 44.6% of fraud by count but 31.2% by value, and CatBoost beats it on money despite a worse PR-AUC. Amount weighting aims to give LightGBM CatBoost's value-sensitivity without giving up its ranking. |
| Q-17 | Should CatBoost be re-run with a larger budget? | Optional | **Answered.** 4,000 rounds gained +0.0009 over 1,500. It had plateaued, not been cut short. LightGBM wins on ranking by a wide margin. |
| Q-18 | Should model selection use savings rather than PR-AUC? | Future | Open. CatBoost saves ~8% more despite a worse PR-AUC. One 42-day window and different feature sets, so not settled. D-69 makes savings visible for every candidate on every future run so the question stays in view. |
| Q-19 | Does the Step 7 dashboard call the live API, or read precomputed results only? | Step 7 | Open. Calling the API makes the demo real but adds a dependency: if the Space sleeps, the dashboard breaks. Recommendation is to call it with a precomputed fallback. |

---

## 13. How to resume from nothing

```powershell
# 1. Clone and enter
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

# 2. Recreate the environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt

# 3. Secrets: copy the template and fill in your own values
Copy-Item .env.example .env
# then edit .env

# 4. Confirm the code is sound before trusting anything it produces
pytest

# 5. Rebuild the raw data (needs a Kaggle account that joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# 6. Rebuild everything downstream
python run.py --step all

# 7. Or, if the DVC remote survived, pull the processed data directly
dvc pull

# 8. Or skip training entirely and pull the model from the Hub
python -c "from src.serving.artifacts import load_artifacts; load_artifacts()"

# 9. Read the current state
code docs/PROJECT_STATE.md
```

---

## 14. Glossary

| Term | Plain meaning |
|------|---------------|
| Parquet | A file format storing tables column by column. Smaller and faster than CSV, and it remembers data types |
| Class imbalance | When one outcome is far rarer than the other, here 3.5% fraud |
| Accuracy | Share of predictions correct. Useless here: always predicting "not fraud" scores 96.5% |
| Precision | Of the transactions you flagged, the share that really were fraud |
| Recall | Of all the fraud that occurred, the share you caught. Report by count **and** by value |
| PR-AUC | Precision-Recall Area Under Curve. Primary metric. Its floor is the fraud rate of whatever you measure, so compare **lift** across periods, not raw scores |
| ROC-AUC | Probability a random fraud scores above a random legitimate transaction. The Kaggle metric. Far less sensitive here than PR-AUC |
| Lift | A score divided by its baseline. Makes periods with different fraud rates comparable |
| Time-based split | Train on earlier data, validate on later data |
| Data leakage | Information unavailable at prediction time influencing training |
| Confounded comparison | A difference between two groups that is really a difference in what those groups contain |
| Frequency encoding | Replacing a category with how often it appeared in training. Unseen values get 0 |
| Fitted transformer | An object that learns from training data, stores it, and applies it later |
| Training and serving skew | Model trained on one set of transformations and fed another. Nothing errors; predictions are just wrong |
| Row independence | Transforming one row gives the same answer as transforming a batch containing it. What makes single-transaction scoring safe |
| Ablation | Removing part of a system on purpose to measure what it contributed |
| Pre-registered decision | A rule written down before the result is seen |
| Expanding-window CV | Folds where each trains on more history and is scored on the period straight after |
| MLflow alias | A movable pointer to a model version, such as `candidate` or `production` |
| SHAP | How much each feature pushed one prediction away from the average. Mean absolute SHAP hides rare-but-decisive features; max absolute SHAP does not |
| DVC | Versions large data files alongside code, keeping a fingerprint in Git |
| Drift | When live data stops resembling training data, so the model quietly gets worse |
| PSI | Population Stability Index. Buckets the reference distribution and measures how much the new data's shares moved. Under 0.10 stable, over 0.25 investigate. Catches a collapse onto one value that a missingness check cannot see |
| KS statistic | The largest gap between two cumulative curves, 0 to 1. The p-value is ignored: at 100,000 rows everything is significant |
| Promotion gate | A check a model must pass before its alias moves to production |
| Unit test | Code that runs your real code with a known input and checks the answer |
| CI | Automated checks on a clean machine every time code is pushed |
| Container | A sealed box holding an operating system, a Python, the libraries, and your code, so it runs identically everywhere |
| Image against container | The image is the recipe, built once. A container is a running instance of it |
| Docker layer caching | Each Dockerfile instruction becomes a cached layer. Copy requirements before source, so a code change does not reinstall dependencies |
| API | A service something else can call over the network. FastAPI generates an interactive `/docs` page from the code |
| Model Hub | Where the trained artefacts live, versioned separately from the code, so a new model ships by restarting rather than rebuilding |
| Write token | A credential that can modify or delete anything in an account. Lives in `.env`, never in a file, a terminal, or a chat |

---

*End of PROJECT_STATE.md. Next: Step 7, the dashboard and portfolio packaging.*
