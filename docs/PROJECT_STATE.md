# PROJECT_STATE.md

**Last updated:** End of Step 7 of 7, project complete
**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Live API:** https://ieee-cis-fraud-detection.onrender.com/docs
**Model artefacts:** https://huggingface.co/Dee-ui/ieee-cis-fraud-detector
**Local path:** `C:\Users\Dauda Agbonoga\Documents\Projects\ieee-cis-fraud-detection`

---

## 0. What this document is

The anchor for the whole project. Rewritten in full at the end of every step, never patched with a diff.

If everything else is lost, this file alone is enough to understand what was built, why every choice was made, what was verified, and what remains open.

---

## 1. Project at a glance

| Item | Value |
|------|-------|
| Goal | A complete, portfolio-grade fraud detection system covering the full machine learning and MLOps lifecycle |
| Dataset | IEEE-CIS Fraud Detection (Kaggle, data provided by Vesta Corporation) |
| Delivery | 7 steps, each with its own markdown guide plus a refreshed copy of this file |
| Platform | Windows, VS Code, PowerShell, Python 3.11.9 |
| Machine | Intel Core Ultra 7 265H, 32 GB RAM |
| Version control | Git and GitHub for code, DVC with a local folder remote for processed data |
| DVC remote | `C:\Users\Dauda Agbonoga\dvcstore`, verified |
| Experiment tracking | MLflow 3.15.1, SQLite backend |
| Production model | `ieee-cis-fraud-detector` version 4, alias `production` |
| API host | Render free tier, Docker runtime |
| Artefact host | Hugging Face Model Hub, public |
| Dashboard host | See Q-20 |
| Secrets | `.env` only, blocked at commit time by `scripts/check_no_secrets.py` |

**Headline result:** LightGBM catching 44.6% of fraud cases (31.2% by value) at a 2% review rate. Validation PR-AUC 0.6068 against a 0.0344 baseline. Kaggle private leaderboard 0.9140. Worth roughly $1.76M a year under the documented cost model.

---

## 2. Why this dataset

Card fraud is a rare-event problem with a real cost structure on both sides: a missed fraud is a direct loss, a false alarm blocks a paying customer. Every technical decision has a business consequence you can point at.

- **Enough positive cases.** 20,663 frauds in 590,540 transactions, 3.4990%.
- **Two joinable tables** with partial coverage, so there is real data engineering.
- **Mostly anonymised features.** 339 columns with no published meaning, so structure has to be found empirically.
- **A test set 30 days in the future**, giving genuine distribution shift to detect rather than a manufactured one.

---

## 3. The 7-step plan, all complete

| Step | Content | Status |
|------|---------|--------|
| 1 | Dataset acquisition, folder scaffold, GitHub repo, Python environment | **Complete, verified** |
| 2 | EDA and data understanding | **Complete, verified** |
| 3 | Feature engineering and preprocessing pipeline | **Complete, verified** |
| 4 | Model training with MLflow experiment tracking | **Complete, verified** |
| 5 | MLOps layer: tests, CI, drift monitoring, promotion gates | **Complete, verified** |
| 6 | Dockerisation and deployment | **Complete, verified** (pivoted from Spaces to Render) |
| 7 | Streamlit dashboard and portfolio packaging | **Delivered, awaiting the user's run** |

---

## 4. Decision log

### Step 1: foundations

| ID | Decision | Rationale |
|----|----------|-----------|
| D-01 | Dataset is IEEE-CIS Fraud Detection | 20,663 positive cases, a joinable second table, a time-separated test set. |
| D-02 | Repository named `ieee-cis-fraud-detection` | Local folder matches as of Step 3. |
| D-03 | Python 3.11 | Kaggle CLI minimum; stable Windows wheels. Confirmed 3.11.9. |
| D-04 | `venv` plus `requirements.txt`, not conda | One format Docker and GitHub Actions both consume. |
| D-05 | Raw data never committed to Git | 1.3 GB of CSV. Reproducibility from `scripts/download_data.py`. |
| D-06 | DVC deferred to Step 3 | Resolved by D-32. |
| D-07 | Paths resolved dynamically in `config/config.py` | Proven four times: a mismatched folder name, a whole-project move, inside a container, and on Render. |
| D-08 | Branch per step, merged by pull request, tagged after merge | Reviewable trail, triggers CI. |
| D-09 | Public GitHub repository | Portfolio piece, free Actions minutes. |
| D-10 | Dependencies split into runtime and dev, plus a lock file | Later extended with `requirements-ci.txt`, `requirements-serve.txt`, `app/requirements.txt`. |
| D-11 | Download script shells out to the Kaggle CLI | The CLI is the documented stable contract. |
| D-12 | Interim and processed data as Parquet | Smaller, faster, preserves data types. |
| D-13 | Notebooks for exploration only | Not testable, importable, or reviewable in diffs. |

### Step 2: data understanding

| ID | Decision | Rationale |
|----|----------|-----------|
| D-14 | `run.py` created in Step 2 rather than Step 3 | Two runnable stages existed. |
| D-15 | Test set joined and saved despite having no labels | Became the drift input, showing genuine shift. |
| D-16 | Left join transaction to identity, keep blanks, add `has_identity` | Boosters learn a direction for blanks. Partially revised by D-31. |
| D-17 | Interim data as Parquet with category dtypes preserved | Type work done once. |
| D-18 | `TransactionAmt` stays `float64`; `TransactionID` and `TransactionDT` become `int32` | `float32` is exact for integers only below 16,777,216 and test `TransactionDT` reaches 34,214,345. `float32` also turns 31937.39 into 31937.390625, and the cents are a fraud signal. |
| D-19 | Reference date 30 November 2017 for display only | The competition never published a start date. |
| D-20 | PR-AUC primary, ROC-AUC secondary, recall at a fixed review rate as the business headline, accuracy never | Decided before any numbers existed. Vindicated three times: it settled the LightGBM/XGBoost tie-break, and across the monitored weeks ROC-AUC spanned 0.032 while PR-AUC spanned 0.165. |
| D-21 | Validation is a time-based split, last 20% by `TransactionDT` | The real test set is 30 days in the future. |
| D-22 | Feature families assigned by rule, unmapped columns reported loudly | Zero unmapped columns. |

### Step 3: feature engineering

| ID | Decision | Rationale |
|----|----------|-----------|
| D-23 | Feature engineering is a fitted object saved with joblib, on `BaseEstimator` and `TransformerMixin` | Proven in production: the deployed container loads it and scores single transactions. |
| D-24 | The transformer is fitted only on the first 80% of the training period | Otherwise each validation row helps compute its own feature. |
| D-25 | Encodings learned from training rows only, never train and test combined | Common in competition write-ups, impossible in production. |
| D-26 | `TransactionDT`, `TransactionID`, and any absolute day counter excluded from features | Test values sit entirely above training values. |
| D-27 | Near-constant columns dropped at a 99% dominance threshold, with a two-directional rescue rule | Dramatically vindicated. See Section 5.5. |
| D-28 | V columns reduced by correlation clustering inside each of the 15 blocks at 0.75 | 337 to 137, per-block rates from 18% to 57%. |
| D-29 | The `uid` fingerprint used only for grouping and counting, never as a feature | Given directly, the model memorises individual customers. |
| D-30 | Text columns become integers with a stored mapping; blank gets its own code; unseen maps to -1 | Without it the same word gets a different number next month. |
| D-31 | `has_identity` kept, with the confound recorded | **Confirmed:** ranked 270th of 284, SHAP exactly 0.0. |
| D-32 | DVC with a local folder remote | No account, no internet, no cost. `dvc pull` verified. |
| D-33 | The dashboard draws from precomputed artifacts, not the raw tables | Delivered as `dashboard_data.json`. |

### Step 4: model training

| ID | Decision | Rationale |
|----|----------|-----------|
| D-34 | A cost model with five stated assumptions, stored in config | Turns an abstract metric into money. |
| D-35 | Costs weighted by the actual transaction amount | Revealed the count-versus-value gap and the CatBoost disagreement. |
| D-36 | Pre-registered uid ablation with a 0.005 threshold set in advance | LightGBM kept them (0.01289), CatBoost dropped them. Same rule, different answers, which is the rule working. |
| D-37 | Five candidates including a dummy and a linear baseline | The dummy gave a measured floor that validated the metric code exactly. |
| D-38 | No class weighting and no resampling | We need ordering; weighting shifts probabilities without improving order. |
| D-39 | Category codes treated as ordinary numbers | Frequency counts already supply the information in a form that cannot overfit. |
| D-40 | Early stopping on validation PR-AUC; CV afterwards with the round count fixed | Keeps CV an honest stability check. |
| D-41 | Final model retrained on all labelled data with rounds scaled by the row ratio | 617 scaled 1.25x to 771. |
| D-42 | MLflow tracking URI built with `.as_posix()` | Windows backslashes are unreliable in a SQLAlchemy URL. |
| D-43 | The chosen model registered under an alias, not a stage | MLflow 3 deprecated stages. |
| D-44 | Deploy the API as a container | **Revised by D-70.** Originally Hugging Face Spaces; now Render. |
| D-45 | The dashboard is built for a hiring manager reading it cold in under two minutes | Loads from one precomputed file, leads with the money, one interactive element. |
| D-46 | A Kaggle late submission is produced | External validation: private 0.914018. |

### Step 5: the MLOps layer

| ID | Decision | Rationale |
|----|----------|-----------|
| D-47 | Every run tagged `run_mode`; quick runs cannot be registered | Version 1 was a 150-round test model that registered itself. |
| D-48 | Model schemas declare integer columns as floats | JSON has one number type; the API would otherwise be rejected. |
| D-49 | Feature importance records max absolute SHAP alongside the mean | Mean importance hides rare-but-decisive features. |
| D-50 | Tests run on synthetic data only | Tests that cannot run in CI do not get run. |
| D-51 | A row-independence test is the primary leakage guard | Also exactly the property the API depends on. |
| D-52 | CI installs a light dependency set | CI runs in 45 seconds. |
| D-53 | Drift measured with PSI primary, KS secondary, missingness alongside | **Confirmed twice on real data.** See Section 5.9. |
| D-54 | The KS statistic is used, never the p-value | At 100,000 rows every difference is significant. |
| D-55 | Feature drift weighted by SHAP importance before becoming a verdict | A raw count fires constantly and gets ignored. |
| D-56 | Promotion runs through six gates and is a separate deliberate command | Refused versions 1 and 2, passed version 4. |
| D-57 | The monitoring stage writes a small `dashboard_data.json` | 8 KB. |
| D-58 | Recall reported by count **and** by value | 44.6% against 31.2%. |

### Step 6: deployment

| ID | Decision | Rationale |
|----|----------|-----------|
| D-59 | Secrets live only in `.env` and platform secret stores, with a pre-commit hook blocking token-shaped strings | A write token was exposed in conversation and revoked. Prevention beats remembering. |
| D-60 | `--experiment` mode: a subset training run never overwrites production artefacts | Caused the version 3 incident. |
| D-61 | The selection model stores a fingerprint of its run and is rebuilt when stale | Without it, monitoring reported on CatBoost while LightGBM was in production. |
| D-62 | Weekly performance reported as lift over each period's own baseline | Raw scores showed a 3% decline; lift showed 21%. |
| D-63 | Drift records usable row counts and flags low-confidence values | A PSI of 7.15 on 720 rows was noise. `DRIFT_MIN_ROWS` raised to 1000. |
| D-64 | The API accepts a **raw transaction**, not the 284 engineered features | A caller has a transaction, not features. |
| D-65 | Artefacts published to the Model Hub and downloaded at container start | A retrained model ships by restarting, not rebuilding. |
| D-66 | Image built from `python:3.11-slim`, non-root, requirements copied before source | 1.27 GB, and a code change does not reinstall dependencies. |
| D-67 | The model repository is public, so the service needs no secrets | A deployment that needs no credential cannot leak one. |
| D-68 | CI builds the image on every push, starts it, and health-checks it | A broken Dockerfile is caught in about a minute. |
| D-69 | Every candidate trained to convergence, savings reported alongside PR-AUC | CatBoost at 4,000 rounds gained +0.0009, and is worth more money despite a worse ranking. |

### Step 7: deployment pivot and packaging

| ID | Decision | Rationale |
|----|----------|-----------|
| D-70 | **The API runs on Render, not Hugging Face Spaces.** The Model Hub remains the artefact source | Spaces returned `402 Payment Required`: Docker and Gradio Spaces now need a PRO subscription on free hardware. Render's free tier runs the same image. **Revises D-44.** |
| D-71 | The container's port comes from `$PORT` with a 7860 fallback | Render injects its own port. The fallback keeps local runs and any future Spaces use working unchanged. Verified with `docker run -e PORT=9000`. |
| D-72 | Hosting free tiers are verified before code is written against them | The Spaces failure arrived at the end of a purpose-built deploy script. Platform pricing is not a fact to assert, it is a thing to check. |
| D-73 | The dashboard reads one committed JSON bundle and calls one API, with a cached fallback | Per D-45 it must open in under three seconds, and a sleeping free-tier API must not leave the page broken. |

---

## 5. Verified results

### 5.1 Raw data

| File | Size | Rows | Columns |
|------|------|------|---------|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 |

Fraud rate **3.4990%**, 20,663 of 590,540. 144,233 have an identity record, **24.4%**.

### 5.2 Ingestion

| Split | Rows | Columns | Memory before | After | Reduction | Parquet |
|-------|------|---------|---------------|-------|-----------|---------|
| train | 590,540 | 435 | 2,567.7 MB | 927.2 MB | 63.9% | 80.3 MB |
| test | 506,691 | 434 | 2,214.5 MB | 795.2 MB | 64.1% | 69.8 MB |

Runtime 3m 07s.

### 5.3 EDA

Train 2017-12-01 to 2018-05-31 (182 days), test 2018-07-01 to 2018-12-30 (183 days), gap **30 days**.

| Group | Transactions | Fraud rate |
|-------|--------------|------------|
| No identity record | 446,307 | 2.0939% |
| Has identity record | 144,233 | 7.8470% |

**The 3.75x figure must never be quoted without its caveat.** Coverage is almost decided by `ProductCD`: W 0%, C 90.8%, H/R/S 99.6%. W has the lowest fraud rate (2.04%) and is 439,670 rows. Restricted to non-W: 7.85% against 5.67%, a lift of **1.39x**. Settled when the model ranked `has_identity` 270th of 284 with SHAP exactly 0.0.

Test identity coverage **28.0%** against training's 24.4%.

| Column | Highest | Lowest |
|--------|---------|--------|
| ProductCD | C at 11.69% | W at 2.04% |
| card6 | credit at 6.68% | debit at 2.43% |
| DeviceType | mobile at 10.17% | missing at 2.10% |
| P_emaildomain | mail.com at 18.96% | aol.com at 2.18% |

### 5.4 The V column blocks

All 339 V columns in 15 blocks sharing an identical missing pattern. **Blocks 10 and 14 both sit at 86.12% missing but are different blocks**, so grouping by missing count would have merged them. **Eight of fifteen interleave** through each other's number ranges, so chopping by number range would cut across the real groupings. Reduction ranged from 18% kept (block 14) to 57% (block 2).

### 5.5 Feature engineering

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

### 5.6 Model training

| Model | PR-AUC | ROC-AUC | Rounds | Time | Savings (42 days) |
|-------|--------|---------|--------|------|-------------------|
| **lightgbm** | **0.60682** | 0.92751 | 617 | 43s | $202,033 |
| xgboost | 0.59907 | **0.93079** | 1,193 | 4m 21s | n/a |
| catboost | 0.52910 | 0.89368 | ~3,970 of 4,000 | 14m 39s | **$218,263** |
| logistic_regression | 0.18309 | 0.82095 | n/a | 1m 04s | n/a |
| dummy | 0.03441 | 0.50000 | n/a | 2s | $0 |

**The dummy check passed exactly**: PR-AUC equals the validation fraud rate, ROC-AUC exactly 0.5.

**Cross-validation:** 0.61833, 0.63763, **0.67082**, 0.60682. Mean 0.63340, spread 0.02800. Fold 4 has the most training data and the second-worst score, a period-difficulty effect.

**Cost, 42 days:** doing nothing costs $711,534. Within 2% capacity: 44.6% recall, $202,013 saved, **annualised $1,760,894**. Threshold **0.4222493056998478**.

| Review rate | Threshold | Recall | Precision | Savings |
|-------------|-----------|--------|-----------|---------|
| 0.5% | 0.95653 | 13.8% | 94.6% | $57,414 |
| 1.0% | 0.83433 | 26.6% | 91.5% | $114,501 |
| 2.0% | 0.42142 | 44.6% | 76.7% | $202,013 |
| 5.0% | 0.09626 | 64.3% | 44.3% | $339,362 |

**Kaggle:** public 0.944058, private **0.914018**.

**Reproducibility, proven by accident.** Version 4, retrained days after version 2, produced PR-AUC 0.60682, CV spread 0.02800, and threshold 0.4222493056998478, identical to sixteen decimal places.

**Top SHAP:** C13 (0.290), C14 (0.133), TransactionAmt_ratio_to_addr1_mean (0.121), C1 (0.117), V70 (0.115), D15_std_by_uid (0.107), D15_mean_by_uid (0.104), **uid_freq (0.103)**, card1_freq (0.100).

### 5.7 Findings that changed how the model is described

**`has_identity` ranked 270th of 284, SHAP exactly 0.0.** D-31 confirmed.

**Mean SHAP hides rare-but-decisive features.** V111 ranked 259th with mean 0.000091, because it is non-constant on 0.29% of rows, so a 5,000-row sample holds about 14 of them. The rescue rule was right; the measurement was wrong. D-49.

**The uid family carries 9.9% of SHAP mass**, four of the top twenty, and is blank on 89 to 92% of test rows.

**The model catches cheap fraud and misses expensive fraud.** Recall 44.6% by count against **31.2% by value**. Mean caught fraud $105, mean missed $186. Quoting recall alone overstates the benefit by about 43%.

**PR-AUC and money disagree.** CatBoost scores 0.52910 against 0.60682 but saves $218,263 against $202,033. One 42-day window, different feature sets (277 against 284), so a signal to investigate rather than a settled result. LightGBM kept for training speed (43s against 14m 39s, which matters for the retraining loop) and better cross-validated stability.

### 5.8 Step 5 verification

**Tests:** 23 passing in 5.32s, later 29 with the service tests. **CI:** green, quality 30s plus docker 38s, total 1m 14s.

**Promotion gates:**

| Version | Model | Outcome |
|---------|-------|---------|
| 1 | LightGBM, quick mode, 150 rounds | **Refused** at gate 1, `run_mode` missing |
| 2 | LightGBM, full | **Refused** at gates 5 and 6, metadata on disk described another version |
| 3 | CatBoost, 4,000 rounds, uid dropped, 277 features | Not promoted |
| 4 | LightGBM, full, 284 features | **Passed all six**, alias `production` |

### 5.9 Monitoring, corrected run against the production model

`lightgbm, 284 features, threshold 0.4222`. Runtime 3m 57s. Verdict **WATCH**.

**Weekly held-out performance:**

| Week | Rows | Fraud rate | PR-AUC | Lift |
|------|------|-----------|--------|------|
| 04-16 to 04-22 | 7,029 | 4.21% | 0.7059 | 16.76 (partial) |
| 04-23 to 04-29 | 18,652 | 2.98% | 0.5931 | 19.93 |
| 04-30 to 05-06 | 22,071 | 3.09% | 0.6819 | **22.10** |
| 05-07 to 05-13 | 20,726 | 3.14% | 0.5689 | 18.14 |
| 05-14 to 05-20 | 20,332 | 3.53% | 0.5414 | 15.35 |
| 05-21 to 05-27 | 19,010 | 4.00% | 0.6497 | 16.25 |
| 05-28 to 06-03 | 10,288 | 3.94% | 0.5461 | 13.87 (partial) |

Full weeks: raw PR-AUC declines **3.1%**, lift declines **21.2%** (20.06 to 15.80). Nearly identical to the CatBoost run's 21%, so the effect belongs to the data, not the model. ROC-AUC spans 0.032 while PR-AUC spans 0.165.

**Monthly drift:**

| Month | Rows | Weighted PSI | Alert rate | Against expected | Drifted (top 20) |
|-------|------|-------------|-----------|------------------|------------------|
| 2018-07 | 78,430 | 0.0816 | 2.83% | 1.42x | 7 (1) |
| 2018-08 | 77,094 | 0.0896 | 2.37% | 1.19x | 9 (1) |
| 2018-09 | 71,288 | 0.0905 | 2.37% | 1.19x | 7 (1) |
| 2018-10 | 80,677 | 0.0955 | 1.99% | 1.00x | 7 (1) |
| 2018-11 | 82,804 | 0.0922 | 1.62% | 0.81x | 6 (1) |
| 2018-12 | 116,398 | **0.1247** | **1.60%** | 0.80x | 13 (1) |

Weighted PSI rises **53%** and now sits at **83% of the 0.15 retrain trigger**. Alert rate falls **43%**; busiest to quietest is **1.77x** at a fixed threshold.

Falling alerts are a symptom, not good news: as the uid features go quiet the model loses signal and its scores regress towards the middle, so fewer clear the threshold.

**The uid prediction, confirmed:**

| Feature | PSI July | PSI December | Missing, training | Missing, test peak |
|---------|----------|--------------|-------------------|--------------------|
| `D15_ratio_to_uid_mean` | 3.088 | **4.135** | 0.340 | 0.910 |
| `uid_freq` | 1.354 | **2.462** | **0.000** | **0.000** |
| `D15_mean_by_uid` | 0.066 | 0.249 | 0.115 | 0.893 |
| `D15_std_by_uid` | 0.081 | 0.173 | 0.296 | 0.916 |
| `TransactionAmt_ratio_to_uid_mean` | 0.545 | 0.648 | 0.000 | 0.889 |

**`uid_freq` is the single feature tripping the top-20 alarm in all six months**, PSI climbing monotonically 1.35 to 2.46 while its missing rate stays at exactly 0.0% throughout. It is the model's 8th most important feature. By December most transactions carry a customer fingerprint that never appeared in training, so the frequency lookup returns 0. The feature has not broken; it has gone quiet.

**This is D-53 confirmed on the exact family flagged in Step 3.** A missing-value check would have reported everything healthy for six months. The chain runs across four steps: Step 3 built the uid and flagged the risk, Step 4's pre-registered ablation kept it with reasons, Step 5's PSI was designed for this failure mode, Step 6's corrected run caught it. Each was set up before the answer was known.

`id_13` also drifts genuinely, missingness swinging 57% to 83% on hundreds of thousands of rows.

### 5.10 Deployment

- **API:** https://ieee-cis-fraud-detection.onrender.com, Render free tier, Docker runtime, health check `/health`, auto-deploy on push to `main`
- **Environment:** `HF_MODEL_REPO=Dee-ui/ieee-cis-fraud-detector`, no token needed
- **Image:** 1.27 GB on disk, 299 MB content size
- **CI:** quality 30s, docker 38s, total 1m 14s, green
- **Verified live:** a `/predict` call with `explain: true` returned probability 0.0156, threshold 0.4222493056998478, decision `pass`, model version 4, and a SHAP breakdown led by C13 at +0.282

**The Spaces failure.** `scripts/deploy_space.py` returned `402 Payment Required`: Docker and Gradio Spaces now require a PRO subscription on free hardware, and only static Spaces are unconditionally free. The pivot to Render needed one code change, making `CMD` respect `$PORT` with a 7860 fallback, verified locally with `docker run -e PORT=9000`.

`deploy/space/README.md` and `scripts/deploy_space.py` remain in the repository as a record of the path not taken. They are unused.

---

## 6. Repository structure

```
ieee-cis-fraud-detection/
├── .dvc/ , .dvcignore , .dockerignore
├── .github/workflows/ci.yml            # quality + docker jobs
├── .pre-commit-config.yaml             # + secret blocking
├── Dockerfile                          # $PORT with 7860 fallback
├── app/
│   ├── streamlit_app.py                                            # Step 7
│   ├── dashboard_data.json             # ~100 KB, committed        # Step 7
│   └── requirements.txt                                            # Step 7
├── config/config.py                    # extended in every step
├── data/
│   ├── raw/                            # git-ignored, 1.29 GB
│   ├── interim/                        # git-ignored, 2 Parquet files
│   └── processed/                      # DVC-tracked
├── deploy/space/README.md              # unused, Spaces path abandoned
├── docker/docker-compose.yml
├── docs/
│   ├── PROJECT_STATE.md                # this file
│   └── steps/step1.md ... step7.md
├── models/                             # git-ignored except metadata
│   ├── feature_engineer.joblib         # 28.0 MB
│   ├── final_model.joblib              #  5.3 MB
│   ├── selection_model.joblib          # fingerprinted
│   └── final_model_metadata.json       # committed
├── reports/
│   ├── eda_summary.md , feature_summary.md , training_summary.md
│   ├── *.csv                           # 10 report tables
│   ├── monitoring/                     # 4 files + dashboard_data.json
│   ├── figures/                        # 19 PNG charts
│   └── explainability/                 # 3 SHAP charts
├── scripts/
│   ├── download_data.py , verify_data.py
│   ├── promote_model.py , check_no_secrets.py
│   ├── publish_model.py , deploy_space.py (unused)
│   └── build_dashboard_data.py                                     # Step 7
├── src/
│   ├── features/engineer.py
│   ├── models/candidates.py
│   ├── monitoring/drift.py , promotion.py
│   ├── pipelines/ingestion.py , eda.py , features.py , training.py , monitoring.py
│   ├── serving/schemas.py , artifacts.py , scoring.py , app.py
│   └── utils/                          # 8 modules
├── tests/                              # 6 files, 29 tests
├── .env                                # git-ignored
├── .env.example , .gitignore , LICENSE , README.md , pyproject.toml
├── requirements.txt , -dev , -ci , -serve , .lock
├── mlflow.db                           # git-ignored
└── run.py                              # 5 stages
```

---

## 7. Environment

| Item | Value |
|------|-------|
| Python | **3.11.9** |
| Rebuild exactly | `pip install -r requirements.lock.txt` |

### 7.1 Library versions and the gotchas each one caused

| Library | Version | Notes |
|---------|---------|-------|
| pandas | 2.3.3 | `observed=True` on category groupbys; empty-frame concat deprecated |
| numpy | 2.4.6 | numpy 2.x, so `np.NaN` and `np.float_` do not exist |
| pyarrow | 24.0.0 | Parquet engine, preserves category dtypes |
| scipy | 1.17.1 | `ks_2samp` for drift |
| scikit-learn | 1.9.0 | `BaseEstimator`, `TransformerMixin`, `Pipeline` |
| lightgbm | 4.7.0 | **Production model.** `eval_set` deprecated for `eval_X`/`eval_y`. Needs `libgomp1` in a slim container |
| xgboost | 3.2.0 | Early stopping in the constructor |
| catboost | 1.2.10 | Plateaued at 0.529 with 4,000 rounds |
| mlflow | 3.15.1 | Aliases not stages; `log_model` uses `name`; URI needs forward slashes; sklearn flavor saves via skops and needs `skops_trusted_types=["numpy.dtype"]` |
| shap | 0.51.0 | Mean absolute SHAP hides rare features |
| fastapi | 0.141.1 | `/docs` generated from the Pydantic schemas |
| uvicorn | 0.52.3 | Binds `$PORT` with a 7860 fallback |
| huggingface_hub | 0.25+ | Model Hub upload and download. Space creation returns 402 without PRO |
| streamlit | 1.61.1 | The dashboard |
| plotly | 6.9.0 | Dashboard charts |
| pytest | 9.1.1 | 29 tests |
| ruff, black, pre-commit | 0.16.3, 26.5.1, 4.6.2 | CI green in 45s |
| dvc | 3.55+ | Local folder remote, verified |

---

## 8. The cost model

**Stated assumptions, not figures from a business.** All five in `config/config.py`.

| Assumption | Value | Reasoning |
|------------|-------|-----------|
| Analyst review | $4.00 per case | ~$60k/year fully loaded is ~$29/hour; a five minute review is $2.40; rounded up |
| Chargeback fee | $25.00 per missed fraud | Card networks charge $15 to $40 per dispute |
| False alarm friction | $1.00 | The softest number, the first to replace |
| Fraud recovered when caught | 90% | Reviews take time, some are judged wrongly |
| Review capacity | 2% of transactions | Roughly one analyst's full shift at this volume |

Missed fraud costs the amount plus the fee; caught fraud costs a review plus the 10% not recovered; a false alarm costs a review plus friction. Costs are weighted by the real transaction amount, computed exactly at every threshold via cumulative sums.

**Framing:** an order of magnitude under stated assumptions, never a forecast. Always pair count-based recall with value-based recall (D-58). Report savings alongside PR-AUC and say so when they disagree (D-69).

---

## 9. Conventions

**Code**
- All paths from `config/config.py`. One seed, `RANDOM_SEED = 42`, proven to give byte-identical reruns.
- Each pipeline stage reads a file and writes a file.
- `pipelines/` orders stages, `features/` builds features, `models/` defines candidates, `monitoring/` watches, `serving/` answers requests, `utils/` supports.
- Anything learned from data is learned from training rows only, then applied unchanged.
- Decision rules that use results are written down before the results are seen.
- When a library is mid-transition, inspect what is installed rather than assuming. Used four times.
- Secrets never appear in a source file, notebook, terminal command, or chat.
- Platform pricing and free tiers are verified before code is written against them (D-72).

**Git**
- Branch `step-NN-short-description`; commits `type: message`; squash-merge; tag `v0.N.0-stepN`, final `v1.0.0`
- CI green before merging. Formatting changes go in their own commit.

**Documentation**
- No em dashes; plain vocabulary; explanation before every code block
- Every file created is stated with full contents and its reason
- When an earlier claim turns out to be wrong, it is corrected openly in the next step

**Corrections made across the project**
1. Parquet size estimate 4x too high (Step 3)
2. Memory reduction range slightly optimistic (Step 3)
3. The 3.75x identity finding was confounded by `ProductCD` (Step 3)
4. The nine `id_` columns predicted to be dropped were all rescued (Step 4)
5. The uid family is 7 features, not 6 (Step 5)
6. Mean absolute SHAP was the wrong tool for rare-but-decisive features (Step 5)
7. The README stated the 42-day saving as an annual figure (Step 5)
8. The `--models catboost` command was destructive and overwrote production artefacts (Step 6)
9. Raw weekly PR-AUC hid a 21% decline because its floor moves with the fraud rate (Step 6)
10. `DRIFT_MIN_ROWS = 500` was too low and produced a PSI of 7.15 that was pure noise (Step 6)
11. **Hugging Face Spaces was stated to be free for Docker SDK. It requires PRO.** Discovered only after a deploy script had been built around it (Step 7)

---

## 10. Completed

- [x] Steps 1 to 4: data, EDA, features, model. Tagged through `v0.4.0-step4`
- [x] Step 5: 29 tests, CI green, drift monitoring, six promotion gates, version 4 promoted
- [x] Step 6: FastAPI service, Docker image, Model Hub publish, Render deployment, token revoked and secret hook installed
- [x] Monitoring re-run against the correct model; both Step 6 predictions confirmed
- [ ] Step 7: dashboard built, deployed, README finalised, tagged `v1.0.0`

---

## 11. Open questions and future work

| # | Question | Status |
|---|----------|--------|
| Q-01 to Q-15 | Earlier questions | **All answered.** |
| Q-16 | Should the model be retrained with amount-weighted examples? | **Open, and the highest-value experiment left.** It catches 44.6% of fraud by count but 31.2% by value, and CatBoost beats it on money despite a worse PR-AUC. Amount weighting aims to give LightGBM CatBoost's value-sensitivity without giving up its ranking. |
| Q-17 | CatBoost with a larger budget | **Answered.** 4,000 rounds gained +0.0009 over 1,500. Plateaued. |
| Q-18 | Should model selection use savings rather than PR-AUC? | Open. CatBoost saves ~8% more despite a worse ranking. One window, different feature sets. D-69 keeps savings visible on every future run. |
| Q-19 | Should the dashboard call the live API? | **Answered.** Yes, with a cached fallback so a sleeping free-tier service never leaves the page broken. D-73. |
| Q-20 | Which host for the dashboard? | Open until the run. Streamlit Community Cloud if its free tier is confirmed, otherwise Render, which is already proven. Verify first, per D-72. |
| Q-21 | When to retrain? | Open, and becoming urgent. Weighted PSI is at 83% of the 0.15 trigger and rose 53% across six months. A retrain including recent data is due within a month or two. |

---

## 12. How to resume from nothing

```powershell
git clone https://github.com/Dee-ui/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt

Copy-Item .env.example .env    # then fill in your own values

pytest                          # confirm the code before trusting its output

kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py
python run.py --step all

# Or skip the pipeline entirely and pull the trained model from the Hub
python -c "from src.serving.artifacts import load_artifacts; load_artifacts()"

code docs/PROJECT_STATE.md
```

---

## 13. Glossary

| Term | Plain meaning |
|------|---------------|
| Parquet | Stores tables column by column. Smaller and faster than CSV, and it remembers data types |
| Class imbalance | One outcome far rarer than the other, here 3.5% fraud |
| Accuracy | Share of predictions correct. Useless here: always predicting "not fraud" scores 96.5% |
| Precision | Of the transactions you flagged, the share that really were fraud |
| Recall | Of all the fraud that occurred, the share you caught. Report by count **and** by value |
| PR-AUC | Precision-Recall Area Under Curve. Primary metric. Its floor is the fraud rate of whatever you measure, so compare **lift** across periods |
| ROC-AUC | The Kaggle metric. Far less sensitive here: it spanned 0.032 across weeks where PR-AUC spanned 0.165 |
| Lift | A score divided by its baseline. Makes periods with different fraud rates comparable |
| Time-based split | Train on earlier data, validate on later data |
| Data leakage | Information unavailable at prediction time influencing training |
| Confounded comparison | A difference between two groups that is really a difference in what those groups contain |
| Frequency encoding | Replacing a category with how often it appeared in training. Unseen values get 0, which is how `uid_freq` went quiet |
| Fitted transformer | An object that learns from training data, stores it, and applies it later |
| Training and serving skew | Model trained on one set of transformations and fed another. Nothing errors; predictions are just wrong |
| Row independence | Transforming one row gives the same answer as transforming a batch containing it |
| Ablation | Removing part of a system on purpose to measure what it contributed |
| Pre-registered decision | A rule written down before the result is seen |
| Expanding-window CV | Folds where each trains on more history and is scored on the period straight after |
| MLflow alias | A movable pointer to a model version, such as `candidate` or `production` |
| SHAP | How much each feature pushed one prediction away from the average. Mean absolute SHAP hides rare-but-decisive features |
| DVC | Versions large data files alongside code, keeping a fingerprint in Git |
| Drift | When live data stops resembling training data, so the model quietly gets worse |
| PSI | Population Stability Index. Buckets the reference distribution and measures how much the new data's shares moved. Under 0.10 stable, over 0.25 investigate. Catches a collapse onto one value that a missingness check cannot see |
| KS statistic | The largest gap between two cumulative curves. The p-value is ignored: at 100,000 rows everything is significant |
| Importance-weighted drift | PSI weighted by SHAP importance, so drift in an ignored feature contributes nothing |
| Promotion gate | A check a model must pass before its alias moves to production |
| CI | Automated checks on a clean machine every time code is pushed |
| Container | A sealed box holding an OS, a Python, the libraries, and your code, so it runs identically everywhere |
| Docker layer caching | Copy requirements before source, so a code change does not reinstall dependencies |
| Model Hub | Where the artefacts live, versioned separately from the code, so a new model ships by restarting rather than rebuilding |
| Write token | A credential that can modify or delete anything in an account. Lives in `.env`, nowhere else |

---

*End of PROJECT_STATE.md. The project is complete.*
