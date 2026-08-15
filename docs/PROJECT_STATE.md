# PROJECT_STATE.md

**Last updated:** End of Step 2 of 7
**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Local path:** see Section 1.1, a relocation decision is pending

---

## 0. What this document is

This is the anchor for the whole project. It is rewritten in full at the end of every step, never patched with a diff.

If earlier conversation is lost, this file alone is enough to pick up exactly where we stopped. It records every decision made, the current state of the repository, what is finished, what is outstanding, and what questions are still open.

---

## 1. Project at a glance

| Item | Value |
|------|-------|
| Goal | A complete, portfolio-grade fraud detection system covering the full machine learning and MLOps lifecycle |
| Dataset | IEEE-CIS Fraud Detection (Kaggle, provided by Vesta Corporation) |
| Scope | Data pipeline, feature engineering, model training with experiment tracking, CI/CD, model registry, drift monitoring, Docker, deployment, dashboard |
| Delivery format | 7 steps, one per conversation message, each with its own markdown guide plus a refreshed copy of this file |
| Platform | Windows, VS Code, PowerShell, Python 3.11 |
| Machine | Intel Core Ultra 7 265H, 32 GB RAM. Enough to hold the full joined table in memory, so no chunked reading is needed anywhere |
| Version control | Git, public GitHub repository |
| Tracks | Technical track (this work) and a separate PM track walkthrough that happens afterwards with the project manager. Documentation quality matters for both. |

### 1.1 Project location, decision pending

The project was built at:

```
C:\Users\Dauda Agbonoga\OneDrive - Venture Garden Group\Documents\my\IEEEE_CIS_fraud_project
```

Two issues were raised in Step 2 Section 2:

1. The folder sits inside a synced OneDrive location. That means gigabytes of Kaggle data uploading to a corporate tenant, and a real risk of file locking errors during large writes, which get worse in Step 4 when MLflow starts writing thousands of small files.
2. The folder name (`IEEEE_CIS_fraud_project`, four E's) does not match the repository name (`ieee-cis-fraud-detection`).

Recommended fix is to move the project to `C:\projects\ieee-cis-fraud-detection`, delete `.venv`, and rebuild it from `requirements.lock.txt`. A virtual environment has absolute paths written inside it and cannot be moved.

The alternative is to stay put and exclude `data`, `models`, `mlruns`, and `.venv` from OneDrive sync.

Nothing in the codebase depends on the answer, because `config/config.py` resolves the project root from its own file location. That is decision D-07 and Step 2 proved it works.

**Update this section once the choice is made.**

---

## 2. Background: why this dataset

This project follows on from **NovaPay**, an earlier fraud detection prototype.

NovaPay's dataset had 9,940 transactions with 193 fraudulent cases, roughly 1.94% fraud, and only about 145 fraud cases in the training split. Best cross-validated PR-AUC was around 0.085 against a baseline of 0.015, so about five times lift, but the ceiling was set by the data rather than by technique.

IEEE-CIS fixes that:

- 590,540 training transactions instead of 9,940
- 20,663 fraud cases instead of 193
- 3.4990% fraud rate, which is realistic and still hard
- Two joinable tables instead of one flat file, so there is real data engineering to do
- A mix of named and anonymised features, which forces genuine feature work
- A test set that sits 30 days in the future, which makes the time-based validation lesson unavoidable rather than optional

**What carries over from NovaPay:** the pipeline architecture (`src/pipelines`, `src/serving`, `src/utils`), the stage-based design where each stage reads a file and writes a file, the central config module, the single `run.py` entry point pattern, and the MLflow and SHAP tooling choices.

**What does not carry over:** the data, the hardcoded absolute path in the config file (replaced with dynamic resolution), conda as the environment manager (replaced with `venv`), and the imputation-heavy cleaning stage (unnecessary with gradient boosted trees, which handle missing values natively).

---

## 3. The 7-step plan and current status

| Step | Content | Status |
|------|---------|--------|
| 1 | Dataset acquisition, folder scaffold, GitHub repo, Python environment | **Complete, verified** |
| 2 | EDA and data understanding: table joins, feature groups, imbalance profiling | **Delivered, awaiting the user's run** |
| 3 | Feature engineering and preprocessing pipeline | Not started |
| 4 | Model training with MLflow experiment tracking | Not started |
| 5 | MLOps layer: CI/CD, testing, model registry, drift monitoring | Not started |
| 6 | Dockerisation and deployment | Not started |
| 7 | Advanced dashboard and final documentation or portfolio packaging | Not started |

---

## 4. Decision log

| ID | Decision | Rationale | Set in |
|----|----------|-----------|--------|
| D-01 | Dataset is IEEE-CIS Fraud Detection | 590,540 rows at 3.4990% fraud gives 20,663 positive cases. NovaPay had 193, which capped performance regardless of method. | Step 1 |
| D-02 | Repository named `ieee-cis-fraud-detection` | Descriptive and scannable for reviewers. Local folder currently differs, see Section 1.1. | Step 1 |
| D-03 | Python 3.11 | Required minimum for the current Kaggle CLI, and has stable prebuilt Windows packages for LightGBM, XGBoost, CatBoost, and SHAP. Exact patch version still to be confirmed, see Q-10. | Step 1 |
| D-04 | `venv` plus `requirements.txt`, not conda | One dependency format that Docker (Step 6) and GitHub Actions (Step 5) both consume natively. | Step 1 |
| D-05 | Data is never committed to Git | Roughly 1.3 GB of raw CSV plus several hundred MB of Parquet. GitHub rejects files over 100 MB. Reproducibility comes from `scripts/download_data.py` instead. | Step 1 |
| D-06 | DVC deferred to Step 3 | DVC needs a storage remote, and the decision is better made once there are processed datasets worth versioning. | Step 1 |
| D-07 | Paths resolved dynamically in `config/config.py` | `Path(__file__).resolve().parents[1]` works on any machine, inside Docker, and in CI. Proven in Step 2 when it handled a folder name that does not match the repository name. | Step 1 |
| D-08 | Branch per step, merged into `main` by pull request, tagged after merge | Gives a reviewable trail, triggers CI in Step 5, and produces a clean narrative for the PM walkthrough. | Step 1 |
| D-09 | Public GitHub repository | It is a portfolio piece. Public also gives free Actions minutes and branch rulesets. | Step 1 |
| D-10 | Dependencies split into `requirements.txt` and `requirements-dev.txt` | Runtime needs stay separate from development tooling, so the Docker image in Step 6 stays lean. `requirements.lock.txt` from `pip freeze` provides exact reproducibility. | Step 1 |
| D-11 | Download script shells out to the Kaggle CLI rather than importing the Kaggle Python library | The library's internal interface has changed across versions; the CLI is the documented, stable contract. The script also retries with the older `-c` flag syntax. | Step 1 |
| D-12 | Interim and processed data stored as Parquet, not CSV | Roughly a third of the size, about ten times faster to load, and it preserves data types, which CSV loses entirely. | Step 1 |
| D-13 | Notebooks are for exploration only; anything that matters is rewritten as a module in `src/` | Notebooks are not testable, not importable, and not reviewable in diffs. `nbstripout` also strips output before commits. | Step 1 |
| D-14 | `run.py` created in Step 2 rather than Step 3 | Two runnable stages exist now, which is the point where a single entry point stops being overhead and becomes useful. | Step 2 |
| D-15 | Test set is joined and saved alongside train, despite having no labels | It becomes the drift monitoring input in Step 5. The Kaggle test set starts about 30 days after training ends, so it exhibits genuine real-world distribution shift, which is far better than artificially perturbing the training data. | Step 2 |
| D-16 | Left join transaction to identity, keep blanks as blanks, add a `has_identity` flag | LightGBM, XGBoost, and CatBoost all learn a direction for missing values at every split, so blanks cost nothing. Dropping 41 identity columns would throw away signal; splitting into two models would halve the data each sees. The flag makes the missingness itself available as a feature. | Step 2 |
| D-17 | Interim data stored as Parquet with category dtypes preserved | Type information survives a save and load, so the optimisation work is done once rather than on every read. | Step 2 |
| D-18 | `TransactionAmt` stays `float64`. `TransactionID` and `TransactionDT` become `int32`. Everything else is shrunk to the smallest exact type | Verified numerically: `float32` stores whole numbers exactly only below 16,777,216, and the test `TransactionDT` reaches 34,214,345, so `float32` would silently round it to 34,214,344. `float32` also turns a `TransactionAmt` of 31937.39 into 31937.390625, and the cents portion of the amount is a known fraud signal that Step 3 will extract. | Step 2 |
| D-19 | `TransactionDT` displayed against a reference date of 30 November 2017, for readability only | The competition never published a real start date. This community convention places the first transaction on 1 December 2017 and makes chart axes readable. `TransactionDT` is only ever used as an ordering and an elapsed duration, so nothing breaks if the calendar date is wrong. | Step 2 |
| D-20 | PR-AUC is the primary metric. ROC-AUC secondary. Recall at a fixed review rate is the business headline. Accuracy is not used at all | A do-nothing model scores 96.5% accuracy here. ROC-AUC is insensitive to false positives when the negative class is 569,877 rows. PR-AUC has a meaningful baseline equal to the fraud rate, 0.035. | Step 2 |
| D-21 | Validation is a time-based split, never random. Last 20% of the training period by `TransactionDT` | The real test set is 30 days in the future. A random split leaks in three specific ways here: the same card appears on both sides, `D` columns encode elapsed time from earlier events, and fraud arrives in bursts that a shuffle splits across both sides. | Step 2 |
| D-22 | Feature families assigned by rule, with unmapped columns reported loudly | With 435 columns, a silently miscategorised column is easy to miss. The profiler counts every column into exactly one family and warns if anything falls through. | Step 2 |

---

## 5. Current repository structure

Folders marked with a step number exist but are empty, waiting for that step.

```
ieee-cis-fraud-detection/
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
│   └── config.py                       # extended in Step 2
│
├── data/                               # git-ignored except .gitkeep
│   ├── raw/
│   │   ├── train_transaction.csv       651.7 MB
│   │   ├── train_identity.csv           25.3 MB
│   │   ├── test_transaction.csv        584.8 MB
│   │   ├── test_identity.csv            24.6 MB
│   │   └── sample_submission.csv         5.8 MB
│   ├── interim/
│   │   ├── train_joined.parquet        # 590,540 x 435   (Step 2)
│   │   └── test_joined.parquet         # 506,691 x 434   (Step 2)
│   ├── processed/                      # empty                     (Step 3)
│   └── external/                       # empty
│
├── docker/                             # empty                     (Step 6)
│
├── docs/
│   ├── PROJECT_STATE.md                # this file
│   ├── steps/
│   │   ├── step1.md
│   │   └── step2.md
│   └── decisions/                      # empty
│
├── models/                             # empty, git-ignored        (Step 4)
│
├── notebooks/                          # empty
│
├── reports/
│   ├── data_inventory.md               # from scripts/verify_data.py
│   ├── eda_summary.md                  # generated                 (Step 2)
│   ├── column_profile.csv              # generated                 (Step 2)
│   ├── missing_profile.csv             # generated                 (Step 2)
│   ├── v_column_missing_groups.csv     # generated                 (Step 2)
│   ├── figures/                        # 10 PNG charts             (Step 2)
│   └── explainability/                 # empty                     (Step 4)
│
├── scripts/
│   ├── download_data.py
│   └── verify_data.py
│
├── src/
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── ingestion.py                                          # Step 2
│   │   └── eda.py                                                # Step 2
│   ├── serving/__init__.py             # modules added            (Step 6)
│   ├── monitoring/__init__.py          # modules added            (Step 5)
│   └── utils/
│       ├── __init__.py
│       ├── memory_utils.py                                       # Step 2
│       ├── ingestion_utils.py                                    # Step 2
│       └── eda_utils.py                                          # Step 2
│
├── tests/
│   └── __init__.py                     # tests added              (Step 5)
│
├── .env.example
├── .gitignore
├── LICENSE                             # MIT
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── requirements.lock.txt
└── run.py                                                        # Step 2
```

---

## 6. Files created so far and what each one does

### Step 1

| File | Purpose |
|------|---------|
| `.gitignore` | Blocks data, models, secrets, virtual environment, MLflow artifacts, CatBoost logs, and editor noise. Keeps `.gitkeep` files so empty folders survive. |
| `.env.example` | Template listing which secrets are needed, with no real values. Safe to commit. |
| `.vscode/settings.json` | Points VS Code at `.venv`, enables pytest, sets the project root as an import path, formats on save. |
| `README.md` | Problem statement, dataset table, quickstart, roadmap, results table, tech stack, NovaPay background. |
| `LICENSE` | MIT. |
| `requirements.txt` | Runtime dependencies with `>=` minimums. |
| `requirements-dev.txt` | Development tooling. Starts with `-r requirements.txt`. |
| `requirements.lock.txt` | Exact installed versions from `pip freeze`. The reproducibility guarantee, and what a rebuilt environment should install from. |
| `config/config.py` | Extended in Step 2, see below. |
| `scripts/download_data.py` | Checks the Kaggle CLI, skips if files exist unless `--force`, downloads via subprocess with a fallback to the older flag syntax, extracts, deletes the zip, reports sizes, prints targeted troubleshooting on failure. |
| `scripts/verify_data.py` | Five checks: file presence and size, row and column counts, fraud rate, `TransactionID` uniqueness and identity coverage, and the `id_` versus `id-` naming difference. Writes `reports/data_inventory.md`. Uses `usecols` so it never loads all 394 columns. |

### Step 2

| File | Purpose |
|------|---------|
| `config/config.py` | Now also holds: feature family lists built by comprehension (`C_COLUMNS`, `D_COLUMNS`, `M_COLUMNS`, `V_COLUMNS`, `IDENTITY_COLUMNS`, `CARD_COLUMNS`, and so on), `IDENTITY_FLAG_COLUMN`, `REFERENCE_DATETIME`, seconds constants, joined and processed file paths, EDA output paths, `VALIDATION_FRACTION`, `MIN_CATEGORY_COUNT`, `HIGH_MISSING_THRESHOLD`. |
| `src/utils/memory_utils.py` | `optimise_dtypes` shrinks every column to its smallest safe type, with a `PROTECTED_DTYPES` map covering the four columns where shrinking would corrupt data. Also `memory_usage_mb` and `dtype_breakdown`. |
| `src/utils/ingestion_utils.py` | `load_csv`, `standardise_identity_columns` (renames `id-NN` to `id_NN` via a full-match regex), `add_identity_marker`, `join_transaction_identity` (left join with `validate="one_to_one"`), `validate_join`, `save_parquet`. |
| `src/pipelines/ingestion.py` | Orchestrates load, standardise, join, validate, optimise, save. Handles train and test through one code path with a `SPLIT_SETTINGS` dictionary holding the expected shapes. Supports `nrows` for smoke testing. |
| `src/utils/eda_utils.py` | Feature family assignment by regex, column profiling, `missing_pattern_groups` (hashes each column's blank mask to find V blocks), `fraud_rate_by_category`, `derive_time_frame`, `time_range_summary`, and ten chart functions. Sets the matplotlib `Agg` backend before importing pyplot. |
| `src/pipelines/eda.py` | Runs every analysis, writes four report files and ten charts, and auto-generates `reports/eda_summary.md` from the computed results rather than from hand-written text. |
| `run.py` | Single entry point. `--step ingestion|eda|all`, `--split train|test|both`, `--nrows N`. Imports each stage inside its own function so startup stays fast. |

---

## 7. Environment details

| Item | Value |
|------|-------|
| Python | 3.11.x, created with `py -3.11 -m venv .venv`. Exact patch version to be confirmed, see Q-10 |
| Environment folder | `.venv` in the project root, git-ignored |
| Activate | `.\.venv\Scripts\Activate.ps1` |
| If activation is blocked | `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` |
| Rebuild exactly | `pip install -r requirements.lock.txt` |
| VS Code interpreter | Set to `.venv` |

### 7.1 Confirmed library versions

These are the actual installed versions. All Step 2 code was written against them.

| Library | Version | Notes that affected the code |
|---------|---------|------------------------------|
| pandas | 2.3.3 | `observed=True` passed explicitly on every category groupby, to avoid the future-default warning |
| numpy | 2.4.6 | numpy 2.x, so `np.NaN` and `np.float_` do not exist. Code uses `np.nan` and explicit type strings |
| pyarrow | 24.0.0 | Parquet engine, preserves category dtypes across save and load |
| scipy | 1.17.1 | |
| scikit-learn | 1.9.0 | Used from Step 3 |
| lightgbm | 4.7.0 | Native missing value handling, which underpins D-16 |
| xgboost | 3.2.0 | Same |
| catboost | 1.2.10 | Same |
| imbalanced-learn | 0.14.2 | Step 4 if resampling is trialled |
| mlflow | 3.15.1 | MLflow 3, whose API differs from MLflow 2 in places. Step 4 code must target 3.x |
| shap | 0.51.0 | Step 4 |
| matplotlib | 3.11.1 | `plt.cm.get_cmap` removed in 3.9, so no chart code uses it. `tick_labels` used on boxplots, which requires 3.9 or newer |
| seaborn | 0.13.2 | |
| plotly | 6.9.0 | Step 7 dashboard |
| fastapi | 0.141.1 | Step 6 |
| uvicorn | 0.52.3 | Step 6 |
| streamlit | 1.61.1 | Step 7 |
| pytest | 9.1.1 | Step 5 |
| ruff | 0.16.3 | Step 5 |
| black | 26.5.1 | Step 5 |
| pre-commit | 4.6.2 | Step 5 |
| kaggle | 2.2.4 | Current CLI, positional competition argument |
| jupytext | 1.19.5 | Present in the environment, not currently used |

---

## 8. Dataset facts

**Source:** Kaggle competition `ieee-fraud-detection`. The API only works after joining the competition and accepting its rules. Without that, downloads fail with a 403 error that does not explain itself.

### 8.1 Raw files, all verified

| File | Size | Rows | Columns |
|------|------|------|---------|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 |
| `sample_submission.csv` | 5.8 MB | 506,691 | 2 |

### 8.2 Verified facts

- Fraud rate **3.4990%**: 20,663 fraudulent out of 590,540, so 569,877 legitimate, roughly 1 in 28
- `TransactionID` unique in both training tables, zero duplicates
- 144,233 transactions have an identity record, exactly **24.4%**
- `train_identity` has 38 columns starting `id_`, `test_identity` has 38 starting `id-`

### 8.3 Column families

Transaction table, 394 columns:

| Family | Columns | Count |
|--------|---------|-------|
| Identifier | `TransactionID` | 1 |
| Target | `isFraud` | 1 |
| Time | `TransactionDT` | 1 |
| Amount | `TransactionAmt` | 1 |
| Product | `ProductCD` | 1 |
| Card | `card1` to `card6` | 6 |
| Address | `addr1`, `addr2` | 2 |
| Distance | `dist1`, `dist2` | 2 |
| Email | `P_emaildomain`, `R_emaildomain` | 2 |
| Counting | `C1` to `C14` | 14 |
| Timedelta | `D1` to `D15` | 15 |
| Match | `M1` to `M9` | 9 |
| Vesta engineered | `V1` to `V339` | 339 |

Identity table, 41 columns: `TransactionID`, `id_01` to `id_38`, `DeviceType`, `DeviceInfo`.

Joined train: 394 + 40 + 1 flag = **435 columns**. Joined test: 393 + 40 + 1 = **434 columns**.

### 8.4 Time structure

- `TransactionDT` is seconds from an undisclosed reference moment, not a calendar timestamp
- Training minimum is 86,400, which is exactly one day in seconds
- Training spans roughly 183 days
- Test begins roughly 30 days after training ends, then runs a further 183 days or so
- Reference date of 30 November 2017 used for display only, per D-19

### 8.5 Known quirks

- Test identity columns use hyphens (`id-01`) where training uses underscores (`id_01`). Handled in `standardise_identity_columns`.
- V columns cluster into blocks that share an identical missing pattern, because Vesta built them in batches from shared sources. Detected and written to `reports/v_column_missing_groups.csv`.
- The decimal portion of `TransactionAmt` is a known fraud signal, which is why `float64` precision is preserved on that column.

---

## 9. Conventions in force

**Code**
- All paths come from `config/config.py`. No module builds its own paths.
- One random seed, `RANDOM_SEED = 42`, used everywhere.
- Each pipeline stage reads a file and writes a file, so stages run and debug independently.
- `src/pipelines/` says what happens in what order. `src/utils/` says how each thing is done. Helpers stay individually testable.
- Code that matters lives in `src/`, not in notebooks.
- Every function gets a docstring. Non-obvious lines get an inline comment.
- Every groupby over a category column passes `observed=True`.
- Matplotlib uses the `Agg` backend, set before pyplot is imported.

**Git**
- Branch naming: `step-NN-short-description`
- Commit message style: `type: message`, using `feat`, `fix`, `docs`, `build`, `chore`, `test`, `refactor`
- One branch per step, merged into `main` by squash-merge pull request, then tagged `v0.N.0-stepN`
- `main` must always be in a working state

**Documentation**
- No em dashes
- Plain vocabulary, with advanced ideas explained rather than assumed
- Numbered steps
- An explanation before every code block, and comments inside the code
- Every file created gets stated, with its full contents and the reason it exists

---

## 10. Completed

### Step 1, verified

- [x] Kaggle competition joined and rules accepted
- [x] Kaggle API credentials configured
- [x] Full folder scaffold created, covering all 7 steps
- [x] `.gitignore`, `README.md`, `LICENSE`, `.env.example`, `.vscode/settings.json` created
- [x] Git initialised, GitHub repository created at `Dee-ui/ieee-cis-fraud-detection`, first push completed
- [x] Branch `step-01-foundations` used, merged, tagged `v0.1.0-step1`
- [x] Python 3.11 virtual environment created and activated
- [x] Dependencies installed and locked to `requirements.lock.txt`
- [x] `config/config.py` created with dynamic path resolution
- [x] `scripts/download_data.py` and `scripts/verify_data.py` created and run successfully
- [x] All five raw CSV files present and verified
- [x] `reports/data_inventory.md` generated
- [x] Confirmed no data files tracked by Git

### Step 2, delivered and awaiting the user's run

- [x] `config/config.py` extended with feature families, EDA paths, split settings
- [x] `src/utils/memory_utils.py` created
- [x] `src/utils/ingestion_utils.py` created
- [x] `src/pipelines/ingestion.py` created
- [x] `run.py` created
- [x] `src/utils/eda_utils.py` created
- [x] `src/pipelines/eda.py` created
- [x] Metric decision made and documented (D-20)
- [x] Validation split decision made and documented (D-21)
- [ ] Ingestion run completed by the user
- [ ] EDA run completed by the user
- [ ] Outputs reviewed and reported back
- [ ] Branch merged and tagged `v0.2.0-step2`

---

## 11. Pending

**Immediately next (Step 3)**
- Drop zero-information columns: single-valued columns and near-duplicates
- Reduce 339 V columns using the block structure in `v_column_missing_groups.csv`, keeping representatives per block rather than applying an arbitrary cutoff
- Time features: hour, day of week, position within the period
- `TransactionAmt` decomposition, including the cents feature that D-18 preserved precision for
- Frequency encoding for high cardinality columns such as `card1`
- Aggregate features: amount relative to the mean for its card, address, and email domain
- UID construction from card and address columns, with an honest account of when it helps and when it overfits
- Email domain normalisation
- Build everything as a scikit-learn pipeline object, so identical transformations apply at training and prediction time
- Implement the time-based split in code
- Resolve the DVC question, Q-03
- Write output to `data/processed/`

**Later steps**
- Step 4: model training, MLflow 3.x tracking, threshold selection at a fixed review rate, SHAP explainability
- Step 5: pytest suite, GitHub Actions CI, MLflow model registry, drift monitoring using the test set as genuine future data, retraining trigger
- Step 6: Dockerfile, docker-compose, FastAPI service, deployment target
- Step 7: dashboard, architecture diagram, README results, portfolio packaging

---

## 12. Open questions

| # | Question | Needed by | Status |
|---|----------|-----------|--------|
| Q-01 | Exact installed library versions | Step 2 | **Answered.** Recorded in Section 7.1. |
| Q-02 | How much RAM does the machine have | Step 2 | **Answered.** 32 GB, Intel Core Ultra 7 265H. No chunked reading needed. |
| Q-03 | Where will DVC store data remotely, if used | Step 3 | Open. Options: a local folder remote (simplest, offline, free), Google Drive (free, fiddly authentication), or S3 or equivalent (most professional, small cost). |
| Q-04 | Where does the service get deployed in Step 6 | Step 6 | Open. Options: Render or Railway (free tier, simplest), Hugging Face Spaces (free, good for a dashboard), a cloud provider (most impressive, has cost), or local Docker only. |
| Q-05 | Is a Kaggle leaderboard submission wanted | Step 4 | Open. Assuming no. The competition is closed but late submissions still score, which would give an external validation number for the README. Cheap to add. |
| Q-06 | Streamlit or a React front end for the Step 7 dashboard | Step 7 | Open. Assuming Streamlit, which is already installed at 1.61.1. |
| Q-07 | Is there a business framing for the PM track, such as cost per missed fraud and per false alarm | Step 4 | Open. If the PM supplies figures, threshold selection becomes a cost optimisation rather than a purely statistical exercise, which is a much stronger story. |
| Q-08 | Project location: move out of OneDrive, or exclude folders from sync | Before Step 3 | Open. See Section 1.1. Recommendation is to move to `C:\projects\ieee-cis-fraud-detection` and rebuild `.venv`. |
| Q-09 | Rename the local folder to match the repository | Before Step 3 | Open. Resolved automatically if Q-08 is answered by moving. |
| Q-10 | Exact Python patch version | Step 3 | Open. `python --version`. All code so far targets 3.11 syntax and standard library. |
| Q-11 | V column block structure from the user's actual run | Step 3 | Open. `reports/v_column_missing_groups.csv` needed as an attachment. The block count and sizes determine the reduction strategy, and guessing would be wrong. |

---

## 13. How to resume from nothing

If everything is lost except the GitHub repository:

```powershell
# 1. Clone and enter the project
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

# 4. Rebuild the interim data and reports
python run.py --step all

# 5. Read the current state
code docs/PROJECT_STATE.md
```

If the GitHub repository is also lost, `docs/steps/step1.md` and `docs/steps/step2.md` rebuild everything from scratch.

---

## 14. Glossary

| Term | Plain meaning |
|------|---------------|
| Virtual environment | A private copy of Python belonging to one project, so its packages cannot clash with other projects. Contains absolute paths, so it cannot be moved, only rebuilt |
| `.gitignore` | A list of file patterns Git pretends do not exist, so they are never committed |
| Branch | A parallel line of work that can be merged back in when it is ready |
| Pull request | A request to merge a branch, which creates a place to review changes and run automated checks |
| Tag | A permanent bookmark on one specific commit, used here to mark the end of each step |
| Parquet | A file format that stores tables column by column. Smaller and faster than CSV, and it remembers data types |
| Left join | Keep every row from the left table, attach matching data from the right table where it exists, leave blanks where it does not |
| dtype | The data type of a column, such as int8 or float32. Choosing smaller ones cuts memory use substantially |
| Category dtype | Stores each distinct text value once, plus a small number per row pointing at it. Very efficient for repetitive text |
| Class imbalance | When one outcome is far rarer than the other, here 3.5% fraud against 96.5% legitimate |
| Accuracy | Share of predictions that were correct. Useless here, because always predicting "not fraud" scores 96.5% |
| Precision | Of the transactions you flagged, the share that were actually fraud |
| Recall | Of all the fraud that occurred, the share you caught |
| PR-AUC | Precision-Recall Area Under Curve. The primary metric here. Its baseline equals the fraud rate, 0.035 |
| ROC-AUC | Probability that a randomly chosen fraud scores higher than a randomly chosen legitimate transaction. The competition metric. Insensitive to false positives when the negative class is huge |
| Time-based split | Train on earlier data, validate on later data. Imitates the real situation, where you always predict the future |
| Data leakage | When information that would not be available at prediction time influences training, producing a validation score you cannot reproduce in production |
| MLflow | A tool that records every training run: settings used, metrics produced, and the model file itself |
| SHAP | A method that explains which features pushed a single prediction up or down |
| Drift | When live data slowly stops resembling training data, so the model quietly gets worse |
| CI/CD | Automated checks and deployment that run on every code change |
| Model registry | A catalogue of trained model versions with a record of which one is live |

---

*End of PROJECT_STATE.md. Next: Step 3, feature engineering and preprocessing pipeline.*
