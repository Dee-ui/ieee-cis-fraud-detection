# PROJECT_STATE.md

**Last updated:** End of Step 1 of 7
**Project:** IEEE-CIS Fraud Detection
**Repository:** `ieee-cis-fraud-detection`
**Local path:** `C:\projects\ieee-cis-fraud-detection`

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
| Version control | Git, public GitHub repository |
| Tracks | Technical track (this work) and a separate PM track walkthrough that happens afterwards with the project manager. Documentation quality matters for both. |

---

## 2. Background: why this dataset

This project follows on from **NovaPay**, an earlier fraud detection prototype.

NovaPay's dataset had 9,940 transactions with 193 fraudulent cases, roughly 1.94% fraud, and only about 145 fraud cases in the training split. Best cross-validated PR-AUC was around 0.085 against a baseline of 0.015, so about five times lift, but the ceiling was set by the data rather than by technique. There were simply not enough fraud examples to learn from.

IEEE-CIS fixes that:

- 590,540 training transactions instead of 9,940
- Roughly 20,600 fraud cases instead of 193
- 3.5% fraud rate, which is realistic and still hard
- Two joinable tables instead of one flat file, so there is real data engineering to do
- A mix of named and anonymised features, which forces genuine feature work

**What carries over from NovaPay:** the pipeline architecture (`src/pipelines`, `src/serving`, `src/utils`), the stage-based design where each stage reads a file and writes a file, the central config module, the single `run.py` entry point pattern, and the MLflow and SHAP tooling choices.

**What does not carry over:** the data, the hardcoded absolute path in the config file (replaced with dynamic resolution), and conda as the environment manager (replaced with `venv`).

---

## 3. The 7-step plan and current status

| Step | Content | Status |
|------|---------|--------|
| 1 | Dataset acquisition, folder scaffold, GitHub repo, Python environment | **Complete** |
| 2 | EDA and data understanding: table joins, feature groups, imbalance profiling | Not started |
| 3 | Feature engineering and preprocessing pipeline | Not started |
| 4 | Model training with MLflow experiment tracking | Not started |
| 5 | MLOps layer: CI/CD, testing, model registry, drift monitoring | Not started |
| 6 | Dockerisation and deployment | Not started |
| 7 | Advanced dashboard and final documentation or portfolio packaging | Not started |

---

## 4. Decision log

| ID | Decision | Rationale | Set in |
|----|----------|-----------|--------|
| D-01 | Dataset is IEEE-CIS Fraud Detection | 590,540 rows at 3.5% fraud gives roughly 20,600 positive cases. NovaPay had 193, which capped performance regardless of method. | Step 1 |
| D-02 | Project and repo named `ieee-cis-fraud-detection` | Descriptive and scannable for reviewers. Appears only in the folder name and README, so it is cheap to change before the first push and expensive after. | Step 1 |
| D-03 | Python 3.11 | Required minimum for the current Kaggle CLI, and has stable prebuilt Windows packages for LightGBM, XGBoost, CatBoost, and SHAP. | Step 1 |
| D-04 | `venv` plus `requirements.txt`, not conda | One dependency format that Docker (Step 6) and GitHub Actions (Step 5) both consume natively. NovaPay used conda; this is a deliberate change. | Step 1 |
| D-05 | Data is never committed to Git | Roughly 1.3 GB extracted, and GitHub rejects files over 100 MB. Reproducibility comes from `scripts/download_data.py` instead. | Step 1 |
| D-06 | DVC deferred to Step 3 | DVC needs a storage remote, and the decision is better made once there are processed datasets worth versioning. The folder layout is already compatible. | Step 1 |
| D-07 | Paths resolved dynamically in `config/config.py` | `Path(__file__).resolve().parents[1]` works on any machine, inside Docker, and in CI. NovaPay's hardcoded absolute path did not. | Step 1 |
| D-08 | Branch per step, merged into `main` by pull request, tagged after merge | Gives a reviewable trail, triggers CI in Step 5, and produces a clean narrative for the PM walkthrough. | Step 1 |
| D-09 | Public GitHub repository | It is a portfolio piece. Public also gives free Actions minutes and branch rulesets. | Step 1 |
| D-10 | Dependencies split into `requirements.txt` and `requirements-dev.txt` | Runtime needs stay separate from development tooling, so the Docker image in Step 6 stays lean. `requirements.lock.txt` from `pip freeze` provides exact reproducibility. | Step 1 |
| D-11 | Download script shells out to the Kaggle CLI rather than importing the Kaggle Python library | The library's internal interface has changed across versions; the CLI is the documented, stable contract. The script also retries with the older `-c` flag syntax for compatibility. | Step 1 |
| D-12 | Interim and processed data will be stored as Parquet, not CSV | Far smaller on disk, much faster to read, and it preserves data types, which CSV loses. | Step 1 |
| D-13 | Notebooks are for exploration only; anything that matters is rewritten as a module in `src/` | Notebooks are not testable, not importable, and not reviewable in diffs. `nbstripout` also strips output before commits. | Step 1 |

---

## 5. Current repository structure

Folders marked with a step number exist but are empty, waiting for that step.

```
ieee-cis-fraud-detection/
│
├── .github/
│   └── workflows/                  # empty                       (Step 5)
│
├── .vscode/
│   └── settings.json               # shared editor configuration
│
├── app/                            # empty                       (Step 7)
│
├── config/
│   ├── __init__.py
│   └── config.py                   # all paths and constants
│
├── data/                           # git-ignored except .gitkeep files
│   ├── raw/
│   │   ├── train_transaction.csv
│   │   ├── train_identity.csv
│   │   ├── test_transaction.csv
│   │   ├── test_identity.csv
│   │   └── sample_submission.csv
│   ├── interim/                    # empty                       (Step 2)
│   ├── processed/                  # empty                       (Step 3)
│   └── external/                   # empty
│
├── docker/                         # empty                       (Step 6)
│
├── docs/
│   ├── PROJECT_STATE.md            # this file
│   ├── steps/
│   │   └── step1.md
│   └── decisions/                  # empty
│
├── models/                         # empty, git-ignored          (Step 4)
│
├── notebooks/                      # empty                       (Step 2)
│
├── reports/
│   ├── data_inventory.md           # written by verify_data.py
│   ├── figures/                    # empty                       (Step 2)
│   └── explainability/             # empty                       (Step 4)
│
├── scripts/
│   ├── download_data.py
│   └── verify_data.py
│
├── src/
│   ├── __init__.py
│   ├── pipelines/__init__.py       # modules added               (Steps 2 to 4)
│   ├── serving/__init__.py         # modules added               (Step 6)
│   ├── monitoring/__init__.py      # modules added               (Step 5)
│   └── utils/__init__.py           # modules added               (Steps 2 to 4)
│
├── tests/
│   └── __init__.py                 # tests added                 (Step 5)
│
├── .env.example
├── .gitignore
├── LICENSE                         # MIT
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── requirements.lock.txt           # generated by pip freeze
└── run.py                          # NOT YET CREATED             (Step 3)
```

---

## 6. Files created so far and what each one does

| File | Purpose |
|------|---------|
| `.gitignore` | Blocks data, models, secrets, virtual environment, MLflow artifacts, CatBoost logs, and editor noise from Git. Keeps `.gitkeep` files so empty folders survive. |
| `.env.example` | Template listing which secrets are needed (Kaggle credentials, MLflow URI) without containing any real values. Safe to commit. |
| `.vscode/settings.json` | Points VS Code at `.venv`, enables pytest, sets the project root as an import path, formats on save. |
| `README.md` | Front door of the project. Problem statement, dataset table, quickstart, roadmap with step checkboxes, results table (empty until Step 4), tech stack, NovaPay background. |
| `LICENSE` | MIT. |
| `requirements.txt` | Runtime dependencies: pandas, numpy, pyarrow, scipy, scikit-learn, lightgbm, xgboost, catboost, imbalanced-learn, mlflow, shap, matplotlib, seaborn, plotly, fastapi, uvicorn, pydantic, streamlit, joblib, pyyaml, python-dotenv. Uses `>=` minimums. |
| `requirements-dev.txt` | Development tooling: kaggle, jupyter, ipykernel, nbstripout, pytest, pytest-cov, httpx, ruff, black, pre-commit. Starts with `-r requirements.txt`. |
| `requirements.lock.txt` | Exact installed versions from `pip freeze`. This is the reproducibility guarantee. |
| `config/config.py` | Project root resolved from the file's own location. Data directories, raw file paths, expected file list, key column names, planned interim and processed file paths, model and report directories, MLflow settings, modelling defaults, and an `ensure_directories()` helper. |
| `scripts/download_data.py` | Checks the Kaggle CLI is available, skips if files already exist unless `--force` is passed, downloads via subprocess with a fallback to the older flag syntax, extracts the archive, deletes the zip, reports file sizes, and prints targeted troubleshooting on failure. |
| `scripts/verify_data.py` | Five checks: file presence and size, row and column counts against published figures, fraud rate within tolerance, `TransactionID` uniqueness plus identity-table coverage, and detection of the `id_` versus `id-` naming difference. Writes `reports/data_inventory.md`. Uses `usecols` so it never loads all 394 columns. |
| `reports/data_inventory.md` | Generated table of file sizes, row counts, column counts, and the measured fraud rate. |
| `docs/steps/step1.md` | The full Step 1 walkthrough. |

---

## 7. Environment details

| Item | Value |
|------|-------|
| Python | 3.11.x, created with `py -3.11 -m venv .venv` |
| Environment folder | `.venv` in the project root, git-ignored |
| Activate | `.\.venv\Scripts\Activate.ps1` |
| If activation is blocked | `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` |
| Install | `pip install -r requirements-dev.txt` (this pulls in `requirements.txt` too) |
| Lock | `pip freeze > requirements.lock.txt` |
| VS Code interpreter | Set to `.venv` via `Ctrl+Shift+P`, `Python: Select Interpreter` |
| Import smoke test | `python -c "import pandas, sklearn, lightgbm, xgboost, catboost, mlflow, shap; print('all imports OK')"` |

**Exact installed versions:** not yet recorded here. To be pasted in at the start of Step 2 from `requirements.lock.txt`, so later code is written against the real versions.

---

## 8. Dataset facts

**Source:** Kaggle competition `ieee-fraud-detection`. The API only works after joining the competition and accepting its rules at `https://www.kaggle.com/competitions/ieee-fraud-detection/rules`. Without that, downloads fail with a 403 error that does not explain itself.

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train_transaction.csv` | 590,540 | 394 | Contains the `isFraud` label |
| `train_identity.csv` | 144,233 | 41 | Device and network signals |
| `test_transaction.csv` | 506,691 | 393 | No label |
| `test_identity.csv` | 141,907 | 41 | No label |
| `sample_submission.csv` | 506,691 | 2 | Kaggle submission template |

**Key facts to carry forward:**

- Fraud rate is roughly 3.5%, giving roughly 20,600 fraud cases and an imbalance of about 1 in 28
- The two tables join on `TransactionID`
- Only about 144,233 of 590,540 transactions, roughly 24%, have an identity record. The join is a left join and the missing identity data is informative, not a defect
- `TransactionDT` is a time delta in seconds from an unknown reference point, not a calendar timestamp. It still orders the data correctly, which matters for splitting
- The test identity file uses hyphens (`id-01`) where the training file uses underscores (`id_01`). This is a known quirk of the released files and is fixed in Step 2
- Column families: `C1` to `C14` counting features, `D1` to `D15` timedelta features, `M1` to `M9` match flags, `V1` to `V339` anonymised Vesta engineered features, plus `card1` to `card6`, `addr1`, `addr2`, `dist1`, `dist2`, `P_emaildomain`, `R_emaildomain`, `ProductCD`

**Actual verification output:** to be pasted in at the start of Step 2.

---

## 9. Conventions in force

**Code**
- All paths come from `config/config.py`. No module builds its own paths.
- One random seed, `RANDOM_SEED = 42`, used everywhere.
- Each pipeline stage reads a file and writes a file, so stages can be run and debugged independently.
- Code that matters lives in `src/`, not in notebooks.
- Every function gets a docstring. Non-obvious lines get an inline comment.

**Git**
- Branch naming: `step-NN-short-description`
- Commit message style: `type: message`, using `feat`, `fix`, `docs`, `build`, `chore`, `test`, `refactor`
- One branch per step, merged into `main` by pull request with a squash merge, then tagged `v0.N.0-stepN`
- `main` must always be in a working state

**Documentation**
- No em dashes
- Plain vocabulary, with advanced ideas explained rather than assumed
- Numbered steps
- An explanation before every code block, and comments inside the code
- Every file created gets stated, with its full contents and the reason it exists

---

## 10. Completed in Step 1

- [x] Kaggle competition joined and rules accepted
- [x] Kaggle API credentials configured
- [x] Full folder scaffold created, covering all 7 steps
- [x] `.gitignore`, `README.md`, `LICENSE`, `.env.example`, `.vscode/settings.json` created
- [x] Git initialised, GitHub repository created, first push completed
- [x] Branch strategy defined and `step-01-foundations` branch used, merged, and tagged
- [x] Python 3.11 virtual environment created and activated
- [x] Runtime and development dependencies installed and locked
- [x] VS Code interpreter set
- [x] `config/config.py` created with dynamic path resolution
- [x] `scripts/download_data.py` created and run successfully
- [x] `scripts/verify_data.py` created and run successfully
- [x] All five raw CSV files present in `data/raw/` and verified
- [x] `reports/data_inventory.md` generated
- [x] Confirmed no data files are tracked by Git

---

## 11. Pending

**Immediately next (Step 2)**
- Join `train_transaction` with `train_identity` on `TransactionID`
- Group all 394 columns into feature families and document what each family appears to represent
- Profile missing values, including the columns that are more than 90% empty
- Profile the class imbalance and establish PR-AUC as the headline metric rather than accuracy
- Interpret `TransactionDT` and decide the train and validation split strategy, which must respect time order
- Reduce memory by choosing smaller numeric types, targeting roughly 1.9 GB down to roughly 550 MB
- Save the joined table to `data/interim/train_joined.parquet`
- Produce the first charts into `reports/figures/`
- Write `src/pipelines/ingestion.py` and `src/utils/ingestion_utils.py`

**Later steps**
- Step 3: feature engineering, preprocessing pipeline, `run.py` entry point, DVC decision
- Step 4: model training, MLflow tracking, threshold selection, SHAP explainability
- Step 5: pytest suite, GitHub Actions CI, MLflow model registry, drift monitoring, retraining trigger
- Step 6: Dockerfile, docker-compose, FastAPI service, deployment target
- Step 7: dashboard, architecture diagram, README results, portfolio packaging

---

## 12. Open questions

| # | Question | Needed by | Current working assumption |
|---|----------|-----------|----------------------------|
| Q-01 | What are the exact installed library versions? | Step 2 | Latest available at install time. Paste `requirements.lock.txt` contents so Step 2 code targets the real versions. |
| Q-02 | How much RAM does the machine have? | Step 2 | Assuming 16 GB. If it is 8 GB, Step 2 switches to chunked reading and more aggressive type downcasting from the start. |
| Q-03 | Where will DVC store data remotely, if we use it? | Step 3 | Options are a local folder remote (simplest, works offline, no cost), Google Drive (free, slightly fiddly to authenticate), or S3 or equivalent (most professional, small cost). |
| Q-04 | Where does the service get deployed in Step 6? | Step 6 | Options are Render or Railway (free tier, simplest), Hugging Face Spaces (free, good for a dashboard), a cloud provider (most impressive, has cost), or local Docker only (no cost, less impressive). |
| Q-05 | Is a Kaggle leaderboard submission wanted? | Step 4 | Assuming no. The competition is closed, but late submissions still score, which would give an external validation number for the README. Cheap to add if wanted. |
| Q-06 | Streamlit or a React front end for the Step 7 dashboard? | Step 7 | Assuming Streamlit, since it is Python-only and fast to build. A React front end looks better but adds significant work. |
| Q-07 | Is there a business framing for the PM track, such as an assumed cost per missed fraud and per false alarm? | Step 4 | Assuming none yet. If the PM supplies figures, threshold selection becomes a cost-optimisation exercise rather than a purely statistical one, which is a much stronger story. |

---

## 13. How to resume from nothing

If everything is lost except the GitHub repository:

```powershell
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

# 2. Recreate the environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# 3. Rebuild the data (requires a Kaggle account that has joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py

# 4. Read the current state
code docs/PROJECT_STATE.md
```

If the GitHub repository is also lost, `docs/steps/step1.md` rebuilds everything from scratch.

---

## 14. Glossary

| Term | Plain meaning |
|------|---------------|
| Virtual environment | A private copy of Python belonging to one project, so its packages cannot clash with other projects |
| `.gitignore` | A list of file patterns Git pretends do not exist, so they are never committed |
| Branch | A parallel line of work that can be merged back in when it is ready |
| Pull request | A request to merge a branch, which creates a place to review changes and to run automated checks |
| Tag | A permanent bookmark on one specific commit, used here to mark the end of each step |
| Parquet | A file format for tables that is smaller and faster than CSV and remembers data types |
| Class imbalance | When one outcome is far rarer than the other, here 3.5% fraud against 96.5% legitimate |
| PR-AUC | Precision-Recall Area Under Curve. The right headline metric for rare events, because it ignores the huge easy negative class that inflates accuracy and ROC-AUC |
| MLflow | A tool that records every training run: settings used, metrics produced, and the model file itself |
| SHAP | A method that explains which features pushed a single prediction up or down |
| Drift | When live data slowly stops resembling training data, so the model quietly gets worse |
| CI/CD | Automated checks and deployment that run on every code change |
| Model registry | A catalogue of trained model versions with a record of which one is live |

---

*End of PROJECT_STATE.md. Next: Step 2, EDA and data understanding.*
