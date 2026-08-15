# Step 1: Foundations
### Dataset acquisition, project scaffold, GitHub repo, Python environment

**Project:** IEEE-CIS Fraud Detection (follow-on to NovaPay)
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Platform:** Windows, VS Code, PowerShell
**Estimated time:** 60 to 90 minutes, most of it waiting on the download and the install
**Step 1 of 7**
**Document version:** 1.1, revised after the step was completed and verified

---

## Changelog: what changed in version 1.1

Version 1.0 was written before the step was run. This version records what actually happened, so the document matches reality and can be followed by anyone rebuilding from scratch.

| Section | Change |
|---------|--------|
| 1 | Decision D-02 updated: the repository is `ieee-cis-fraud-detection`, and the local folder currently differs. Explained rather than hidden. |
| 2.4 | Disk space figures confirmed against the real download. |
| 4.3 | Added a warning about picking a location outside OneDrive, and about paths containing spaces. Version 1.0 mentioned this only in passing. |
| 6.5 | Exact installed versions now recorded in Section 6.7, rather than left as "latest available". |
| 6.7 | New section listing the confirmed environment. |
| 8.5 | Expected verification output replaced with the actual verified output. |
| 8.6 | Added a note that `verify_data.py` writes its report to the project root, which is how the local folder name was discovered. |
| 11 | Checklist marked as completed, with the actual figures. |
| 12 | Added three troubleshooting rows discovered during the real run. |
| 13 | Updated: this information has now been supplied. |

---

## 0. How to use this document

Work top to bottom. Do not skip ahead. Each section tells you three things:

1. **Why** we are doing it, so you can explain it later on the PM track
2. **What** to type or create
3. **How to check** it worked before moving on

Where you see a code block labelled `powershell`, that goes into the VS Code terminal. Where you see a block labelled `python`, `text`, `yaml`, or `markdown`, that is file content you create and paste.

At the very end there is a checklist. Do not start Step 2 until every box ticks.

---

## 1. Decisions made in this step

These are locked in now so the rest of the project has a stable base. Each one is repeated in `PROJECT_STATE.md` so you never lose them.

| ID | Decision | Why |
|----|----------|-----|
| D-01 | Dataset is IEEE-CIS Fraud Detection from Kaggle | 590,540 training rows with 3.4990% fraud, so 20,663 fraud cases. NovaPay had 193 fraud cases in 9,940 rows, which was not enough signal to train a strong model. |
| D-02 | Repository name: `ieee-cis-fraud-detection` | Descriptive and instantly readable to a recruiter or hiring manager scanning your GitHub. **Note as built:** the local folder ended up named `IEEEE_CIS_fraud_project`, which does not match. Nothing breaks, because of D-07, but Step 2 Section 2 offers a tidy-up. |
| D-03 | Python 3.11 | The Kaggle CLI now requires Python 3.11 or newer. 3.11 also has stable prebuilt Windows packages for every library we need (LightGBM, XGBoost, CatBoost, SHAP). Newer versions sometimes lag on prebuilt packages, which means slow or failed installs on Windows. |
| D-04 | Virtual environment with `venv`, not conda | NovaPay used conda. We switch because `venv` plus `requirements.txt` is what Docker expects in Step 6 and what GitHub Actions expects in Step 5. One dependency file, used everywhere, is less to keep in sync. |
| D-05 | Data is never committed to Git | The extracted dataset is roughly 1.3 GB. GitHub rejects files over 100 MB. Instead the repo contains a download script, so anyone can rebuild the data folder in one command. |
| D-06 | DVC (data version control) deferred to Step 3 | It needs somewhere to store the data remotely. We will decide on that when we actually have processed datasets worth versioning. The folder layout is already DVC-friendly. |
| D-07 | Paths resolved dynamically in `config/config.py` | NovaPay hardcoded an absolute Windows path. That breaks on any other machine, inside Docker, and in CI. We compute the project root from the file's own location instead. **This decision immediately paid off:** the local folder name does not match the repository name, and nothing had to change. |
| D-08 | Branch strategy: `main` plus one short-lived branch per step, merged by pull request | Even solo, this gives you a reviewable trail per step, and in Step 5 it is what triggers the CI pipeline. It is also a clean story for your PM walkthrough. |
| D-09 | Public GitHub repo | It is a portfolio piece. Public also unlocks free GitHub Actions minutes and branch rulesets. |

---

## 2. Prerequisites

### 2.1 What you need installed

Open PowerShell (in VS Code: `Ctrl` + `` ` ``) and run these one at a time. Each one prints a version number if the tool is installed.

The `py --list` command is a Windows-only helper called the Python Launcher. It shows every Python version on the machine, which matters because you may have several.

```powershell
# Show every Python version installed on this machine
py --list

# Confirm Python 3.11 specifically is available
py -3.11 --version

# Confirm Git is installed
git --version

# Confirm VS Code can be launched from the terminal
code --version
```

**Expected output:** `py -3.11 --version` prints something like `Python 3.11.9`. `git --version` prints something like `git version 2.45.0`.

### 2.2 If something is missing

| Missing | Fix |
|---------|-----|
| Python 3.11 | Download from python.org/downloads. During install, tick **Add python.exe to PATH** and tick **py launcher**. |
| Git | Download from git-scm.com/download/win. Accept all defaults. |
| `code` command | Open VS Code, press `Ctrl+Shift+P`, type `shell command`, choose "Install 'code' command in PATH". |

### 2.3 Accounts you need

1. A **Kaggle account** at kaggle.com (free)
2. A **GitHub account** at github.com (free)

### 2.4 Disk space

Confirmed against the real download:

- Raw data: roughly **1.29 GB** across five CSV files
- Python environment: roughly **2.5 GB**, because the machine learning libraries are large
- Interim Parquet files created in Step 2: a further **500 to 800 MB**

Allow **5 GB free** to be comfortable.

---

## 3. Kaggle setup and dataset access

This is the part that trips people up, so we do it carefully and in order.

### 3.1 Understand the gate

IEEE-CIS is a **competition** dataset, not a plain public dataset. Kaggle will not let the API download competition files until your account has **joined the competition and accepted its rules**. If you skip this, the API returns a `403 Forbidden` error even though your credentials are perfectly valid. The error message does not always say why, which is why people lose an hour here.

So: join first, download second. Always.

### 3.2 Join the competition and accept the rules

1. Log in to Kaggle in your browser.
2. Go to: `https://www.kaggle.com/competitions/ieee-fraud-detection/rules`
3. Read the rules page. Scroll to the bottom.
4. Click the button that says **I Understand and Accept**.
5. Confirm it worked: go to `https://www.kaggle.com/competitions/ieee-fraud-detection/data`. If you can see a **Download All** button and the file list, you are in. If it still asks you to accept rules, you are not.

> The competition is closed for scoring, which is fine. Joining a closed competition is still allowed and still grants data access. We are not submitting to a leaderboard, we are using the data.

**Note for your PM track:** the rules permit non-commercial and academic use of this data. A portfolio project is fine. Worth mentioning when you walk through data provenance.

### 3.3 Get your API credentials

The Kaggle command line tool has been updated recently and now offers several ways to log in. Try them in this order.

**Option A (simplest, recommended).** Once the `kaggle` package is installed later in Section 6, you run one command and a browser window opens for you to approve access:

```powershell
# Opens a browser, you approve, credentials are stored for you
kaggle auth login
```

**Option B (the classic file method).** Use this if Option A fails or if you want credentials that work offline.

1. Go to `https://www.kaggle.com/settings/api`
2. Find the section named **Legacy API Credentials**
3. Click **Create Legacy API Key**. A file named `kaggle.json` downloads.
4. Create the folder Kaggle looks in, and move the file there.

The command below creates a hidden folder called `.kaggle` in your Windows user directory. `$env:USERPROFILE` is a built-in variable that expands to `C:\Users\YourName`, so this works no matter what your username is, including when it contains a space.

```powershell
# Create the folder Kaggle expects credentials to live in
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null

# Move the downloaded file into it (adjust the source path if your Downloads folder differs)
Move-Item -Path "$env:USERPROFILE\Downloads\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\kaggle.json" -Force

# Confirm the file is now in the right place
Test-Path "$env:USERPROFILE\.kaggle\kaggle.json"
```

**Expected output:** `True`

**Option C (environment variable).** Useful later for CI in Step 5, where there is no browser and no file. Copy the token string from the API settings page and set it:

```powershell
# Sets the token for the current PowerShell window only
$env:KAGGLE_API_TOKEN = "paste_your_token_here"
```

### 3.4 Security warning

`kaggle.json` is a password in disguise. Anyone who has it can act as you on Kaggle.

- Never put it inside the project folder.
- Never commit it. Our `.gitignore` blocks it by name as a safety net, but the real protection is keeping it in `C:\Users\YourName\.kaggle\`.
- If you ever paste it into a chat, a screenshot, or a video for the PM walkthrough, go to the Kaggle settings page and click **Expire Token** immediately, then create a new one.

We verify that authentication actually works in Section 8, after the environment exists.

---

## 4. Project folder scaffold

### 4.1 The full structure

This is the shape the project holds for all 7 steps. Folders marked with a step number are created empty now and filled in later. Creating them now means the structure never has to be reorganised mid-project, and every step knows exactly where its output belongs.

```
ieee-cis-fraud-detection/
│
├── .github/
│   └── workflows/                  # CI/CD pipeline definitions        (Step 5)
│
├── .vscode/
│   └── settings.json               # shared editor settings
│
├── app/                            # dashboard application             (Step 7)
│
├── config/
│   ├── __init__.py
│   └── config.py                   # every path and constant, one place
│
├── data/
│   ├── raw/                        # untouched Kaggle CSVs, never edited
│   ├── interim/                    # joined and cleaned, mid-pipeline   (Step 2)
│   ├── processed/                  # model-ready feature tables        (Step 3)
│   └── external/                   # anything not from Kaggle
│
├── docker/                         # Dockerfile and compose files       (Step 6)
│
├── docs/
│   ├── PROJECT_STATE.md            # the anchor document
│   ├── steps/                      # step1.md ... step7.md
│   └── decisions/                  # architecture decision records
│
├── models/                         # trained model files               (Step 4)
│
├── notebooks/                      # exploration only, never production
│
├── reports/
│   ├── figures/                    # charts for the README and dashboard
│   └── explainability/             # SHAP plots, importance charts     (Step 4)
│
├── scripts/
│   ├── download_data.py            # pulls the dataset from Kaggle
│   └── verify_data.py              # proves the download is correct
│
├── src/
│   ├── __init__.py
│   ├── pipelines/                  # one module per pipeline stage
│   │   └── __init__.py
│   ├── serving/                    # FastAPI prediction service        (Step 6)
│   │   └── __init__.py
│   ├── monitoring/                 # drift detection, retraining       (Step 5)
│   │   └── __init__.py
│   └── utils/                      # small reusable helper functions
│       └── __init__.py
│
├── tests/
│   └── __init__.py                 # automated tests                   (Step 5)
│
├── .env.example                    # template for secrets, safe to commit
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt                # what the project needs to run
├── requirements-dev.txt            # what you need to develop it
└── run.py                          # single pipeline entry point       (Step 2)
```

### 4.2 Why each top-level folder exists

| Folder | Purpose | Why it is separate |
|--------|---------|--------------------|
| `.github/workflows/` | GitHub Actions configuration files. | GitHub only looks in this exact path. Non-negotiable location. |
| `.vscode/` | Editor settings shared with anyone who clones the repo. | Guarantees the same interpreter, formatter, and test runner for every contributor. Removes "works on my machine" arguments. |
| `app/` | The user-facing dashboard. | The dashboard is a consumer of the model, not part of the model. Keeping it out of `src/` means you can containerise the API without dragging dashboard code along. |
| `config/` | Paths, constants, hyperparameter defaults. | The single place to change a setting. When a path is wrong, you fix it once instead of hunting through fifteen files. |
| `data/` with four subfolders | Raw, interim, processed, external. | This is the standard data science layout. `raw/` is treated as read-only and is never modified, so you can always rebuild everything downstream from a known starting point. `interim/` holds work in progress, such as the joined tables. `processed/` holds only what the model actually consumes. |
| `docker/` | Dockerfiles and compose files. | Keeps container configuration out of the project root, which stays readable. |
| `docs/` | The project state file, per-step guides, decision records. | Documentation is a deliverable here, not an afterthought. This folder is what makes the PM walkthrough easy. |
| `models/` | Serialised trained models. | Model files are binary and large. Keeping them in one folder makes them simple to ignore in Git and simple to mount into a container. |
| `notebooks/` | Jupyter notebooks for exploring. | Notebooks are for thinking, not for running in production. Anything that matters gets rewritten as a proper module in `src/`. This separation is one of the clearest signals of a professional project. |
| `reports/` | Generated charts, metrics tables, summaries. | Outputs meant to be read by a human, as opposed to `models/` which is read by machines. These are small and get committed, because they make the README look good. |
| `scripts/` | Standalone runnable utilities. | Things you run once in a while (download data, verify data) rather than every pipeline run. Separating them keeps `src/` focused on the pipeline itself. |
| `src/` | All reusable project code. | The importable heart of the project. Subdivided by responsibility: pipelines transform data, serving exposes the model, monitoring watches it, utils supports the rest. We carry forward the `pipelines` / `serving` / `utils` split from NovaPay because it worked, and add `monitoring` because Step 5 needs it. |
| `tests/` | Automated tests. | Test tools look here by default. Having real tests is what separates a project with CI from a project with a CI badge. |

### 4.3 Choose where the project lives

Two things about the location matter more than they look.

**Avoid folders that sync to OneDrive, Dropbox, or Google Drive.** The sync client watches every file and uploads changes. This project writes gigabytes: 1.3 GB of raw CSV now, several hundred megabytes of Parquet in Step 2, thousands of small MLflow files in Step 4, and model binaries after that. Three specific problems follow. The sync client can briefly hold a file open while uploading, so your script's write fails with a permission error that works fine on the next attempt. The client can also replace a local file with a small placeholder to save space, which makes reads slow or fails them outright. And if it is a company account, you are uploading public Kaggle data into corporate storage that is not meant for it.

**Avoid spaces in the path if you can.** Python handles them fine because we use `Path` objects. But every terminal command that mentions the path has to wrap it in double quotes, and forgetting one quote produces a confusing error.

A simple path like `C:\projects` avoids both problems.

> **Note as built:** this project was actually created at `C:\Users\Dauda Agbonoga\OneDrive - Venture Garden Group\Documents\my\IEEEE_CIS_fraud_project`, which has both issues. Everything in Step 1 worked regardless. Step 2 Section 2 sets out how to move it, and why moving requires rebuilding `.venv` rather than just dragging the folder.

```powershell
# Create a projects folder if you do not already have one, then go into it
New-Item -ItemType Directory -Force -Path "C:\projects" | Out-Null
Set-Location "C:\projects"

# Create the project folder and enter it
New-Item -ItemType Directory -Force -Path "ieee-cis-fraud-detection" | Out-Null
Set-Location "ieee-cis-fraud-detection"

# Print where you are, so you can confirm
Get-Location
```

**Expected output:** `C:\projects\ieee-cis-fraud-detection`

### 4.4 Create the scaffold

Now create every folder. We put the folder names in a list variable first, then loop through it. This is easier to read and easier to edit than twenty separate commands.

```powershell
# The list of every folder the project needs
$folders = @(
    ".github\workflows",
    ".vscode",
    "app",
    "config",
    "data\raw",
    "data\interim",
    "data\processed",
    "data\external",
    "docker",
    "docs\steps",
    "docs\decisions",
    "models",
    "notebooks",
    "reports\figures",
    "reports\explainability",
    "scripts",
    "src\pipelines",
    "src\serving",
    "src\monitoring",
    "src\utils",
    "tests"
)

# Loop through the list and create each folder.
# Out-Null hides the confirmation message so the terminal stays readable.
foreach ($folder in $folders) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
}

Write-Host "Folders created." -ForegroundColor Green
```

Next, create the empty marker files. Two kinds:

- `__init__.py` tells Python "this folder is a package you can import from". Without it, `from src.pipelines.ingestion import ...` fails.
- `.gitkeep` is a convention, not a real feature. Git refuses to track empty folders, so we drop a tiny empty file inside each one we want to keep in the repo.

```powershell
# Files that make folders importable by Python, plus placeholders that keep
# empty folders visible in Git
$files = @(
    "config\__init__.py",
    "src\__init__.py",
    "src\pipelines\__init__.py",
    "src\serving\__init__.py",
    "src\monitoring\__init__.py",
    "src\utils\__init__.py",
    "tests\__init__.py",
    "data\raw\.gitkeep",
    "data\interim\.gitkeep",
    "data\processed\.gitkeep",
    "data\external\.gitkeep",
    "models\.gitkeep",
    "reports\figures\.gitkeep",
    "reports\explainability\.gitkeep",
    "docs\decisions\.gitkeep",
    "notebooks\.gitkeep",
    "app\.gitkeep",
    "docker\.gitkeep",
    ".github\workflows\.gitkeep"
)

# Only create a file if it does not already exist, so nothing gets wiped
foreach ($file in $files) {
    if (-not (Test-Path $file)) {
        New-Item -ItemType File -Force -Path $file | Out-Null
    }
}

Write-Host "Placeholder files created." -ForegroundColor Green
```

### 4.5 Check it worked

`tree` is a Windows command that draws the folder structure. `/A` uses plain characters that display reliably in every terminal.

```powershell
# Draw the folder structure (folders only)
tree /A
```

**Expected output:** a tree matching Section 4.1, minus the files.

Now open the project in VS Code so you can create files by right-clicking instead of typing paths:

```powershell
# Open the current folder in VS Code
code .
```

---

## 5. Git and GitHub setup

### 5.1 Why we do this before downloading data

The download is 1.3 GB. If Git is initialised **after** the data lands and `.gitignore` is not in place, your very first `git add .` tries to stage 1.3 GB of CSV files. Undoing that is annoying and it can leave large blobs in the repository history forever.

So the order is: Git first, `.gitignore` second, data third. Always.

### 5.2 Create `.gitignore`

In VS Code, right-click in the file explorer panel, choose **New File**, name it exactly `.gitignore` (the leading dot matters), and paste the content below.

A `.gitignore` file lists patterns Git should pretend do not exist. Anything matching is never staged, never committed, never pushed.

```text
# ---------------------------------------------------------
# Python
# ---------------------------------------------------------
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.ruff_cache/

# ---------------------------------------------------------
# Virtual environments
# ---------------------------------------------------------
.venv/
venv/
env/

# ---------------------------------------------------------
# Secrets. Never commit these.
# ---------------------------------------------------------
.env
kaggle.json
*.pem
*.key

# ---------------------------------------------------------
# Data. Too large for Git and fully re-downloadable.
# The .gitkeep exceptions preserve the empty folder structure.
# ---------------------------------------------------------
data/raw/*
data/interim/*
data/processed/*
data/external/*
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep
!data/external/.gitkeep
*.zip

# ---------------------------------------------------------
# Trained models. Large binaries, regenerated by the pipeline.
# ---------------------------------------------------------
models/*
!models/.gitkeep

# ---------------------------------------------------------
# MLflow experiment tracking (Step 4)
# ---------------------------------------------------------
mlruns/
mlartifacts/
mlflow.db

# ---------------------------------------------------------
# CatBoost writes training logs into the working directory
# ---------------------------------------------------------
catboost_info/

# ---------------------------------------------------------
# Editor and OS noise. We keep shared VS Code settings on purpose.
# ---------------------------------------------------------
.vscode/*
!.vscode/settings.json
!.vscode/extensions.json
.idea/
Thumbs.db
.DS_Store
desktop.ini

# ---------------------------------------------------------
# Logs and temporary files
# ---------------------------------------------------------
*.log
tmp/
temp/
```

### 5.3 Create `.env.example`

An `.env` file holds secrets on your machine. It is ignored by Git. The `.example` version has the same keys with the values removed, so anyone cloning the repo knows what to fill in. This is a standard and expected pattern, and reviewers look for it.

Create `.env.example` with this content:

```text
# Copy this file to .env and fill in real values.
# .env is git-ignored and must never be committed.

# Kaggle API. Only needed if you are not using `kaggle auth login`.
KAGGLE_USERNAME=
KAGGLE_KEY=
KAGGLE_API_TOKEN=

# MLflow tracking location. Defaults to a local SQLite file if left blank.
MLFLOW_TRACKING_URI=
```

### 5.4 Create `README.md`

The README is the front door of the project. Recruiters read it and often nothing else. We write the skeleton now with clear placeholders, and fill in real results as each step completes.

```markdown
# IEEE-CIS Fraud Detection

An end-to-end machine learning and MLOps project that detects fraudulent card
transactions in the IEEE-CIS dataset, covering the full lifecycle from raw data
to a monitored, containerised, deployed service with an interactive dashboard.

> Status: in progress. Step 1 of 7 complete.

---

## Problem

Card fraud is rare and expensive. Rare, because roughly 3.5% of transactions in
this dataset are fraudulent, so a model that predicts "never fraud" is still
96.5% accurate and completely useless. Expensive, because every missed fraud is
a direct loss and every false alarm annoys a real customer.

The goal is a model that ranks transactions by fraud risk well enough to be
useful at a realistic review capacity, plus the engineering around it that makes
it deployable, observable, and maintainable.

## Dataset

IEEE-CIS Fraud Detection (Kaggle competition, provided by Vesta Corporation).

| Table | Rows | Columns | Notes |
|-------|------|---------|-------|
| `train_transaction` | 590,540 | 394 | Transaction level, contains the `isFraud` label |
| `train_identity` | 144,233 | 41 | Device and identity signals, only for some transactions |
| `test_transaction` | 506,691 | 393 | No label |
| `test_identity` | 141,907 | 41 | No label |

The two training tables join on `TransactionID`. Only 24.4% of transactions
have a matching identity record, which is itself a signal.

Data is not stored in this repository. See Quickstart to download it.

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
pip install -r requirements.txt -r requirements-dev.txt

# 3. Data (requires a Kaggle account that has joined the competition)
kaggle auth login
python scripts/download_data.py
python scripts/verify_data.py
```

## Project structure

_See `docs/PROJECT_STATE.md` for the annotated structure and current status._

## Roadmap

- [x] Step 1: Dataset acquisition, scaffold, repo, environment
- [ ] Step 2: Exploratory data analysis and data understanding
- [ ] Step 3: Feature engineering and preprocessing pipeline
- [ ] Step 4: Model training with MLflow experiment tracking
- [ ] Step 5: MLOps layer: CI/CD, testing, model registry, drift monitoring
- [ ] Step 6: Dockerisation and deployment
- [ ] Step 7: Dashboard and portfolio packaging

## Results

_Populated in Step 4._

| Metric | Baseline | Best model |
|--------|----------|------------|
| PR-AUC | 0.035 | TBD |
| ROC-AUC | 0.500 | TBD |
| Recall at 1% review rate | TBD | TBD |

## Tech stack

Python 3.11, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, MLflow, SHAP,
FastAPI, Docker, GitHub Actions, Streamlit.

## Background

This project is a follow-on to NovaPay, an earlier fraud detection prototype.
NovaPay's dataset contained only 193 fraud cases across 9,940 transactions,
which capped model performance regardless of technique. The pipeline design
patterns carry forward; the data does not.

## Licence

MIT. See `LICENSE`.
```

### 5.5 Create `LICENSE`

MIT is the usual choice for a portfolio project. It is short and permissive.

Create a file named `LICENSE` (no extension) containing the standard MIT text. The fastest route: go to `https://choosealicense.com/licenses/mit/`, copy the text, paste it in, and replace `[year]` with `2026` and `[fullname]` with your name.

### 5.6 Create `.vscode/settings.json`

These settings make VS Code use the project's own Python environment automatically and behave consistently.

```json
{
  "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe",
  "python.terminal.activateEnvironment": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "python.analysis.extraPaths": ["."],
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "editor.rulers": [88],
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  }
}
```

What these do, in plain terms: point VS Code at the environment we are about to create; activate it automatically in every new terminal; use pytest for tests; treat the project root as importable so `from src...` resolves; clean up whitespace on save; draw a line at 88 characters so lines stay readable; format Python files on save.

### 5.7 Initialise Git and make the first commit

`git init` creates a hidden `.git` folder that starts tracking changes. The `-b main` part names the first branch `main` immediately, rather than the older default name.

```powershell
# Start tracking this folder with Git, naming the first branch "main"
git init -b main

# Tell Git who you are (only needed once per machine, but harmless to repeat)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Show what Git can currently see
git status
```

**Expected output:** a list of untracked files including `README.md`, `.gitignore`, `config/`, `src/`, and so on. You should **not** see anything under `data/` other than `.gitkeep` files.

Stage and commit. `git add .` stages everything not ignored. `git commit` saves a snapshot with a message.

```powershell
# Stage every file that is not ignored
git add .

# Check exactly what is about to be committed, before committing it
git status

# Save the snapshot
git commit -m "chore: initial project scaffold, gitignore, readme, licence"
```

> **Read the `git status` output carefully.** If you see any `.csv` file, stop and fix `.gitignore` before committing. This is the single most common way a repo gets permanently bloated.

### 5.8 Create the GitHub repository and push

**Option A: GitHub CLI (faster).** If you have `gh` installed, it creates the remote repo and pushes in one command. Install it from `https://cli.github.com/` if you want this route.

```powershell
# Log in to GitHub from the terminal (opens a browser)
gh auth login

# Create a public repo from the current folder, add it as "origin", and push
gh repo create ieee-cis-fraud-detection --public --source=. --remote=origin --push
```

**Option B: the website.**

1. Go to `https://github.com/new`
2. Repository name: `ieee-cis-fraud-detection`
3. Visibility: **Public**
4. Do **not** tick "Add a README", "Add .gitignore", or "Choose a licence". You already have all three locally, and adding them on GitHub creates a conflict on first push.
5. Click **Create repository**
6. Back in your terminal, connect the local folder to the GitHub repo and push:

```powershell
# Point the local repo at GitHub (replace with your username)
git remote add origin https://github.com/Dee-ui/ieee-cis-fraud-detection.git

# Push the main branch and set it as the default upstream target
git push -u origin main
```

**Check it worked:** refresh the repository page in your browser. You should see your README rendered.

### 5.9 Branch strategy

The rules, in full:

1. `main` always works. Nothing broken ever lands there.
2. Each step gets one branch, named `step-NN-short-description`.
3. Work happens on the step branch. Commit often.
4. When the step is finished and verified, open a pull request into `main` and merge it.
5. Tag `main` after each merge, so you can jump back to the exact state at the end of any step.

Why bother when you are the only developer:

- In Step 5, CI runs automatically on every pull request. That only works if pull requests exist.
- Your commit history becomes a readable narrative of the build, which is exactly what you want for the PM walkthrough and for anyone reviewing the repo.
- If a step goes badly wrong, you delete the branch instead of untangling `main`.

Create the Step 1 branch now and move your work onto it. `git switch -c` creates a branch and moves to it in one command.

```powershell
# Create the Step 1 branch and switch to it
git switch -c step-01-foundations

# Confirm which branch you are on (the current one has an asterisk)
git branch
```

Everything else in this document is committed on `step-01-foundations`. We merge into `main` at the end, in Section 10.

---

## 6. Python environment

### 6.1 What a virtual environment is and why it matters

A virtual environment is a private copy of Python that belongs to this project only. Packages installed inside it are invisible to everything else on your machine.

Without one, every project shares one global Python. Project A needs pandas 1.5, project B needs pandas 2.2, and installing one breaks the other. With one, each project pins exactly what it needs. It also means `pip freeze` produces an accurate list of what this project actually depends on, which is what Docker and CI consume later.

One property to remember for later: **a virtual environment contains absolute paths written inside it, so it cannot be moved.** If you relocate the project folder, delete `.venv` and rebuild it. Step 2 covers exactly how.

### 6.2 Create and activate it

The folder `.venv` is the conventional name. VS Code detects it automatically, and our `.gitignore` already excludes it.

```powershell
# Create a virtual environment named .venv using Python 3.11
py -3.11 -m venv .venv
```

Now activate it. Activating means "for this terminal window, `python` and `pip` refer to the environment's copies, not the global ones".

```powershell
# Turn the environment on for this terminal
.\.venv\Scripts\Activate.ps1
```

**Expected output:** your prompt gains a `(.venv)` prefix.

**If you get a red error mentioning "running scripts is disabled on this system":** Windows blocks scripts by default. This command allows them for the current terminal session only, which is the safest scope. Run it, then activate again.

```powershell
# Allow scripts in this PowerShell window only, then activate
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
```

If you would rather not repeat that each time, this sets it permanently for your user account (still safe, it only allows local scripts and signed remote ones):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Confirm the environment is really the one being used. `Get-Command python` shows the full path of the `python` that will run.

```powershell
# Should point inside your project's .venv folder
Get-Command python | Select-Object -ExpandProperty Source

# Should print Python 3.11.x
python --version
```

### 6.3 Create `requirements.txt`

This file lists what the **project itself** needs in order to run: load data, train, serve predictions, show a dashboard. Step 6 copies this file into the Docker image.

We use `>=` (at least this version) rather than `==` (exactly this version). Reason: `==` on Windows sometimes pins a version with no prebuilt package available, which forces a slow compile-from-source or fails outright. We get exact reproducibility a different way, with a lock file, in Section 6.5.

```text
# ---------------------------------------------------------
# Data handling
# ---------------------------------------------------------
pandas>=2.2              # tables, joins, groupbys, the core of everything
numpy>=1.26              # numeric arrays underneath pandas
pyarrow>=16.0            # fast Parquet files, much smaller and faster than CSV
scipy>=1.13              # statistics used in EDA and feature selection

# ---------------------------------------------------------
# Modelling
# ---------------------------------------------------------
scikit-learn>=1.5        # pipelines, metrics, cross-validation, baselines
lightgbm>=4.3            # gradient boosting, fast and strong on tabular data
xgboost>=2.1             # gradient boosting, the usual benchmark competitor
catboost>=1.2            # gradient boosting, handles categorical columns natively
imbalanced-learn>=0.12   # resampling techniques for rare-event problems

# ---------------------------------------------------------
# Experiment tracking and explainability
# ---------------------------------------------------------
mlflow>=2.16             # logs every run: parameters, metrics, artifacts, models
shap>=0.46               # explains why the model flagged a specific transaction

# ---------------------------------------------------------
# Visualisation
# ---------------------------------------------------------
matplotlib>=3.9          # base plotting library
seaborn>=0.13            # statistical charts with far less code
plotly>=5.24             # interactive charts for the dashboard

# ---------------------------------------------------------
# Serving and dashboard
# ---------------------------------------------------------
fastapi>=0.115           # the prediction API
uvicorn[standard]>=0.30  # the web server that runs FastAPI
pydantic>=2.8            # validates incoming request data against a schema
streamlit>=1.38          # the dashboard framework

# ---------------------------------------------------------
# Supporting utilities
# ---------------------------------------------------------
joblib>=1.4              # saves and loads trained model files
pyyaml>=6.0              # reads configuration files
python-dotenv>=1.0       # loads variables from a .env file
```

### 6.4 Create `requirements-dev.txt`

This file lists tools you need while **building** the project but which the running application does not need. Keeping them separate means the Docker image in Step 6 stays smaller, and it makes the dependency story easy to explain.

```text
# Everything in the runtime requirements, plus development tooling.
-r requirements.txt

# ---------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------
kaggle>=1.7              # the Kaggle command line tool and API client

# ---------------------------------------------------------
# Notebooks
# ---------------------------------------------------------
jupyter>=1.1             # notebook server
ipykernel>=6.29          # lets VS Code run notebooks against this environment
nbstripout>=0.7          # strips notebook output before committing, keeps diffs small

# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------
pytest>=8.3              # the test runner
pytest-cov>=5.0          # measures how much code the tests actually touch
httpx>=0.27              # used to test the FastAPI endpoints in Step 5

# ---------------------------------------------------------
# Code quality
# ---------------------------------------------------------
ruff>=0.6                # very fast linter and formatter, replaces flake8 and isort
black>=24.8              # opinionated code formatter
pre-commit>=3.8          # runs the checks automatically before each commit (Step 5)
```

### 6.5 Install everything

The first command upgrades pip itself, which avoids a batch of confusing warnings. The second installs the development file, and because that file begins with `-r requirements.txt`, it installs both.

This takes 5 to 15 minutes. The gradient boosting libraries are large.

```powershell
# Upgrade the installer itself first
python -m pip install --upgrade pip

# Install runtime and development dependencies in one go
pip install -r requirements-dev.txt
```

Now create the lock file. `pip freeze` prints every installed package with its exact version. Saving that gives you a perfect snapshot: if a library releases a breaking change in three months, you can still rebuild exactly what worked today.

```powershell
# Record the exact versions that got installed
pip freeze > requirements.lock.txt
```

Check the key libraries imported correctly. This is worth doing, because a failed compile on Windows sometimes leaves a package half-installed rather than clearly failing.

```powershell
python -c "import pandas, sklearn, lightgbm, xgboost, catboost, mlflow, shap; print('all imports OK')"
```

**Expected output:** `all imports OK`

### 6.6 Point VS Code at the environment

1. Press `Ctrl+Shift+P`
2. Type `Python: Select Interpreter`
3. Choose the one whose path contains `.venv`

The version and environment name now show in the bottom status bar. If a terminal was open before you did this, close it and open a new one so it picks up the change.

### 6.7 What actually got installed

These are the confirmed versions on this build. Step 2 code was written against them specifically, and three of them changed how that code had to be written.

| Library | Version | Note |
|---------|---------|------|
| pandas | 2.3.3 | Requires `observed=True` on category groupbys to avoid a future-default warning |
| numpy | 2.4.6 | numpy 2.x, so `np.NaN` and `np.float_` no longer exist |
| pyarrow | 24.0.0 | Parquet engine |
| scipy | 1.17.1 | |
| scikit-learn | 1.9.0 | |
| lightgbm | 4.7.0 | |
| xgboost | 3.2.0 | |
| catboost | 1.2.10 | |
| imbalanced-learn | 0.14.2 | |
| mlflow | 3.15.1 | MLflow 3, not 2. The API differs in places, which matters in Step 4 |
| shap | 0.51.0 | |
| matplotlib | 3.11.1 | `plt.cm.get_cmap` was removed in 3.9, so no project code uses it |
| seaborn | 0.13.2 | |
| plotly | 6.9.0 | |
| fastapi | 0.141.1 | |
| uvicorn | 0.52.3 | |
| streamlit | 1.61.1 | |
| pytest | 9.1.1 | |
| ruff | 0.16.3 | |
| black | 26.5.1 | |
| pre-commit | 4.6.2 | |
| kaggle | 2.2.4 | Current CLI, positional competition argument |

The full list is in `requirements.lock.txt`. That file, not `requirements.txt`, is what rebuilds an identical environment.

---

## 7. The configuration module

### 7.1 Why this file exists

Every script needs to know where the data is, where models go, what the random seed is. If each script hardcodes that, changing one folder means editing ten files and missing three.

`config/config.py` is the single source of truth. Everything else imports from it.

The important improvement over NovaPay: NovaPay hardcoded an absolute path like `C:\Users\Name\OneDrive\...`. That works on exactly one machine. It breaks the moment the project runs inside Docker, inside GitHub Actions, or on anyone else's laptop. We compute the project root from the location of the config file itself, so it is correct everywhere, always.

> This paid off immediately. The local folder ended up named `IEEEE_CIS_fraud_project` while the repository is `ieee-cis-fraud-detection`, and not one line of code had to change.

### 7.2 Create `config/config.py`

Paste this into `config/config.py` (the file already exists as an empty placeholder, so just open it).

Step 2 extends this file substantially. This is the Step 1 version.

```python
"""
Central configuration for the IEEE-CIS Fraud Detection project.

Every path and every global constant lives here. No other module should
build its own file paths. Import from this file instead.
"""

import os
from pathlib import Path

# ---------------------------------------------------------
# Project root, resolved dynamically.
#
# __file__ is the path of this file: <root>/config/config.py
# .resolve() turns it into a full absolute path
# .parents[0] is the config folder, .parents[1] is the project root
#
# This is why the project works on any machine, in Docker, and in CI.
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# Reproducibility
#
# One seed used everywhere means two runs of the same code give the
# same numbers. Without it you cannot tell a real improvement from
# random luck.
# ---------------------------------------------------------

RANDOM_SEED = 42


# ---------------------------------------------------------
# Kaggle source
# ---------------------------------------------------------

KAGGLE_COMPETITION = "ieee-fraud-detection"


# ---------------------------------------------------------
# Data directories
#
# The "/" operator on Path objects joins paths correctly on every
# operating system. No backslash-versus-forward-slash problems.
# ---------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"


# ---------------------------------------------------------
# Raw input files, exactly as Kaggle names them
# ---------------------------------------------------------

TRAIN_TRANSACTION_FILE = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_FILE = RAW_DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_FILE = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_FILE = RAW_DATA_DIR / "test_identity.csv"
SAMPLE_SUBMISSION_FILE = RAW_DATA_DIR / "sample_submission.csv"

# Every file the download is expected to produce. Used by verification.
EXPECTED_RAW_FILES = [
    TRAIN_TRANSACTION_FILE,
    TRAIN_IDENTITY_FILE,
    TEST_TRANSACTION_FILE,
    TEST_IDENTITY_FILE,
    SAMPLE_SUBMISSION_FILE,
]


# ---------------------------------------------------------
# Key column names
#
# Naming these once avoids typos scattered through the codebase.
# ---------------------------------------------------------

TARGET_COLUMN = "isFraud"          # 1 means fraud, 0 means legitimate
ID_COLUMN = "TransactionID"        # unique row identifier
JOIN_KEY = "TransactionID"         # links transaction and identity tables
TIME_COLUMN = "TransactionDT"      # seconds since an unknown reference point
AMOUNT_COLUMN = "TransactionAmt"   # transaction value


# ---------------------------------------------------------
# Pipeline stage outputs (populated in Steps 2 and 3)
# ---------------------------------------------------------

JOINED_TRAIN_FILE = INTERIM_DATA_DIR / "train_joined.parquet"
JOINED_TEST_FILE = INTERIM_DATA_DIR / "test_joined.parquet"
FEATURES_TRAIN_FILE = PROCESSED_DATA_DIR / "train_features.parquet"
FEATURES_TEST_FILE = PROCESSED_DATA_DIR / "test_features.parquet"


# ---------------------------------------------------------
# Model and report directories
# ---------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPLAINABILITY_DIR = REPORTS_DIR / "explainability"


# ---------------------------------------------------------
# MLflow experiment tracking (configured properly in Step 4)
#
# os.getenv reads an environment variable and falls back to the second
# argument if it is not set. That lets CI and Docker point MLflow
# somewhere else without changing this file.
# ---------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)
MLFLOW_EXPERIMENT_NAME = "ieee-cis-fraud-detection"


# ---------------------------------------------------------
# Modelling defaults (revisited in Step 4)
# ---------------------------------------------------------

TEST_SIZE = 0.2                      # share of data held out for evaluation
CV_FOLDS = 5                         # cross-validation splits
FRAUD_PROBABILITY_THRESHOLD = 0.5    # placeholder, tuned properly in Step 4


# ---------------------------------------------------------
# Helper: make sure every output folder exists before writing to it
# ---------------------------------------------------------

def ensure_directories() -> None:
    """Create all output folders if they are missing. Safe to call repeatedly."""
    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        EXPLAINABILITY_DIR,
    ]
    for directory in directories:
        # parents=True also creates any missing parent folders
        # exist_ok=True means "do nothing if it is already there"
        directory.mkdir(parents=True, exist_ok=True)
```

### 7.3 Check it works

This runs Python, imports the config, and prints the project root it calculated.

```powershell
python -c "from config.config import PROJECT_ROOT, RAW_DATA_DIR; print(PROJECT_ROOT); print(RAW_DATA_DIR)"
```

**Expected output:** two lines, the second ending in `\data\raw`.

If you get `ModuleNotFoundError: No module named 'config'`, you are not in the project root folder. Run `Get-Location` and `Set-Location` back to it.

---

## 8. Download the dataset

### 8.1 What you are about to download

| File | Rows | Columns | Contains |
|------|------|---------|----------|
| `train_transaction.csv` | 590,540 | 394 | Transaction details plus the `isFraud` label |
| `train_identity.csv` | 144,233 | 41 | Device, browser, and network signals |
| `test_transaction.csv` | 506,691 | 393 | Same as train, no label |
| `test_identity.csv` | 141,907 | 41 | Same as train identity |
| `sample_submission.csv` | 506,691 | 2 | Kaggle's submission format template |

The download is roughly 120 MB compressed and expands to roughly 1.29 GB.

Two things to know now, because they cause confusion later:

1. **Not every transaction has an identity record.** Only 144,233 of 590,540 do, which is 24.4%. The join in Step 2 is a left join, and the missing identity data is itself informative rather than a defect.
2. **The test identity file uses different column punctuation.** Training uses `id_01`, `id_02` and so on with underscores. The test file uses `id-01`, `id-02` with hyphens. This is a known quirk in the released files. We handle it explicitly in Step 2 rather than being surprised by it.

### 8.2 Create `scripts/download_data.py`

A design note first. The Kaggle Python library's internal interface has changed between versions, so code written against it can break on upgrade. The command line interface is the documented, stable contract. So this script calls the command line tool using `subprocess`, which is Python's way of running a terminal command from inside a program. That makes the script resilient to library changes.

```python
"""
Download the IEEE-CIS Fraud Detection dataset from Kaggle into data/raw.

Prerequisites:
  1. A Kaggle account that has JOINED the competition and accepted its rules.
     Without this, the download fails with a 403 error.
  2. Kaggle credentials set up (run `kaggle auth login`, or place kaggle.json
     in your user folder under .kaggle).

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --force     # re-download even if files exist
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Add the project root to the list of places Python looks for imports.
# Without this line, running the script directly cannot find the config package,
# because Python only looks in the script's own folder by default.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import (  # noqa: E402
    KAGGLE_COMPETITION,
    RAW_DATA_DIR,
    EXPECTED_RAW_FILES,
    ensure_directories,
)


def check_kaggle_cli_available() -> None:
    """Confirm the kaggle command exists before trying to use it."""
    # shutil.which searches the system PATH for an executable and returns
    # its location, or None if it is not found.
    if shutil.which("kaggle") is None:
        print("ERROR: the 'kaggle' command was not found.")
        print("Fix: activate your virtual environment, then run:")
        print("     pip install kaggle")
        sys.exit(1)  # exit code 1 means "finished with an error"

    print("Kaggle CLI found.")


def files_already_present() -> bool:
    """Return True only if every expected CSV is already on disk."""
    # all() returns True when every item in the list is True.
    return all(file_path.exists() for file_path in EXPECTED_RAW_FILES)


def download_competition_files() -> Path:
    """
    Run the Kaggle CLI to download the competition archive.

    Returns the path to the downloaded zip file.
    """
    print(f"\nDownloading '{KAGGLE_COMPETITION}' into {RAW_DATA_DIR} ...")
    print("This is roughly 120 MB and may take a few minutes.\n")

    # The command as a list of pieces. Passing a list rather than one long
    # string avoids problems with spaces in folder names.
    #   -p  : where to put the download
    #   -o  : overwrite anything already there
    command = [
        "kaggle",
        "competitions",
        "download",
        KAGGLE_COMPETITION,
        "-p",
        str(RAW_DATA_DIR),
        "-o",
    ]

    # check=False means "do not raise an exception automatically", so we can
    # print a helpful message ourselves instead of a raw stack trace.
    result = subprocess.run(command, check=False)

    # Older versions of the CLI expect the competition name after -c.
    # If the first attempt failed, try that older form before giving up.
    if result.returncode != 0:
        print("\nFirst attempt failed. Retrying with the older -c flag syntax ...")
        legacy_command = [
            "kaggle",
            "competitions",
            "download",
            "-c",
            KAGGLE_COMPETITION,
            "-p",
            str(RAW_DATA_DIR),
            "-o",
        ]
        result = subprocess.run(legacy_command, check=False)

    if result.returncode != 0:
        print("\nERROR: the download failed.")
        print("Most common causes, in order of likelihood:")
        print("  1. You have not joined the competition and accepted its rules.")
        print(f"     Go to https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/rules")
        print("  2. Your credentials are missing or expired. Run: kaggle auth login")
        print("  3. No internet connection, or a proxy is blocking the request.")
        sys.exit(1)

    zip_path = RAW_DATA_DIR / f"{KAGGLE_COMPETITION}.zip"
    if not zip_path.exists():
        print(f"ERROR: expected {zip_path} after download, but it is not there.")
        sys.exit(1)

    print(f"\nDownloaded: {zip_path.name}")
    return zip_path


def extract_archive(zip_path: Path) -> None:
    """Unzip the archive into data/raw and then delete the zip."""
    print(f"\nExtracting {zip_path.name} ...")

    # "with" makes sure the zip file is closed properly even if an error occurs.
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.namelist()
        for index, member in enumerate(members, start=1):
            print(f"  [{index}/{len(members)}] {member}")
            archive.extract(member, RAW_DATA_DIR)

    # The zip is no longer needed and takes up 120 MB.
    zip_path.unlink()
    print("\nExtraction complete. Archive removed.")


def report_results() -> None:
    """Print each expected file with its size, and flag anything missing."""
    print("\n" + "=" * 60)
    print("FILES IN data/raw")
    print("=" * 60)

    missing = []
    for file_path in EXPECTED_RAW_FILES:
        if file_path.exists():
            # st_size is in bytes. Divide twice by 1024 to reach megabytes.
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  OK       {file_path.name:<28} {size_mb:>9.1f} MB")
        else:
            print(f"  MISSING  {file_path.name}")
            missing.append(file_path.name)

    if missing:
        print(f"\nWARNING: {len(missing)} expected file(s) missing: {missing}")
        sys.exit(1)

    print("\nAll expected files are present.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the IEEE-CIS Fraud Detection dataset from Kaggle."
    )
    parser.add_argument(
        "--force",
        action="store_true",  # makes it a simple on/off flag with no value
        help="Download again even if the files already exist.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("IEEE-CIS FRAUD DETECTION: DATA DOWNLOAD")
    print("=" * 60)

    ensure_directories()
    check_kaggle_cli_available()

    if files_already_present() and not args.force:
        print("\nAll files are already present. Nothing to do.")
        print("Use --force to download them again.")
        report_results()
        return

    zip_path = download_competition_files()
    extract_archive(zip_path)
    report_results()

    print("\nNext: run  python scripts/verify_data.py")


# This guard means the code only runs when the file is executed directly,
# not when it is imported by another module.
if __name__ == "__main__":
    main()
```

### 8.3 Run the download

Make sure your virtual environment is active (you see `(.venv)` in the prompt) and you are in the project root.

```powershell
python scripts/download_data.py
```

**Expected output:** a progress bar, then an extraction list, then a table of five files with sizes, then `All expected files are present.`

**If you get a 403 error:** you have not joined the competition. Go back to Section 3.2. This is not a credentials problem, so regenerating your token will not help.

### 8.4 Create `scripts/verify_data.py`

Downloading is not the same as having correct data. This script proves the files are complete and sound, and writes a small inventory report that Step 2 builds on.

One technique worth understanding: `train_transaction.csv` has 394 columns and is around 650 MB. Loading all of it just to count rows would use several gigabytes of memory for no reason. So we use the `usecols` option to read **only the columns we need**. Reading one column instead of 394 is roughly 400 times cheaper. This habit matters a lot on large data.

```python
"""
Verify the downloaded IEEE-CIS data and write an inventory report.

Checks performed:
  1. Every expected file exists and is not empty
  2. Row and column counts match published figures
  3. The fraud rate is close to the expected 3.5%
  4. TransactionID is unique in both training tables
  5. The known id_ versus id- column naming difference is detected

Usage:
  python scripts/verify_data.py
"""

import sys
from pathlib import Path

import pandas as pd

# Make the project root importable, same reason as in download_data.py
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config.config import (  # noqa: E402
    EXPECTED_RAW_FILES,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    ID_COLUMN,
    TRAIN_TRANSACTION_FILE,
    TRAIN_IDENTITY_FILE,
    TEST_TRANSACTION_FILE,
    TEST_IDENTITY_FILE,
    ensure_directories,
)

# Published figures for this dataset. We compare against these rather than
# trusting the download blindly.
EXPECTED_SHAPES = {
    "train_transaction.csv": (590_540, 394),
    "train_identity.csv": (144_233, 41),
    "test_transaction.csv": (506_691, 393),
    "test_identity.csv": (141_907, 41),
}

EXPECTED_FRAUD_RATE = 0.035  # about 3.5%
FRAUD_RATE_TOLERANCE = 0.005  # accept anything within half a percentage point


def check_files_exist() -> bool:
    """Confirm every expected file is on disk and larger than zero bytes."""
    print("\n1. FILE PRESENCE")
    print("-" * 60)

    all_ok = True
    for file_path in EXPECTED_RAW_FILES:
        if not file_path.exists():
            print(f"  MISSING  {file_path.name}")
            all_ok = False
            continue

        size_mb = file_path.stat().st_size / 1024 / 1024
        if size_mb == 0:
            print(f"  EMPTY    {file_path.name}")
            all_ok = False
        else:
            print(f"  OK       {file_path.name:<28} {size_mb:>9.1f} MB")

    return all_ok


def count_rows_cheaply(file_path: Path) -> int:
    """
    Count rows without loading the whole file.

    We read a single column. pandas still scans the file, but it only keeps
    one column in memory instead of hundreds.
    """
    single_column = pd.read_csv(file_path, usecols=[ID_COLUMN])
    return len(single_column)


def count_columns_cheaply(file_path: Path) -> int:
    """Read only the first row to discover how many columns there are."""
    header_only = pd.read_csv(file_path, nrows=1)
    return header_only.shape[1]


def check_shapes() -> bool:
    """Compare actual row and column counts against published figures."""
    print("\n2. TABLE SHAPES")
    print("-" * 60)
    print(f"  {'File':<28} {'Rows':>10} {'Cols':>6}  {'Expected':>18}")

    all_ok = True
    for file_name, (expected_rows, expected_cols) in EXPECTED_SHAPES.items():
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            continue

        actual_rows = count_rows_cheaply(file_path)
        actual_cols = count_columns_cheaply(file_path)
        matches = (actual_rows == expected_rows) and (actual_cols == expected_cols)
        status = "match" if matches else "DIFFERENT"

        print(
            f"  {file_name:<28} {actual_rows:>10,} {actual_cols:>6}  "
            f"{expected_rows:>10,} x {expected_cols:<3} {status}"
        )

        if not matches:
            all_ok = False

    return all_ok


def check_fraud_rate() -> bool:
    """Load only the label column and measure how rare fraud is."""
    print("\n3. CLASS BALANCE")
    print("-" * 60)

    labels = pd.read_csv(TRAIN_TRANSACTION_FILE, usecols=[TARGET_COLUMN])

    total = len(labels)
    fraud_count = int(labels[TARGET_COLUMN].sum())
    legit_count = total - fraud_count
    fraud_rate = fraud_count / total

    print(f"  Total transactions : {total:,}")
    print(f"  Fraudulent         : {fraud_count:,}")
    print(f"  Legitimate         : {legit_count:,}")
    print(f"  Fraud rate         : {fraud_rate:.4%}")
    print(f"  Imbalance ratio    : 1 fraud per {legit_count / fraud_count:.0f} legitimate")

    within_tolerance = abs(fraud_rate - EXPECTED_FRAUD_RATE) < FRAUD_RATE_TOLERANCE
    print(f"  Expected about {EXPECTED_FRAUD_RATE:.1%}: "
          f"{'as expected' if within_tolerance else 'UNEXPECTED, investigate'}")

    return within_tolerance


def check_id_uniqueness() -> bool:
    """Confirm TransactionID identifies exactly one row in each table."""
    print("\n4. KEY INTEGRITY")
    print("-" * 60)

    all_ok = True
    for file_path in [TRAIN_TRANSACTION_FILE, TRAIN_IDENTITY_FILE]:
        ids = pd.read_csv(file_path, usecols=[ID_COLUMN])[ID_COLUMN]
        duplicate_count = int(ids.duplicated().sum())
        unique_count = ids.nunique()

        print(f"  {file_path.name:<28} unique={unique_count:>9,}  duplicates={duplicate_count}")
        if duplicate_count > 0:
            all_ok = False

    # How many transactions actually have an identity record. This drives
    # a design decision in Step 2, so it is worth knowing now.
    transaction_ids = set(pd.read_csv(TRAIN_TRANSACTION_FILE, usecols=[ID_COLUMN])[ID_COLUMN])
    identity_ids = set(pd.read_csv(TRAIN_IDENTITY_FILE, usecols=[ID_COLUMN])[ID_COLUMN])

    overlap = len(transaction_ids & identity_ids)  # & is set intersection
    coverage = overlap / len(transaction_ids)

    print(f"\n  Transactions with an identity record: {overlap:,} "
          f"({coverage:.1%} of all transactions)")
    print("  The remaining transactions will have missing identity columns "
          "after the join. That is expected.")

    return all_ok


def check_column_naming() -> None:
    """Detect the known id_ versus id- naming difference between train and test."""
    print("\n5. COLUMN NAMING CHECK")
    print("-" * 60)

    train_columns = pd.read_csv(TRAIN_IDENTITY_FILE, nrows=1).columns.tolist()
    test_columns = pd.read_csv(TEST_IDENTITY_FILE, nrows=1).columns.tolist()

    train_underscore = [c for c in train_columns if c.startswith("id_")]
    test_hyphen = [c for c in test_columns if c.startswith("id-")]

    print(f"  train_identity columns starting 'id_' : {len(train_underscore)}")
    print(f"  test_identity  columns starting 'id-' : {len(test_hyphen)}")

    if test_hyphen:
        print("\n  NOTE: the test identity file uses hyphens where the training")
        print("  file uses underscores. This is a known quirk of the released")
        print("  files, not a download problem. Step 2 renames them to match.")


def write_inventory_report() -> None:
    """Save a small markdown summary that Step 2 can build on."""
    ensure_directories()
    report_path = REPORTS_DIR / "data_inventory.md"

    lines = ["# Raw data inventory", ""]
    lines.append("| File | Size (MB) | Rows | Columns |")
    lines.append("|------|-----------|------|---------|")

    for file_name in EXPECTED_SHAPES:
        file_path = RAW_DATA_DIR / file_name
        if not file_path.exists():
            continue
        size_mb = file_path.stat().st_size / 1024 / 1024
        rows = count_rows_cheaply(file_path)
        cols = count_columns_cheaply(file_path)
        lines.append(f"| `{file_name}` | {size_mb:.1f} | {rows:,} | {cols} |")

    labels = pd.read_csv(TRAIN_TRANSACTION_FILE, usecols=[TARGET_COLUMN])
    fraud_rate = labels[TARGET_COLUMN].mean()
    lines.append("")
    lines.append(f"Fraud rate in training data: **{fraud_rate:.4%}**")
    lines.append("")
    lines.append("Generated by `scripts/verify_data.py`.")

    # encoding="utf-8" avoids Windows writing the file in a legacy encoding
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nInventory report written to: {report_path}")


def main() -> None:
    print("=" * 60)
    print("IEEE-CIS FRAUD DETECTION: DATA VERIFICATION")
    print("=" * 60)

    if not check_files_exist():
        print("\nFAILED: files are missing. Run scripts/download_data.py first.")
        sys.exit(1)

    shapes_ok = check_shapes()
    balance_ok = check_fraud_rate()
    keys_ok = check_id_uniqueness()
    check_column_naming()
    write_inventory_report()

    print("\n" + "=" * 60)
    if shapes_ok and balance_ok and keys_ok:
        print("VERIFICATION PASSED. The dataset is ready for Step 2.")
    else:
        print("VERIFICATION FINISHED WITH WARNINGS. Review the sections above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### 8.5 Run the verification

This takes 1 to 3 minutes because it reads through the large files, even though it only keeps one column at a time.

```powershell
python scripts/verify_data.py
```

**Actual verified output from this build:**

```
============================================================
IEEE-CIS FRAUD DETECTION: DATA VERIFICATION
============================================================

1. FILE PRESENCE
------------------------------------------------------------
  OK       train_transaction.csv            651.7 MB
  OK       train_identity.csv                25.3 MB
  OK       test_transaction.csv             584.8 MB
  OK       test_identity.csv                 24.6 MB
  OK       sample_submission.csv              5.8 MB

2. TABLE SHAPES
------------------------------------------------------------
  File                               Rows   Cols            Expected
  train_transaction.csv           590,540    394     590,540 x 394 match
  train_identity.csv              144,233     41     144,233 x 41  match
  test_transaction.csv            506,691    393     506,691 x 393 match
  test_identity.csv               141,907     41     141,907 x 41  match

3. CLASS BALANCE
------------------------------------------------------------
  Total transactions : 590,540
  Fraudulent         : 20,663
  Legitimate         : 569,877
  Fraud rate         : 3.4990%
  Imbalance ratio    : 1 fraud per 28 legitimate
  Expected about 3.5%: as expected

4. KEY INTEGRITY
------------------------------------------------------------
  train_transaction.csv        unique=  590,540  duplicates=0
  train_identity.csv           unique=  144,233  duplicates=0

  Transactions with an identity record: 144,233 (24.4% of all transactions)
  The remaining transactions will have missing identity columns after the join.
  That is expected.

5. COLUMN NAMING CHECK
------------------------------------------------------------
  train_identity columns starting 'id_' : 38
  test_identity  columns starting 'id-' : 38

  NOTE: the test identity file uses hyphens where the training
  file uses underscores. This is a known quirk of the released
  files, not a download problem. Step 2 renames them to match.

Inventory report written to: <project_root>\reports\data_inventory.md

============================================================
VERIFICATION PASSED. The dataset is ready for Step 2.
============================================================
```

These figures are now the reference values for the whole project. Step 2 checks against them again after the join, and any drift means something went wrong.

### 8.6 Confirm the data is invisible to Git

This is the safety check that matters most. `git status` should show only `reports/data_inventory.md` and your new script files, never a CSV.

```powershell
git status
```

If you do see CSV files listed, `.gitignore` is not being applied. Most likely cause: the file is named `gitignore.txt` instead of `.gitignore`, because Windows sometimes appends an extension. Check with:

```powershell
# List files starting with a dot in the current folder
Get-ChildItem -Force -Filter ".*" | Select-Object Name
```

> **Useful side effect:** the last line of the verification output prints the full absolute path where the report was written. That is the quickest way to confirm exactly where the project root resolved to, which is how the folder name mismatch described in D-02 was spotted.

---

## 9. Optional: pin your notebook hygiene now

Notebooks store their output inside the file. Commit a notebook with charts in it and Git records megabytes of base64 image data, which makes diffs unreadable. `nbstripout` (already installed) removes output automatically on commit.

```powershell
# Configure this repository to strip notebook output before every commit
nbstripout --install
```

This writes a small setting into `.git/config` and a filter rule. It runs once and then you forget about it.

---

## 10. Commit, push, and merge Step 1

### 10.1 Commit the work

We use several small commits rather than one large one, because that makes the history readable. The `type: message` prefix is the Conventional Commits style, which is widely used and reads well.

```powershell
# Stage and commit the environment and configuration work
git add requirements.txt requirements-dev.txt requirements.lock.txt .vscode/settings.json .env.example
git commit -m "build: add python 3.11 environment, dependencies, and editor settings"

# Stage and commit the configuration module
git add config/
git commit -m "feat: add central configuration module with dynamic path resolution"

# Stage and commit the data scripts and the inventory they produced
git add scripts/ reports/data_inventory.md
git commit -m "feat: add kaggle download and data verification scripts"

# Stage and commit the documentation
git add docs/
git commit -m "docs: add step 1 guide and project state"

# See the history you have built
git log --oneline
```

### 10.2 Save the step documents into the repo

Copy `step1.md` into `docs/steps/step1.md` and `PROJECT_STATE.md` into `docs/PROJECT_STATE.md`. Keeping them in the repository means the documentation travels with the code, and it gives your PM walkthrough a single place to read from.

### 10.3 Push the branch

```powershell
# Push the step branch and set it to track the remote copy
git push -u origin step-01-foundations
```

### 10.4 Open and merge the pull request

Using the GitHub CLI:

```powershell
# Create a pull request from this branch into main
gh pr create --base main --head step-01-foundations `
  --title "Step 1: foundations" `
  --body "Project scaffold, Kaggle data acquisition, GitHub setup, Python 3.11 environment."

# Merge it, squashing the commits into one tidy entry on main, then delete the branch
gh pr merge --squash --delete-branch
```

Using the website: GitHub shows a yellow banner offering to open a pull request from the branch you just pushed. Click it, add a title, click **Create pull request**, then **Squash and merge**, then **Delete branch**.

### 10.5 Return to main and tag the milestone

A tag is a permanent bookmark on a specific commit. Tagging each step means you can always return to exactly how the project looked when a step finished.

```powershell
# Move back to main and pull down the merged result
git switch main
git pull

# Create an annotated tag and push it
git tag -a v0.1.0-step1 -m "Step 1 complete: foundations"
git push origin v0.1.0-step1
```

---

## 11. Verification checklist, as completed

**Kaggle**
- [x] Kaggle account exists and is logged in
- [x] Competition rules accepted at the IEEE-CIS rules page
- [x] The competition data page shows files rather than a rules prompt
- [x] Credentials configured
- [x] `kaggle.json` is outside the project folder

**Scaffold**
- [x] Project folder created. Note: inside OneDrive, and named `IEEEE_CIS_fraud_project`. Addressed in Step 2 Section 2
- [x] Folder structure matches Section 4.1
- [x] `__init__.py` files exist in `config`, `src`, and each `src` subfolder
- [x] `.gitkeep` files exist in every otherwise-empty folder

**Git and GitHub**
- [x] `.gitignore` exists with a leading dot and correct contents
- [x] `README.md` and `LICENSE` exist
- [x] `git log` shows commits
- [x] Public repository at `https://github.com/Dee-ui/ieee-cis-fraud-detection`
- [x] `git status` shows no CSV files
- [x] Branch `step-01-foundations` created, pushed, merged into `main`
- [x] Tag `v0.1.0-step1` pushed

**Environment**
- [x] `.venv` folder exists
- [x] Terminal prompt shows `(.venv)` when activated
- [x] Python 3.11.x
- [x] Import smoke test succeeds
- [x] `requirements.lock.txt` exists, versions recorded in Section 6.7
- [x] VS Code interpreter set to `.venv`

**Data**
- [x] Five CSV files present in `data/raw/`, totalling 1,292.2 MB
- [x] `python scripts/verify_data.py` prints VERIFICATION PASSED
- [x] `reports/data_inventory.md` exists
- [x] Fraud rate 3.4990%, 20,663 of 590,540
- [x] Identity coverage 24.4%, 144,233 transactions
- [x] Zero duplicate `TransactionID` values in either training table

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `403 Forbidden` on download | Competition rules not accepted | Section 3.2. Regenerating your token will not help. |
| `kaggle: command not found` | Environment not activated, or `kaggle` not installed | Run `.\.venv\Scripts\Activate.ps1`, then `pip install kaggle` |
| `running scripts is disabled on this system` | Windows execution policy | `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process` |
| `ModuleNotFoundError: No module named 'config'` | Running from the wrong folder | `Set-Location` back to the project root, the folder containing `README.md` |
| `MemoryError` when reading CSVs | Loading all 394 columns at once | Use `usecols` to read only what you need, as `verify_data.py` does |
| Git wants to commit CSV files | `.gitignore` misnamed or missing | Confirm the file is exactly `.gitignore`, then run `git rm -r --cached data` and commit again |
| `pip install` fails compiling a package | No prebuilt Windows package for your Python version | Confirm you are on Python 3.11, not a newer release |
| Odd file locking or permission errors | Project sits inside a OneDrive-synced folder | Section 4.3, and Step 2 Section 2 for the full fix |
| VS Code cannot resolve `from src...` imports | Interpreter not set, or extra paths missing | `Ctrl+Shift+P`, `Python: Select Interpreter`, choose `.venv`. Restart the terminal. |
| A terminal command fails on a path with spaces | PowerShell reads spaces as argument separators | Wrap the whole path in double quotes |
| `.venv` stops working after moving the project | A virtual environment contains absolute paths and cannot be moved | Delete `.venv` and rebuild with `py -3.11 -m venv .venv`, then `pip install -r requirements.lock.txt` |
| Verification report written somewhere unexpected | The project root resolved differently from what you assumed | Read the path printed on the last line. That is where `config.py` calculated the root to be. |

---

## 13. Information supplied for Step 2

All three items were provided and are now recorded:

1. **Verification output.** Section 8.5, and carried into `PROJECT_STATE.md` Section 8.
2. **Library versions.** Section 6.7, and `PROJECT_STATE.md` Section 7.1.
3. **Repository URL.** `https://github.com/Dee-ui/ieee-cis-fraud-detection`

Additionally supplied: 32 GB RAM, Intel Core Ultra 7 265H. That is enough to hold the entire joined table in memory, so no chunked reading is needed anywhere in the project.

Still outstanding, tracked as Q-10 in `PROJECT_STATE.md`: the exact Python patch version from `python --version`.

---

## 14. What Step 2 covers

- Joining `train_transaction` and `train_identity`, and what to do about the 75.6% of transactions with no identity record
- Grouping the 435 joined columns into meaningful families: `C` counting features, `D` timedelta features, `M` match flags, `V` engineered Vesta features, plus the identity block
- Decoding `TransactionDT`, which is seconds from an unknown starting point rather than a real timestamp, and what that means for splitting the data correctly
- Profiling the imbalance properly, and why accuracy is the wrong metric here while PR-AUC is the right one
- Missing value patterns, including the V column blocks that share an identical missing pattern
- Memory reduction, taking the joined table from roughly 2 GB down to roughly 600 MB, with a careful account of the two columns where shrinking would corrupt the data
- Saving the joined tables as Parquet in `data/interim/`
- Producing ten charts into `reports/figures/` and an auto-generated summary report

---

*End of Step 1 v1.1. `PROJECT_STATE.md` follows as a separate document.*
