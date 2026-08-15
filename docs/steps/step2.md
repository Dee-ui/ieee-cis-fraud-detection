# Step 2: EDA and Data Understanding
### Table joins, feature groups, missing value patterns, imbalance profiling

**Project:** IEEE-CIS Fraud Detection
**Repository:** https://github.com/Dee-ui/ieee-cis-fraud-detection
**Platform:** Windows, VS Code, PowerShell, Python 3.11
**Estimated time:** 2 to 3 hours, of which about 20 minutes is the machine running
**Step 2 of 7**

---

## 0. How to use this document

Same rules as Step 1. Work top to bottom, do not skip.

Every section tells you **why**, then **what to type or create**, then **how to check it worked**.

Code blocks labelled `powershell` go in the VS Code terminal. Blocks labelled `python` are file contents you create and paste.

Section 2 asks you to make one decision before any code runs. Do that first.

There is a checklist in Section 19. Do not start Step 3 until every box ticks.

---

## 1. Where Step 1 left you

Everything below is now confirmed fact, not assumption, because your verification run produced it.

**Data on disk, all five files verified:**

| File | Size | Rows | Columns |
|------|------|------|---------|
| `train_transaction.csv` | 651.7 MB | 590,540 | 394 |
| `train_identity.csv` | 25.3 MB | 144,233 | 41 |
| `test_transaction.csv` | 584.8 MB | 506,691 | 393 |
| `test_identity.csv` | 24.6 MB | 141,907 | 41 |
| `sample_submission.csv` | 5.8 MB | 506,691 | 2 |

**Class balance, confirmed:**

- 590,540 transactions
- 20,663 fraudulent
- 569,877 legitimate
- Fraud rate 3.4990%
- Roughly 1 fraud per 28 legitimate transactions

**Key integrity, confirmed:**

- `TransactionID` is unique in both training tables, zero duplicates
- 144,233 transactions have an identity record, which is 24.4% of all transactions
- `train_identity` has 38 columns starting `id_`, `test_identity` has 38 starting `id-`, so the naming quirk is real and needs handling

**Machine:** Intel Core Ultra 7 265H, 32 GB RAM. That is comfortably enough to hold the whole joined table in memory at once, so nothing in this step needs chunked reading.

**Library versions now recorded:**

| Library | Version | Why it matters here |
|---------|---------|---------------------|
| pandas | 2.3.3 | `observed=True` must be passed explicitly on grouped operations over category columns, otherwise pandas emits a warning about a future default change |
| numpy | 2.4.6 | This is numpy 2.x, where `np.NaN` and `np.float_` no longer exist. All code below uses `np.nan` and explicit type names |
| pyarrow | 24.0.0 | Powers Parquet reading and writing, and preserves category columns across a save and load |
| scikit-learn | 1.9.0 | Used from Step 3 onward |
| lightgbm | 4.7.0 | Handles missing values natively, which shapes a decision in Section 5.4 |
| xgboost | 3.2.0 | Same |
| catboost | 1.2.10 | Same |
| mlflow | 3.15.1 | Step 4. Note this is MLflow 3, whose API differs from MLflow 2 in places |
| shap | 0.51.0 | Step 4 |
| matplotlib | 3.11.1 | `plt.cm.get_cmap` was removed in 3.9, so none of the plotting code below uses it |
| seaborn | 0.13.2 | Charts |

**Repository:** `https://github.com/Dee-ui/ieee-cis-fraud-detection`, Step 1 merged and tagged.

---

## 2. Housekeeping: one decision to make before you run anything

Your project currently lives here:

```
C:\Users\Dauda Agbonoga\OneDrive - Venture Garden Group\Documents\my\IEEEE_CIS_fraud_project
```

Two things about that path are worth fixing now rather than later. Neither is an emergency, but both get more annoying with every step.

### 2.1 The project is inside a synced OneDrive folder

OneDrive watches that folder and tries to upload everything in it to your company's cloud storage.

Right now that means 1.3 GB of CSV files. After this step it also means roughly 400 MB of Parquet files. After Step 4 it means MLflow run folders, which are thousands of small files that OneDrive is particularly slow at, plus trained model binaries.

The concrete problems this causes:

1. **Sync churn.** Every pipeline run rewrites large files and OneDrive re-uploads them.
2. **File locking.** OneDrive can briefly hold a file open while uploading it. If your script tries to write to that file at the same moment, Python raises a permission error. This shows up as a confusing intermittent failure that works fine on the next run.
3. **Files On-Demand placeholders.** OneDrive can replace a local file with a small pointer to free up disk space. Python then tries to read it, OneDrive downloads it on the spot, and reads become slow or fail.
4. **Company storage quota.** You may be uploading gigabytes of public Kaggle data into a corporate tenant, which is not what it is for.

### 2.2 The folder name and the repository name do not match

The folder is `IEEEE_CIS_fraud_project`, with four E's, and the repository is `ieee-cis-fraud-detection`. The four E's look like a typo. Nothing breaks because of this, since `config/config.py` works out the project root from its own location rather than from a hardcoded string. That is exactly the situation decision D-07 was designed for, and it is now proven. But a mismatched folder name is a small papercut every time you navigate, and it will look untidy if you ever screen share during the PM walkthrough.

### 2.3 Option A, recommended: move the project

This fixes both issues in one go and takes about five minutes.

One important detail: **a virtual environment cannot be moved.** When you create a `.venv`, Python writes the absolute path of that folder into several files inside it. Move the folder and those paths point at nothing. So we delete `.venv` and rebuild it from `requirements.lock.txt`, which reinstalls the exact same versions you already have.

```powershell
# 1. Close VS Code completely first, so no files are locked.

# 2. Create the destination parent folder
New-Item -ItemType Directory -Force -Path "C:\projects" | Out-Null

# 3. Move the project, renaming it to match the repository
Move-Item `
  -Path "$env:USERPROFILE\OneDrive - Venture Garden Group\Documents\my\IEEEE_CIS_fraud_project" `
  -Destination "C:\projects\ieee-cis-fraud-detection"

# 4. Go into the moved project
Set-Location "C:\projects\ieee-cis-fraud-detection"
```

Now rebuild the environment. The backtick at the end of a line in PowerShell means "this command continues on the next line".

```powershell
# Delete the old environment, which no longer works after the move
Remove-Item -Recurse -Force .venv

# Create a fresh one
py -3.11 -m venv .venv

# Turn it on
.\.venv\Scripts\Activate.ps1

# Reinstall the exact same versions you already had
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
```

Check everything survived the move:

```powershell
# Should print the new path
python -c "from config.config import PROJECT_ROOT; print(PROJECT_ROOT)"

# Should still list your Step 1 commits
git log --oneline

# Should still point at GitHub
git remote -v

# Should still find all five CSV files
python scripts/verify_data.py
```

Then reopen in VS Code and re-select the interpreter (`Ctrl+Shift+P`, `Python: Select Interpreter`, choose `.venv`).

### 2.4 Option B: stay where you are, but stop OneDrive syncing the heavy folders

If you would rather not move, do this instead.

1. Right-click the OneDrive cloud icon in the system tray
2. Settings, then the **Account** tab, then **Choose folders**
3. Untick `Documents\my\IEEEE_CIS_fraud_project\data`
4. Repeat later in the project for `models`, `mlruns`, and `.venv` once those exist

Also right-click the project folder in File Explorer and choose **Always keep on this device**, which stops OneDrive turning your files into placeholders.

One more thing to be aware of with this option: your path contains spaces, in `Dauda Agbonoga` and in `OneDrive - Venture Garden Group`. Python handles that fine because `config.py` uses `Path` objects. But any terminal command that mentions the path must wrap it in double quotes, or PowerShell will read the spaces as separators between arguments.

### 2.5 Tell me which you chose

Everything in this document works either way. The commands below never mention an absolute path, precisely so they do not care. Just let me know in your Step 3 message which option you took, so `PROJECT_STATE.md` records the right one.

### 2.6 Start the Step 2 branch

Whichever option you picked, do this before writing any code.

```powershell
# Make sure you are on main and up to date
git switch main
git pull

# Create and switch to the Step 2 branch
git switch -c step-02-eda

# Confirm which branch you are on
git branch
```

---

## 3. Decisions made in this step

These get added to the decision log in `PROJECT_STATE.md`.

| ID | Decision | Why |
|----|----------|-----|
| D-14 | `run.py` is created now, in Step 2, rather than in Step 3 as originally planned | Step 2 produces two runnable pipeline stages, ingestion and EDA. Two stages is the point at which a single entry point stops being overhead and starts being useful. |
| D-15 | The test set is joined and saved alongside the training set, even though it has no labels | It becomes genuinely useful later. In Step 5, drift monitoring needs real future data whose distribution has actually shifted, and the Kaggle test set starts about 30 days after training data ends. That is a far better drift demonstration than randomly perturbing the training data. |
| D-16 | Left join transaction to identity, keep the missing values as missing, and add a `has_identity` flag column | The three real options were: drop identity columns entirely (throws away 41 columns of signal); split into two models (doubles the work and halves the data each model sees); or left join and let the model handle the gaps. LightGBM, XGBoost, and CatBoost all handle missing values natively, so the third option costs nothing. The `has_identity` flag makes the missingness itself available as a feature, because whether a transaction has an identity record turns out to be informative. |
| D-17 | Interim data is stored as Parquet, with category dtypes preserved | Parquet is roughly a third of the size of the equivalent CSV, loads about ten times faster, and remembers data types. CSV forgets every type, so you would redo the type work on every single load. |
| D-18 | `TransactionAmt` stays `float64`. `TransactionID` and `TransactionDT` become `int32`. Everything else is shrunk to the smallest type that holds it exactly | Explained fully in Section 7. The short version: float32 rounds `TransactionAmt` at about the third decimal place, and the cents portion of the amount is a known fraud signal in this dataset. float32 also cannot store the test set's largest `TransactionDT` value exactly. |
| D-19 | `TransactionDT` is displayed against a reference date of 30 November 2017 for readability only. No modelling decision depends on that date being correct | The competition never published a real start date. The community derived this one, and it is used widely. It makes charts readable. Since we only ever use `TransactionDT` as an ordering and as a time delta, being wrong about the calendar date would change nothing that matters. |
| D-20 | PR-AUC is the primary metric. ROC-AUC is reported as a secondary metric. Accuracy is not used at all | Reasoning in Section 16. |
| D-21 | Validation is a time-based split, never a random split | Reasoning in Section 17. |
| D-22 | Feature family assignment is computed by rule and any unmapped column is reported loudly | With 435 columns, a silently miscategorised column is easy to miss. The code counts every column into exactly one family and prints a warning if anything falls through, so nothing goes missing without you seeing it. |

---

## 4. What Step 2 produces

**New code files:**

| File | Purpose |
|------|---------|
| `config/config.py` | Replaced with an extended version: feature family definitions, EDA output paths, split settings |
| `src/utils/memory_utils.py` | Shrinks a DataFrame to the smallest safe data types |
| `src/utils/ingestion_utils.py` | Loading, column name standardising, joining, join validation |
| `src/pipelines/ingestion.py` | Orchestrates load, standardise, join, optimise, save |
| `src/utils/eda_utils.py` | Feature families, column profiling, missing value analysis, all chart functions |
| `src/pipelines/eda.py` | Orchestrates the analysis and writes reports and figures |
| `run.py` | Single entry point for every pipeline stage |

**New data outputs:**

| File | Contents |
|------|----------|
| `data/interim/train_joined.parquet` | 590,540 rows by 435 columns, type-optimised |
| `data/interim/test_joined.parquet` | 506,691 rows by 434 columns, type-optimised |

**New report outputs:**

| File | Contents |
|------|----------|
| `reports/eda_summary.md` | The written findings, auto-generated |
| `reports/column_profile.csv` | One row per column: family, dtype, missing percentage, unique count, range |
| `reports/missing_profile.csv` | Columns ranked by how much data they are missing |
| `reports/v_column_missing_groups.csv` | V columns grouped by identical missing pattern |
| `reports/figures/*.png` | Ten charts |

---

## 5. Understanding the data before writing any code

Read this section before you touch the keyboard. It is the part that makes the code make sense, and it is the part you will be explaining on the PM track.

### 5.1 The two tables and how they relate

You have two training tables that both carry `TransactionID`.

**`train_transaction`** has one row per transaction, 590,540 of them, and carries the label `isFraud`.

**`train_identity`** has one row per transaction too, but only for 144,233 transactions. It holds device and network information: what browser was used, what operating system, screen resolution, and a set of anonymised identity signals.

The relationship is one row to at most one row. Your Step 1 verification proved there are no duplicate IDs on either side, so a join cannot accidentally multiply your rows. We still tell pandas to enforce that, so if the assumption ever breaks the code stops instead of silently producing a bigger table.

### 5.2 What the 394 transaction columns actually are

Vesta Corporation, who supplied the data, anonymised most of it. But the column names still tell you which family each one belongs to, and the families are documented by the competition hosts and by Vesta's own responses in the competition discussion.

| Family | Columns | Count | What it holds |
|--------|---------|-------|---------------|
| Identifier | `TransactionID` | 1 | Row identifier, not a feature |
| Target | `isFraud` | 1 | 1 means fraud, 0 means legitimate |
| Time | `TransactionDT` | 1 | Seconds elapsed from an unknown reference moment |
| Amount | `TransactionAmt` | 1 | Transaction value in US dollars |
| Product | `ProductCD` | 1 | Product code: W, C, R, H, or S |
| Card | `card1` to `card6` | 6 | Payment card attributes. `card4` is the network such as visa or mastercard. `card6` is the type such as credit or debit. `card1`, `card2`, `card3`, `card5` are numeric codes whose meaning is masked |
| Address | `addr1`, `addr2` | 2 | Billing region codes. `addr2` behaves like a country code |
| Distance | `dist1`, `dist2` | 2 | Distances, units unspecified, probably between billing and shipping or IP location |
| Email | `P_emaildomain`, `R_emaildomain` | 2 | Email domain of the purchaser and of the recipient |
| Counting | `C1` to `C14` | 14 | Counts of things linked to the card, such as how many addresses or phone numbers are associated with it. Exact meanings masked |
| Timedelta | `D1` to `D15` | 15 | Gaps in days: how long since a previous transaction, how long since the card was first seen, and similar |
| Match | `M1` to `M9` | 9 | Whether two pieces of information agree, for example whether the billing name matches the card name. Values are T or F, except `M4` which is M0, M1, or M2 |
| Vesta | `V1` to `V339` | 339 | Features Vesta engineered themselves: ranking, counting, and entity relationship signals. Completely anonymised |

Those add to 394 exactly. The code verifies that arithmetic rather than trusting it.

### 5.3 What the 41 identity columns are

| Family | Columns | Count | What it holds |
|--------|---------|-------|---------------|
| Identifier | `TransactionID` | 1 | Join key |
| Identity signals | `id_01` to `id_38` | 38 | A mix of numeric and text. Some are network and device signals, some are behavioural. `id_30` is the operating system, `id_31` is the browser, `id_33` is the screen resolution |
| Device | `DeviceType`, `DeviceInfo` | 2 | `DeviceType` is desktop or mobile. `DeviceInfo` is a free text device string |

That is 41. After the join we drop the duplicate `TransactionID` and add our own `has_identity` flag, so:

394 transaction columns + 40 identity columns + 1 flag = **435 columns** in the joined training table.

For test, there is no `isFraud`, so: 393 + 40 + 1 = **434 columns**.

### 5.4 The 76% missing identity problem, and why it is not really a problem

Only 24.4% of transactions have an identity record. After a left join, the other 75.6% have blanks across all 40 identity columns.

The instinct is to see that as damage. It is not, for two reasons.

**First, the missingness is informative.** Whether a transaction produced an identity record is not random. It depends on how the payment was made and through what channel. If fraud is more common in the channels that produce identity records, then "has an identity record" is itself a useful signal. The analysis in this step measures exactly that, by comparing the fraud rate among transactions with an identity record against those without.

**Second, the models we are using handle missing values natively.** This is worth understanding properly, because it is the reason the decision is easy.

A gradient boosted tree splits data by asking yes or no questions, such as "is `card2` greater than 300". When a value is missing, LightGBM does not guess it or throw the row away. During training it tries sending all the missing-valued rows down the left branch, then tries sending them all down the right branch, and keeps whichever produced the better split. The missing values get their own learned direction at every single split. So a blank is treated as a third category alongside yes and no, rather than as damaged data.

This is why filling in missing values, which is what NovaPay did with its imputation step, is often unnecessary and sometimes actively harmful with tree models. Filling a blank with the column average tells the model something false, that this transaction was average, when the truth is that the information was never captured.

So: left join, keep the blanks, add a flag. That is D-16.

One caution the code checks for. If `has_identity` turns out to be almost perfectly determined by `ProductCD`, then the flag adds nothing new, because the model can already see `ProductCD`. The EDA prints a cross-tabulation so you can see whether that is the case.

### 5.5 TransactionDT, and why it is the most important column in the dataset

`TransactionDT` is not a timestamp. It is a count of seconds from a starting moment that Vesta never disclosed. The smallest value in the training data is 86,400, which is exactly one day in seconds.

That vagueness does not matter. What matters is that it **orders the data correctly**, and that ordering drives three separate things.

**First, the training data covers about 183 days.** You can confirm this yourself: the training range spans roughly 15.8 million seconds, and 15,811,131 divided by 86,400 is about 183 days.

**Second, the test set comes strictly after the training set, with a gap.** The test data starts at about 18.4 million seconds. Training ends at about 15.8 million. The difference is roughly 2.59 million seconds, which is about 30 days. So there is a full month of unseen time between the last training transaction and the first test transaction, then the test period runs for another 183 days or so.

This is a deliberate design by the competition hosts, and it mirrors how fraud detection works in real life. You train on the past and score the future. You never get to train on the day you are scoring.

**Third, and this is the consequence that governs everything from Step 3 onward: your validation split must respect time.**

If you shuffle all 590,540 rows and take a random 20% as validation, you are training on transactions from June and validating on transactions from January. That is time travel. Your model gets to see the future while learning, and your validation score comes out too good, because your validation set is not actually harder than your training set. Then you deploy, real future data arrives, and performance collapses.

Worse, this dataset is full of columns that make the leak severe. The `D` columns are time deltas measured from card first-seen dates. The `C` columns are running counts. Card identifiers repeat across time. A random split scatters the same card across both sides of the split, so the model effectively memorises rather than generalises.

The fix is simple: sort by `TransactionDT`, train on the earlier portion, validate on the later portion. That is D-21, and Section 17 sets out the exact split.

For charts, we convert `TransactionDT` to a readable date using a reference of 30 November 2017, so that the first transaction falls on 1 December 2017. That reference is a community convention, not an official fact. It exists purely so that the x-axis of a chart says "January 2018" instead of "4,500,000 seconds". No modelling decision depends on it. That is D-19.

### 5.6 The V columns and their block structure

There are 339 V columns, which is 78% of all your columns. They are entirely anonymised.

There is one structural fact about them that is genuinely useful. The V columns fall into **blocks that share exactly the same missing value pattern**. If `V1` is blank on a particular row, then a whole set of other V columns are blank on that same row too, every single time.

Why that happens: Vesta engineered these features in groups, each group computed from a common underlying source. If that source was unavailable for a transaction, every feature derived from it is unavailable together.

Why it is useful: columns within one block are usually highly related to each other, often measuring near identical things. In Step 3 that gives you a principled way to cut 339 columns down to something far smaller, by keeping a representative or two from each block instead of dropping columns arbitrarily or keeping all of them.

We do not do that reduction now. Step 2 identifies the blocks and writes them to a CSV. Step 3 acts on it.

### 5.7 The imbalance, stated plainly

3.4990% fraud means 1 fraud in every 28.6 transactions.

Two consequences to hold on to.

**Accuracy is meaningless here.** A model that predicts "not fraud" for every single transaction, and therefore has learned nothing whatsoever, scores 96.5% accuracy. Any metric that a useless model can score well on is not a metric.

**But this imbalance is workable.** NovaPay had 145 fraud cases in training. You have around 16,500 in a time-based 80% split. That is roughly a hundred times more examples of the thing you are trying to detect. This is the entire reason the dataset was swapped.

Section 16 sets out what to measure instead of accuracy.

---

## 6. Update `config/config.py`

### 6.1 Why it changes

Step 1's config knew about raw files. It now needs to know about feature families, EDA outputs, and split settings. Everything goes here rather than being scattered, so there is exactly one place to look when a path or a constant is wrong.

### 6.2 The file

**Replace the entire contents** of `config/config.py` with this. Nothing from the Step 1 version is removed, only added to.

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
# This is why the project works on any machine, in Docker, and in CI,
# and why the project folder can be moved or renamed without breaking.
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# Reproducibility
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

EXPECTED_RAW_FILES = [
    TRAIN_TRANSACTION_FILE,
    TRAIN_IDENTITY_FILE,
    TEST_TRANSACTION_FILE,
    TEST_IDENTITY_FILE,
    SAMPLE_SUBMISSION_FILE,
]


# ---------------------------------------------------------
# Key column names
# ---------------------------------------------------------

TARGET_COLUMN = "isFraud"
ID_COLUMN = "TransactionID"
JOIN_KEY = "TransactionID"
TIME_COLUMN = "TransactionDT"
AMOUNT_COLUMN = "TransactionAmt"

# Added during ingestion: 1 if the transaction had an identity record.
IDENTITY_FLAG_COLUMN = "has_identity"


# ---------------------------------------------------------
# Feature families.
#
# Built with list comprehensions so the definition stays short and
# cannot contain a typo in the middle of a long hand-written list.
#
# f"C{i}" for i in range(1, 15) produces C1, C2, ... C14.
# range(1, 15) stops BEFORE 15, which is why the upper number is
# always one more than the last column you want.
#
# f"id_{i:02d}" pads the number to two digits, so 1 becomes "01",
# producing id_01 ... id_38 rather than id_1 ... id_38.
# ---------------------------------------------------------

C_COLUMNS = [f"C{i}" for i in range(1, 15)]           # C1 ... C14
D_COLUMNS = [f"D{i}" for i in range(1, 16)]           # D1 ... D15
M_COLUMNS = [f"M{i}" for i in range(1, 10)]           # M1 ... M9
V_COLUMNS = [f"V{i}" for i in range(1, 340)]          # V1 ... V339
IDENTITY_COLUMNS = [f"id_{i:02d}" for i in range(1, 39)]  # id_01 ... id_38

CARD_COLUMNS = [f"card{i}" for i in range(1, 7)]      # card1 ... card6
ADDRESS_COLUMNS = ["addr1", "addr2"]
DISTANCE_COLUMNS = ["dist1", "dist2"]
EMAIL_COLUMNS = ["P_emaildomain", "R_emaildomain"]
DEVICE_COLUMNS = ["DeviceType", "DeviceInfo"]

# Columns that arrive as text rather than numbers. Note this is the
# KNOWN list. The profiling code detects text columns at runtime rather
# than trusting this, and reports any disagreement.
KNOWN_TEXT_COLUMNS = (
    ["ProductCD", "card4", "card6"]
    + EMAIL_COLUMNS
    + M_COLUMNS
    + DEVICE_COLUMNS
)


# ---------------------------------------------------------
# Time interpretation
#
# TransactionDT is seconds from an undisclosed starting moment. This
# reference date is a community convention that makes charts readable
# by placing the first transaction on 1 December 2017.
#
# Nothing in the modelling depends on this being the true date. We only
# ever use TransactionDT as an ordering and as an elapsed duration.
# ---------------------------------------------------------

REFERENCE_DATETIME = "2017-11-30"

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


# ---------------------------------------------------------
# Pipeline stage outputs
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
# EDA outputs
# ---------------------------------------------------------

EDA_SUMMARY_FILE = REPORTS_DIR / "eda_summary.md"
COLUMN_PROFILE_FILE = REPORTS_DIR / "column_profile.csv"
MISSING_PROFILE_FILE = REPORTS_DIR / "missing_profile.csv"
V_GROUPS_FILE = REPORTS_DIR / "v_column_missing_groups.csv"
DATA_INVENTORY_FILE = REPORTS_DIR / "data_inventory.md"


# ---------------------------------------------------------
# MLflow experiment tracking (configured properly in Step 4)
# ---------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
)
MLFLOW_EXPERIMENT_NAME = "ieee-cis-fraud-detection"


# ---------------------------------------------------------
# Splitting and modelling defaults
#
# VALIDATION_FRACTION is the share of TRAINING data held back for
# validation. It is taken from the END of the time range, never at
# random, because the real test set is strictly in the future.
# ---------------------------------------------------------

VALIDATION_FRACTION = 0.2
CV_FOLDS = 5
FRAUD_PROBABILITY_THRESHOLD = 0.5    # placeholder, tuned properly in Step 4


# ---------------------------------------------------------
# Analysis thresholds
# ---------------------------------------------------------

# A category needs at least this many transactions before we report a
# fraud rate for it. Without a floor, a category with 3 transactions and
# 1 fraud shows a 33% fraud rate and looks alarming for no reason.
MIN_CATEGORY_COUNT = 500

# Columns missing more than this share of their values get flagged for
# review in Step 3.
HIGH_MISSING_THRESHOLD = 0.90


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

### 6.3 Check it

```powershell
python -c "from config.config import V_COLUMNS, IDENTITY_COLUMNS, C_COLUMNS; print(len(V_COLUMNS), len(IDENTITY_COLUMNS), len(C_COLUMNS)); print(V_COLUMNS[:3], V_COLUMNS[-1]); print(IDENTITY_COLUMNS[:2], IDENTITY_COLUMNS[-1])"
```

**Expected output:**

```
339 38 14
['V1', 'V2', 'V3'] V339
['id_01', 'id_02'] id_38
```

If the counts are wrong, a `range()` upper bound is off by one.

---

## 7. Create `src/utils/memory_utils.py`

### 7.1 Why memory optimisation matters here

When pandas reads a CSV it has no idea how big the numbers in a column will get, so it plays safe. Every whole number becomes `int64`, which is 8 bytes. Every decimal becomes `float64`, which is also 8 bytes. Every piece of text becomes a Python object, which is far more expensive again because each individual string is a separate object with its own overhead.

A column like `isFraud` only ever holds 0 or 1. It needs 1 byte per row, not 8. A column like `ProductCD` only ever holds one of five short strings. Storing 590,540 separate copies of those strings is enormously wasteful when you could store five strings once and then 590,540 small numbers pointing at them. That second approach is what pandas calls a **category** dtype.

Across 435 columns and 590,540 rows the difference is roughly 2 GB versus roughly 600 MB.

You have 32 GB of RAM, so 2 GB is not going to crash anything. Do it anyway, for three reasons:

1. Every later operation, sorting, grouping, joining, training, gets faster on a smaller table
2. Step 6 runs this inside a Docker container, which usually has far less memory than your laptop
3. It is the kind of engineering care that a reviewer notices

### 7.2 The two places where shrinking would break the data

This is the part worth understanding properly, because getting it wrong corrupts your data silently. Nothing errors, the numbers just quietly become slightly wrong.

**`float32` cannot store large whole numbers exactly.**

A `float32` uses 24 bits for the significant digits. That means it stores whole numbers exactly only up to 2 to the power of 24, which is 16,777,216. Above that it starts rounding to the nearest representable value.

`TransactionDT` in the test set reaches about 34,214,345. Convert that to `float32` and you get 34,214,344. One second lost, silently. Every downstream time calculation is then slightly wrong, and nothing warns you.

The fix is `int32`, which holds whole numbers up to 2,147,483,647 exactly. Comfortable headroom.

**`float32` cannot hold cent-level precision on `TransactionAmt`.**

`float32` gives about 7 significant digits. The largest transaction amount in this dataset is around 31,937.39. That figure already uses 7 digits, so `float32` starts rounding in the third decimal place. Specifically, 31937.39 becomes 31937.390625, an error of about 0.0006.

Normally that would not matter. Here it does, because of a well known quirk of this dataset: **the decimal portion of the transaction amount is itself a fraud signal.** Legitimate purchases cluster on round amounts and on currency-conversion artefacts. Fraudulent ones distribute differently. Feature engineering in Step 3 will extract the cents as a feature, and that only works if the cents are still accurate.

So `TransactionAmt` stays `float64`. That is one column at 8 bytes instead of 4, costing about 2 MB. Cheap insurance.

Both of these are why the code below has an explicit protected list rather than shrinking everything blindly.

### 7.3 The file

Create `src/utils/memory_utils.py`:

```python
"""
Memory optimisation helpers for large tabular data.

Why this file exists
--------------------
pandas reads every whole number as int64 (8 bytes per value) and every
decimal as float64 (8 bytes per value), regardless of whether the column
needs that much room. A column that only ever holds 0 or 1 needs 1 byte,
not 8. A text column with five distinct values can be stored as five
strings plus a small pointer per row, instead of one full string per row.

On 590,540 rows and 435 columns that is roughly 2 GB versus roughly 600 MB.

The functions here shrink each column to the smallest type that still
holds its values exactly, and refuse to shrink where doing so would lose
information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Columns pinned to a specific type instead of being auto-shrunk.
#
# TransactionAmt: float32 keeps about 7 significant digits. The largest
#   amount here is roughly 31,937.39, so float32 starts rounding in the
#   third decimal place. The cents portion of the amount is a known fraud
#   signal in this dataset, so we keep full float64 precision on purpose.
#
# TransactionDT: a whole number of seconds reaching about 34 million in
#   the test set. float32 stores whole numbers exactly only up to
#   16,777,216, so it would silently round. int32 holds it exactly.
#
# TransactionID: int32 comfortably covers the largest ID, around 4.7M.
#
# isFraud: only ever 0 or 1, so int8 is plenty.
# ---------------------------------------------------------

PROTECTED_DTYPES = {
    "TransactionID": "int32",
    "TransactionDT": "int32",
    "isFraud": "int8",
    "TransactionAmt": "float64",
}

# Largest whole number a float32 can hold exactly: 2 ** 24.
FLOAT32_EXACT_INTEGER_LIMIT = 2 ** 24

# A text column becomes a category when its distinct values make up less
# than this share of its rows. Above the threshold, categories stop saving
# memory because there are nearly as many categories as rows.
CATEGORY_UNIQUE_RATIO = 0.5


def memory_usage_mb(frame: pd.DataFrame) -> float:
    """
    Report how much memory a DataFrame occupies, in megabytes.

    deep=True is important. Without it, pandas reports only the size of
    the pointers in a text column, not the size of the strings themselves,
    which massively understates the real usage.
    """
    return float(frame.memory_usage(deep=True).sum()) / 1024 ** 2


def _downcast_integer_series(series: pd.Series) -> pd.Series:
    """Shrink a whole-number column to the smallest integer type that fits."""
    minimum = series.min()
    maximum = series.max()

    # Try each type from smallest to largest and take the first that fits.
    # np.iinfo tells us the lowest and highest value a given integer type
    # can hold.
    for candidate in ("int8", "int16", "int32", "int64"):
        limits = np.iinfo(candidate)
        if minimum >= limits.min and maximum <= limits.max:
            return series.astype(candidate)

    return series


def _downcast_float_series(series: pd.Series) -> pd.Series:
    """
    Shrink a decimal column to float32, but only when that is safe.

    Two cases are handled differently.

    Whole-number columns (common here, because many numeric columns are
    counts that pandas read as float only because they contain blanks):
    float32 is safe only if every value sits below 2 ** 24. Above that,
    float32 rounds whole numbers, so we keep float64.

    Genuine decimal columns: float32 keeps about 7 significant digits,
    which is ample for the rates, distances, and aggregates in this
    dataset. The one column where cent-level precision matters,
    TransactionAmt, is in PROTECTED_DTYPES and never reaches this code.
    """
    values = series.to_numpy(dtype="float64", copy=False)
    present = np.isfinite(values)

    # A column that is entirely blank costs nothing to shrink.
    if not present.any():
        return series.astype("float32")

    observed = values[present]

    # np.equal(a, np.round(a)) is True where a value has no fractional part.
    is_whole_number_column = bool(np.all(observed == np.round(observed)))

    if is_whole_number_column:
        largest = float(np.max(np.abs(observed)))
        if largest < FLOAT32_EXACT_INTEGER_LIMIT:
            return series.astype("float32")
        # Too large for float32 to hold exactly, so leave it alone.
        return series

    return series.astype("float32")


def _convert_text_series(series: pd.Series) -> pd.Series:
    """
    Turn a repetitive text column into a category column.

    A category stores each distinct value once, then stores a small
    integer per row pointing at it. For a column like ProductCD, with five
    distinct values across 590,540 rows, that is a very large saving.

    dropna=False counts blank as one of the distinct values, which is what
    we want when judging repetitiveness.
    """
    row_count = len(series)
    if row_count == 0:
        return series

    distinct_count = series.nunique(dropna=False)
    if distinct_count / row_count < CATEGORY_UNIQUE_RATIO:
        return series.astype("category")

    return series


def optimise_dtypes(
    frame: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Shrink every column of a DataFrame to its smallest safe type.

    Returns the modified frame and a small dictionary of before and after
    figures, so the calling code can report and log the saving.

    The frame is modified in place rather than copied. Copying a 2 GB table
    just to shrink it would need 4 GB at the peak, which defeats the point.
    """
    before_mb = memory_usage_mb(frame)

    # Apply the pinned types first, so the loop below skips them.
    for column, target_dtype in PROTECTED_DTYPES.items():
        if column in frame.columns:
            frame[column] = frame[column].astype(target_dtype)

    for column in frame.columns:
        if column in PROTECTED_DTYPES:
            continue

        series = frame[column]

        # Already a category, nothing to do. This check must come first:
        # a category of strings can also look like a text column to the
        # checks below, and converting it again wastes time.
        if isinstance(series.dtype, pd.CategoricalDtype):
            continue

        if pd.api.types.is_bool_dtype(series):
            continue

        if pd.api.types.is_object_dtype(series):
            frame[column] = _convert_text_series(series)
        elif pd.api.types.is_integer_dtype(series):
            frame[column] = _downcast_integer_series(series)
        elif pd.api.types.is_float_dtype(series):
            frame[column] = _downcast_float_series(series)

    after_mb = memory_usage_mb(frame)
    reduction_pct = (1 - after_mb / before_mb) * 100 if before_mb else 0.0

    summary = {
        "before_mb": round(before_mb, 1),
        "after_mb": round(after_mb, 1),
        "reduction_pct": round(reduction_pct, 1),
    }

    if verbose:
        print(
            f"  Memory: {summary['before_mb']:,.1f} MB -> "
            f"{summary['after_mb']:,.1f} MB "
            f"({summary['reduction_pct']:.1f}% smaller)"
        )

    return frame, summary


def dtype_breakdown(frame: pd.DataFrame) -> pd.Series:
    """Count how many columns hold each data type. Useful as a sanity check."""
    return frame.dtypes.astype(str).value_counts()
```

---

## 8. Create `src/utils/ingestion_utils.py`

### 8.1 Why these functions are separate from the pipeline

`src/pipelines/ingestion.py` will describe **what happens in what order**. This file holds **how each individual thing is done**.

The split matters for three reasons. The pipeline file stays short enough to read in one screen and understand the whole flow. Each helper can be tested on its own in Step 5. And a helper can be reused elsewhere without dragging the whole pipeline along. This is the same `pipelines` and `utils` separation NovaPay used, carried forward because it worked.

### 8.2 The file

Create `src/utils/ingestion_utils.py`:

```python
"""
Helper functions for loading and joining the raw IEEE-CIS tables.

The pipeline module decides what happens in what order. This module holds
the individual operations, so each one can be read, reused, and tested on
its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Matches the test-file identity column names: id-01, id-02 ... id-38.
# The competition released these with hyphens while the training file uses
# underscores. re.fullmatch requires the WHOLE name to match, so this
# cannot accidentally catch some other column that merely contains "id-".
TEST_IDENTITY_PATTERN = re.compile(r"^id-\d{2}$")


def load_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame.

    low_memory=False tells pandas to read the whole column before deciding
    its type. The default reads in chunks and can guess different types for
    different chunks of the same column, which produces a warning and
    occasionally a genuinely wrong type.

    nrows exists so you can test the pipeline on a small slice without
    waiting for the full file. Leave it as None for a real run.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            f"Run  python scripts/download_data.py  first."
        )

    print(f"  Reading {path.name} ...")
    frame = pd.read_csv(path, low_memory=False, nrows=nrows)
    print(f"    {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    return frame


def standardise_identity_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Rename test identity columns from id-NN to id_NN.

    Without this, the joined training table would have a column called
    id_01 and the joined test table would have one called id-01. Any model
    trained on the first would then fail on the second, because it would be
    looking for a column that does not exist under that name.

    Returns the frame and how many columns were renamed, so the caller can
    report it.
    """
    rename_map = {
        column: column.replace("-", "_")
        for column in frame.columns
        if TEST_IDENTITY_PATTERN.match(column)
    }

    if rename_map:
        frame = frame.rename(columns=rename_map)

    return frame, len(rename_map)


def add_identity_marker(identity_frame: pd.DataFrame, flag_column: str) -> pd.DataFrame:
    """
    Add a column of 1s to the identity table before joining.

    After a left join, this column is 1 for transactions that had an
    identity record and blank for those that did not. Filling the blanks
    with 0 then gives a clean flag.

    Doing it this way, rather than checking whether some identity column is
    blank afterwards, is safer: an identity record could legitimately exist
    while every one of its individual fields is blank.
    """
    identity_frame = identity_frame.copy()
    identity_frame[flag_column] = 1
    return identity_frame


def join_transaction_identity(
    transaction_frame: pd.DataFrame,
    identity_frame: pd.DataFrame,
    join_key: str,
    flag_column: str,
) -> pd.DataFrame:
    """
    Left join the transaction table to the identity table.

    "Left join" means: keep every row from the left table (transactions),
    attach matching information from the right table (identity) where it
    exists, and leave blanks where it does not. No transaction is ever
    dropped for lacking an identity record.

    validate="one_to_one" makes pandas check that the join key is unique on
    both sides before joining. Step 1 verification proved that it is. We
    enforce it anyway, because if that assumption ever broke, the join would
    silently produce MORE rows than it started with, and a silently wrong
    row count is far worse than a crash.
    """
    print(f"  Joining on {join_key} ...")

    merged = transaction_frame.merge(
        identity_frame,
        on=join_key,
        how="left",
        validate="one_to_one",
    )

    # Rows with no identity record have a blank flag. Make it a real 0.
    merged[flag_column] = merged[flag_column].fillna(0).astype("int8")

    matched = int(merged[flag_column].sum())
    total = len(merged)
    print(
        f"    {matched:,} of {total:,} transactions matched an identity "
        f"record ({matched / total:.1%})"
    )

    return merged


def validate_join(
    merged: pd.DataFrame,
    expected_rows: int,
    expected_columns: int,
    join_key: str,
) -> dict:
    """
    Confirm the join produced exactly the table we expected.

    Three checks:
      1. Row count unchanged. A left join must never change it.
      2. Column count as predicted.
      3. Join key still unique.

    Returns a dictionary of results rather than raising, so the caller can
    report every problem at once instead of stopping at the first.
    """
    actual_rows = len(merged)
    actual_columns = merged.shape[1]
    duplicate_keys = int(merged[join_key].duplicated().sum())

    results = {
        "rows_match": actual_rows == expected_rows,
        "columns_match": actual_columns == expected_columns,
        "keys_unique": duplicate_keys == 0,
        "actual_rows": actual_rows,
        "actual_columns": actual_columns,
        "expected_rows": expected_rows,
        "expected_columns": expected_columns,
        "duplicate_keys": duplicate_keys,
    }

    results["passed"] = (
        results["rows_match"] and results["columns_match"] and results["keys_unique"]
    )

    return results


def save_parquet(frame: pd.DataFrame, path: Path) -> float:
    """
    Save a DataFrame to Parquet and report the resulting file size in MB.

    Parquet stores data column by column rather than row by row, which
    compresses far better than CSV because values within a column are
    similar to each other. It also records the data type of every column,
    so a later read gets int8 and category columns back exactly as they
    were. CSV forgets all of that and you redo the type work every time.

    index=False leaves out the row numbers, which carry no information here
    because TransactionID is already the identifier.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Writing {path.name} ...")
    frame.to_parquet(path, index=False, engine="pyarrow", compression="snappy")

    size_mb = path.stat().st_size / 1024 ** 2
    print(f"    {size_mb:,.1f} MB on disk")
    return size_mb
```

---

## 9. Create `src/pipelines/ingestion.py`

### 9.1 What this stage does

Six things, in order:

1. Load the transaction table
2. Load the identity table
3. Standardise the identity column names (only does anything for test)
4. Add the identity marker, then left join
5. Validate that the join produced what we expected
6. Shrink the data types, then save as Parquet

It handles both train and test through the same code path, with the differences passed in as arguments. Writing the logic once means train and test cannot drift apart, which is a real risk when they are handled by separate scripts.

### 9.2 The file

Create `src/pipelines/ingestion.py`:

```python
"""
Ingestion stage: load the raw CSV tables, join them, and save as Parquet.

Input:  data/raw/{split}_transaction.csv
        data/raw/{split}_identity.csv
Output: data/interim/{split}_joined.parquet

Run with:
    python run.py --step ingestion
"""

from __future__ import annotations

from pathlib import Path

from config.config import (
    IDENTITY_FLAG_COLUMN,
    JOIN_KEY,
    JOINED_TEST_FILE,
    JOINED_TRAIN_FILE,
    TEST_IDENTITY_FILE,
    TEST_TRANSACTION_FILE,
    TRAIN_IDENTITY_FILE,
    TRAIN_TRANSACTION_FILE,
    ensure_directories,
)
from src.utils.ingestion_utils import (
    add_identity_marker,
    join_transaction_identity,
    load_csv,
    save_parquet,
    standardise_identity_columns,
    validate_join,
)
from src.utils.memory_utils import dtype_breakdown, optimise_dtypes

# What each split should produce. Having the expected figures written down
# turns a silent mistake into a loud one.
#
# Column arithmetic for train:
#   394 transaction columns
# +  40 identity columns (41 minus the shared TransactionID)
# +   1 has_identity flag
# = 435
#
# Test is the same minus isFraud, so 434.
SPLIT_SETTINGS = {
    "train": {
        "transaction_file": TRAIN_TRANSACTION_FILE,
        "identity_file": TRAIN_IDENTITY_FILE,
        "output_file": JOINED_TRAIN_FILE,
        "expected_rows": 590_540,
        "expected_columns": 435,
    },
    "test": {
        "transaction_file": TEST_TRANSACTION_FILE,
        "identity_file": TEST_IDENTITY_FILE,
        "output_file": JOINED_TEST_FILE,
        "expected_rows": 506_691,
        "expected_columns": 434,
    },
}


def ingest_split(split: str, nrows: int | None = None) -> dict:
    """
    Run the full ingestion process for one split, either "train" or "test".

    nrows limits how many rows are read, for quick testing. When it is set,
    the row and column count checks are skipped, because a 1,000 row sample
    obviously will not have 590,540 rows.
    """
    if split not in SPLIT_SETTINGS:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    settings = SPLIT_SETTINGS[split]
    is_sample_run = nrows is not None

    print("\n" + "-" * 60)
    print(f"INGESTING: {split}")
    if is_sample_run:
        print(f"(sample run, first {nrows:,} rows only)")
    print("-" * 60)

    ensure_directories()

    # --- 1 and 2: load both tables -----------------------------------
    transactions = load_csv(settings["transaction_file"], nrows=nrows)
    identities = load_csv(settings["identity_file"], nrows=nrows)

    # --- 3: fix the id- versus id_ naming difference ------------------
    identities, renamed_count = standardise_identity_columns(identities)
    if renamed_count:
        print(f"  Renamed {renamed_count} identity columns from id-NN to id_NN")
    else:
        print("  No identity columns needed renaming")

    # --- 4: mark and join ---------------------------------------------
    identities = add_identity_marker(identities, IDENTITY_FLAG_COLUMN)
    merged = join_transaction_identity(
        transactions,
        identities,
        join_key=JOIN_KEY,
        flag_column=IDENTITY_FLAG_COLUMN,
    )

    # Free the two source tables. They are no longer needed and together
    # they take up as much memory as the joined table does.
    del transactions, identities

    # --- 5: validate ---------------------------------------------------
    if is_sample_run:
        print("  Skipping shape validation (sample run)")
        validation = {"passed": True, "skipped": True}
    else:
        validation = validate_join(
            merged,
            expected_rows=settings["expected_rows"],
            expected_columns=settings["expected_columns"],
            join_key=JOIN_KEY,
        )

        print(
            f"  Shape: {validation['actual_rows']:,} rows x "
            f"{validation['actual_columns']} columns "
            f"(expected {validation['expected_rows']:,} x "
            f"{validation['expected_columns']})"
        )

        if not validation["passed"]:
            print("  WARNING: the join did not produce the expected shape.")
            if not validation["rows_match"]:
                print("    Row count differs.")
            if not validation["columns_match"]:
                print("    Column count differs.")
            if not validation["keys_unique"]:
                print(f"    {validation['duplicate_keys']} duplicate join keys.")

    # --- 6: shrink and save --------------------------------------------
    merged, memory_summary = optimise_dtypes(merged, verbose=True)

    print("  Column types after optimisation:")
    for dtype_name, count in dtype_breakdown(merged).items():
        print(f"    {dtype_name:<12} {count:>4} columns")

    size_mb = save_parquet(merged, settings["output_file"])

    return {
        "split": split,
        "rows": len(merged),
        "columns": merged.shape[1],
        "identity_matches": int(merged[IDENTITY_FLAG_COLUMN].sum()),
        "memory_before_mb": memory_summary["before_mb"],
        "memory_after_mb": memory_summary["after_mb"],
        "memory_reduction_pct": memory_summary["reduction_pct"],
        "parquet_size_mb": round(size_mb, 1),
        "output_file": str(settings["output_file"]),
        "validation_passed": validation["passed"],
    }


def run_ingestion(splits: list[str] | None = None, nrows: int | None = None) -> dict:
    """
    Run ingestion for the requested splits. Defaults to both.

    Returns a dictionary keyed by split name, so the caller can print or
    log a summary of everything that happened.
    """
    if splits is None:
        splits = ["train", "test"]

    print("=" * 60)
    print("STAGE: INGESTION")
    print("=" * 60)

    results = {}
    for split in splits:
        results[split] = ingest_split(split, nrows=nrows)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    for split, result in results.items():
        status = "OK" if result["validation_passed"] else "CHECK WARNINGS"
        print(
            f"  {split:<6} {result['rows']:>9,} rows x "
            f"{result['columns']:>3} cols   "
            f"{result['memory_after_mb']:>7,.1f} MB in memory   "
            f"{result['parquet_size_mb']:>6,.1f} MB on disk   {status}"
        )

    return results
```

---

## 10. Create `run.py`

### 10.1 Why a single entry point

Without it, you end up with a folder of scripts and no obvious order. Six months later, or when a reviewer opens the repo, nobody knows which one runs first.

One entry point with named stages solves that. `python run.py --step ingestion` reads as an instruction. It also gives you `--step all` for a clean full rebuild, which is what CI will call in Step 5 and what Docker will call in Step 6.

This is the same pattern NovaPay used, carried forward. Originally planned for Step 3, brought forward because there are now two stages to run. That is D-14.

### 10.2 The file

Create `run.py` in the project root:

```python
"""
run.py: single entry point for the IEEE-CIS Fraud Detection pipeline.

Usage:
    python run.py --step ingestion
    python run.py --step ingestion --split train
    python run.py --step ingestion --nrows 5000     # quick smoke test
    python run.py --step eda
    python run.py --step all

Stages available so far:
    ingestion   Load raw CSVs, join transaction to identity, save Parquet
    eda         Profile the joined training data, write reports and charts
    all         Every stage above, in order

More stages are added in Steps 3 to 6.
"""

from __future__ import annotations

import argparse
import time


def run_ingestion_stage(args: argparse.Namespace) -> dict:
    # Imported inside the function rather than at the top of the file so
    # that starting the script is fast even when a stage is not being used.
    # pandas alone takes about a second to import.
    from src.pipelines.ingestion import run_ingestion

    splits = ["train", "test"] if args.split == "both" else [args.split]
    return run_ingestion(splits=splits, nrows=args.nrows)


def run_eda_stage(args: argparse.Namespace) -> dict:
    from src.pipelines.eda import run_eda

    return run_eda()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IEEE-CIS Fraud Detection pipeline runner",
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["ingestion", "eda", "all"],
        help="Which pipeline stage to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["train", "test", "both"],
        help="Which data split to ingest. Only affects the ingestion stage.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Read only this many rows. For quick testing, not real runs.",
    )
    args = parser.parse_args()

    started_at = time.time()

    if args.step == "ingestion":
        run_ingestion_stage(args)
    elif args.step == "eda":
        run_eda_stage(args)
    elif args.step == "all":
        run_ingestion_stage(args)
        run_eda_stage(args)

    elapsed = time.time() - started_at
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\nFinished in {minutes}m {seconds}s.")


# Only run when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()
```

---

## 11. Run the ingestion stage

### 11.1 Smoke test first

Always test on a small slice before committing to a full run. If something is wrong, you find out in fifteen seconds instead of five minutes.

```powershell
# Read only the first 5,000 rows of each file
python run.py --step ingestion --split train --nrows 5000
```

**What you should see:** loading messages, a rename message saying nothing needed renaming (correct, train already uses underscores), a join message, a memory reduction line, a dtype breakdown, and a small Parquet file written.

If that works, delete the sample output so it does not get mistaken for the real thing:

```powershell
Remove-Item data\interim\train_joined.parquet
```

### 11.2 The real run

```powershell
python run.py --step ingestion
```

This takes roughly 5 to 10 minutes on your machine. Most of it is pandas reading two large CSVs. The terminal will look idle at times, which is normal.

**Expected output, abbreviated:**

```
============================================================
STAGE: INGESTION
============================================================

------------------------------------------------------------
INGESTING: train
------------------------------------------------------------
  Reading train_transaction.csv ...
    590,540 rows x 394 columns
  Reading train_identity.csv ...
    144,233 rows x 41 columns
  No identity columns needed renaming
  Joining on TransactionID ...
    144,233 of 590,540 transactions matched an identity record (24.4%)
  Shape: 590,540 rows x 435 columns (expected 590,540 x 435)
  Memory: 2,0xx.x MB -> 6xx.x MB (approximately 70% smaller)
  Column types after optimisation:
    float32       3xx columns
    ...
  Writing train_joined.parquet ...
    3xx.x MB on disk

------------------------------------------------------------
INGESTING: test
------------------------------------------------------------
  ...
  Renamed 38 identity columns from id-NN to id_NN
  ...
  Shape: 506,691 rows x 434 columns (expected 506,691 x 434)
  ...
```

### 11.3 What to check in that output

Four things, and all four matter.

1. **Train renames 0 columns, test renames 38.** That is the naming quirk being handled exactly where it should be. If train renamed anything, something is wrong.
2. **Identity match count is 144,233, which is 24.4%.** This matches your Step 1 verification exactly, so the join did what it was supposed to.
3. **Shapes match the expected figures.** 590,540 by 435 and 506,691 by 434. If either differs, stop and tell me before continuing.
4. **Memory drops by roughly 65 to 75%.** The exact figure varies slightly. Anything in that range is healthy. A much smaller reduction suggests the optimiser skipped columns it should have shrunk.

### 11.4 Confirm the files exist

```powershell
Get-ChildItem data\interim | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,1)}}
```

**Expected:** two Parquet files, roughly 250 to 400 MB each.

### 11.5 Confirm a reload gives back the same types

Parquet's advantage over CSV is that it remembers types. Prove it:

```powershell
python -c "import pandas as pd; from config.config import JOINED_TRAIN_FILE; df = pd.read_parquet(JOINED_TRAIN_FILE, columns=['TransactionID','TransactionDT','TransactionAmt','isFraud','ProductCD','has_identity']); print(df.dtypes); print(df.head())"
```

**Expected:** `TransactionID` as int32, `TransactionDT` as int32, `TransactionAmt` as float64, `isFraud` as int8, `ProductCD` as category, `has_identity` as int8.

That is decision D-18 working as designed. Notice how fast that command was despite the file being hundreds of megabytes: Parquet stores data column by column, so asking for six columns reads only those six.

---

## 12. Create `src/utils/eda_utils.py`

### 12.1 What is in here

Three kinds of function.

**Profiling:** work out which family each column belongs to, then measure missingness, uniqueness, and range for every column.

**Pattern analysis:** group the V columns by identical missing pattern, and compute fraud rates by category.

**Charts:** ten plotting functions, each producing one PNG.

### 12.2 A note on the chart setup

Two details in the code that are worth knowing about.

`matplotlib.use("Agg")` switches matplotlib to a mode that draws straight to a file with no window. Without it, running a plotting script from a terminal can try to open a window, which either flashes up annoyingly or fails outright depending on the environment. In Docker in Step 6 it would definitely fail. Setting it once, before pyplot is imported, avoids all of that.

`plt.close(fig)` after saving each chart releases the figure's memory. Skip it and matplotlib holds every figure you ever made, then warns you about it after twenty of them.

### 12.3 The file

Create `src/utils/eda_utils.py`:

```python
"""
Analysis and charting helpers for the EDA stage.

Three groups of function:
  1. Profiling: assign columns to families, measure missingness and spread
  2. Pattern analysis: V column missing blocks, fraud rate by category
  3. Charts: one function per figure, each saving a PNG

Matplotlib is set to the "Agg" backend, which draws directly to a file with
no window. This has to happen before pyplot is imported. Without it, running
this from a terminal or inside a container tries to open a display and fails.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config.config import (  # noqa: E402
    MIN_CATEGORY_COUNT,
    REFERENCE_DATETIME,
    SECONDS_PER_DAY,
    TARGET_COLUMN,
    TIME_COLUMN,
)

# A single visual style for every chart, so the report looks coherent.
FRAUD_COLOUR = "#c0392b"
LEGIT_COLOUR = "#2c7fb8"
NEUTRAL_COLOUR = "#7f8c8d"

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


# =========================================================
# 1. Profiling
# =========================================================

def family_for_column(name: str) -> str:
    """
    Work out which feature family a column belongs to, from its name.

    re.fullmatch requires the ENTIRE name to match the pattern, unlike
    re.match which only checks the start. That distinction matters here:
    a plain re.match on "C\\d" would also match "card1", putting a card
    column into the counting family.

    Order matters. Exact names are checked before patterns, so that a
    specific rule always beats a general one.
    """
    exact_names = {
        "TransactionID": "identifier",
        "isFraud": "target",
        "TransactionDT": "time",
        "TransactionAmt": "amount",
        "ProductCD": "product",
        "has_identity": "engineered",
        "addr1": "address",
        "addr2": "address",
        "dist1": "distance",
        "dist2": "distance",
        "DeviceType": "device",
        "DeviceInfo": "device",
    }
    if name in exact_names:
        return exact_names[name]

    if name.endswith("_emaildomain"):
        return "email"
    if re.fullmatch(r"card\d", name):
        return "card"
    if re.fullmatch(r"C\d{1,2}", name):
        return "counting_C"
    if re.fullmatch(r"D\d{1,2}", name):
        return "timedelta_D"
    if re.fullmatch(r"M\d", name):
        return "match_M"
    if re.fullmatch(r"V\d{1,3}", name):
        return "vesta_V"
    if re.fullmatch(r"id_\d{2}", name):
        return "identity_id"

    # Anything reaching here is a column we did not anticipate. The caller
    # reports these loudly rather than letting them disappear.
    return "unmapped"


def profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row of description per column.

    Records family, dtype, how much is missing, how many distinct values
    there are, and for numeric columns the smallest and largest value.

    This table is the reference document for Step 3. Feature engineering
    decisions get made from it.
    """
    row_count = len(frame)
    records = []

    for column in frame.columns:
        series = frame[column]
        missing_count = int(series.isna().sum())

        record = {
            "column": column,
            "family": family_for_column(column),
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": round(missing_count / row_count * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
        }

        # min and max only make sense for numbers. Calling them on a
        # category of text either fails or returns something meaningless.
        if pd.api.types.is_numeric_dtype(series):
            record["min_value"] = float(series.min()) if missing_count < row_count else np.nan
            record["max_value"] = float(series.max()) if missing_count < row_count else np.nan
        else:
            record["min_value"] = np.nan
            record["max_value"] = np.nan

        records.append(record)

    profile = pd.DataFrame(records)
    return profile.sort_values(["family", "column"]).reset_index(drop=True)


def family_summary(profile: pd.DataFrame) -> pd.DataFrame:
    """Roll the column profile up to one row per family."""
    summary = (
        profile.groupby("family", observed=True)
        .agg(
            columns=("column", "count"),
            mean_missing_pct=("missing_pct", "mean"),
            max_missing_pct=("missing_pct", "max"),
        )
        .round(2)
        .sort_values("columns", ascending=False)
        .reset_index()
    )
    return summary


# =========================================================
# 2. Pattern analysis
# =========================================================

def missing_pattern_groups(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Group columns that share an identical missing value pattern.

    How it works: for each column, produce a True/False array marking which
    rows are blank, then run that array through a hash function. A hash
    turns any amount of data into a short fixed-length fingerprint, and two
    inputs produce the same fingerprint only if they are identical. So
    columns landing on the same fingerprint have exactly the same rows
    blank, on all 590,540 rows.

    Why this matters: Vesta built the V features in batches from shared
    source data. When a source was unavailable, every feature built from it
    went blank together. Columns in one block therefore tend to measure
    closely related things, which gives Step 3 a principled basis for
    cutting 339 V columns down to a manageable number.
    """
    available = [column for column in columns if column in frame.columns]
    fingerprints: dict[str, list[str]] = {}

    for column in available:
        blank_mask = frame[column].isna().to_numpy()
        # .tobytes() turns the array into raw bytes so it can be hashed.
        fingerprint = hashlib.md5(blank_mask.tobytes()).hexdigest()
        fingerprints.setdefault(fingerprint, []).append(column)

    records = []
    row_count = len(frame)

    for group_index, (fingerprint, group_columns) in enumerate(
        sorted(fingerprints.items(), key=lambda item: -len(item[1])), start=1
    ):
        missing_count = int(frame[group_columns[0]].isna().sum())
        records.append(
            {
                "group_id": group_index,
                "n_columns": len(group_columns),
                "missing_pct": round(missing_count / row_count * 100, 2),
                "columns": ", ".join(group_columns),
            }
        )

    return pd.DataFrame(records)


def fraud_rate_by_category(
    frame: pd.DataFrame,
    column: str,
    min_count: int = MIN_CATEGORY_COUNT,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Fraud rate per distinct value of a column.

    Two deliberate choices.

    Blank is treated as its own category labelled "(missing)" rather than
    being dropped, because "we do not know this customer's email domain" is
    a real and potentially predictive situation.

    Categories with fewer than min_count transactions are excluded. Without
    that floor, a category with 3 transactions and 1 fraud reports a 33%
    fraud rate and dominates the chart while meaning nothing.
    """
    # Convert to plain text first. Calling fillna on a category column with
    # a value that is not already one of its categories raises an error.
    labels = frame[column].astype("object").fillna("(missing)")

    working = pd.DataFrame(
        {"category": labels, "target": frame[TARGET_COLUMN].to_numpy()}
    )

    grouped = (
        working.groupby("category", observed=True)["target"]
        .agg(transactions="size", frauds="sum")
        .reset_index()
    )
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"]
    grouped = grouped[grouped["transactions"] >= min_count]
    grouped = grouped.sort_values("fraud_rate", ascending=False)

    if top_n is not None:
        grouped = grouped.head(top_n)

    return grouped.reset_index(drop=True)


def derive_time_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build a small table of readable time columns from TransactionDT.

    Only the few columns needed for time charts are copied out, rather than
    adding columns to the 435-column table, which would mean duplicating
    hundreds of megabytes for no reason.

    The reference date is a community convention that puts the first
    transaction on 1 December 2017. It exists so chart axes read as dates
    instead of raw second counts. No modelling decision depends on it.
    """
    reference = pd.Timestamp(REFERENCE_DATETIME)
    seconds = frame[TIME_COLUMN]

    timestamps = reference + pd.to_timedelta(seconds, unit="s")

    time_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "date": timestamps.dt.floor("D"),
            "hour": timestamps.dt.hour.astype("int8"),
            "day_of_week": timestamps.dt.dayofweek.astype("int8"),
            "day_index": (seconds // SECONDS_PER_DAY).astype("int32"),
        }
    )

    if TARGET_COLUMN in frame.columns:
        time_frame[TARGET_COLUMN] = frame[TARGET_COLUMN].to_numpy()

    return time_frame


def time_range_summary(frame: pd.DataFrame, label: str) -> dict:
    """Describe the time span a table covers, in both seconds and dates."""
    reference = pd.Timestamp(REFERENCE_DATETIME)
    minimum = int(frame[TIME_COLUMN].min())
    maximum = int(frame[TIME_COLUMN].max())

    return {
        "label": label,
        "min_seconds": minimum,
        "max_seconds": maximum,
        "span_days": round((maximum - minimum) / SECONDS_PER_DAY, 1),
        "start_date": (reference + pd.to_timedelta(minimum, unit="s")).date().isoformat(),
        "end_date": (reference + pd.to_timedelta(maximum, unit="s")).date().isoformat(),
    }


# =========================================================
# 3. Charts
# =========================================================

def _save(figure: plt.Figure, path: Path) -> Path:
    """Save a figure and release its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"    saved {path.name}")
    return path


def plot_class_balance(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of fraud versus legitimate counts, on a log scale."""
    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    total = int(counts.sum())

    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(
        ["Legitimate (0)", "Fraud (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=[LEGIT_COLOUR, FRAUD_COLOUR],
    )

    # A log scale is essential here. On a normal scale the fraud bar is so
    # short next to the legitimate bar that it is barely visible, which is
    # the point being made but makes for a useless chart.
    axis.set_yscale("log")
    axis.set_ylabel("Transactions (log scale)")
    axis.set_title(
        f"Class balance: {counts.get(1, 0):,} fraud out of {total:,} "
        f"({counts.get(1, 0) / total:.3%})"
    )

    for bar, value in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n({value / total:.2%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    return _save(figure, output_dir / "01_class_balance.png")


def plot_amount_distribution(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Transaction amount distribution, fraud against legitimate."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    legitimate = frame.loc[frame[TARGET_COLUMN] == 0, "TransactionAmt"]
    fraudulent = frame.loc[frame[TARGET_COLUMN] == 1, "TransactionAmt"]

    # log=True on the x-axis because amounts are heavily skewed: most
    # transactions are small, a few are enormous. On a linear axis every
    # bar crushes into the leftmost sliver of the chart.
    axes[0].hist(
        np.log10(legitimate.clip(lower=0.01)),
        bins=60,
        alpha=0.6,
        density=True,
        label="Legitimate",
        color=LEGIT_COLOUR,
    )
    axes[0].hist(
        np.log10(fraudulent.clip(lower=0.01)),
        bins=60,
        alpha=0.6,
        density=True,
        label="Fraud",
        color=FRAUD_COLOUR,
    )
    axes[0].set_xlabel("log10(transaction amount in USD)")
    axes[0].set_ylabel("Share of transactions")
    axes[0].set_title("Amount distribution by class")
    axes[0].legend()

    axes[1].boxplot(
        [np.log10(legitimate.clip(lower=0.01)), np.log10(fraudulent.clip(lower=0.01))],
        tick_labels=["Legitimate", "Fraud"],
        showfliers=False,
    )
    axes[1].set_ylabel("log10(transaction amount in USD)")
    axes[1].set_title("Amount spread by class")

    figure.suptitle("Transaction amount", fontweight="bold")
    return _save(figure, output_dir / "02_amount_distribution.png")


def plot_volume_over_time(time_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Daily transaction volume across the training period."""
    daily = time_frame.groupby("date", observed=True).size()

    figure, axis = plt.subplots(figsize=(13, 4.5))
    axis.plot(daily.index, daily.to_numpy(), color=NEUTRAL_COLOUR, linewidth=1.2)
    axis.set_ylabel("Transactions per day")
    axis.set_title("Transaction volume over the training period")
    figure.autofmt_xdate()

    return _save(figure, output_dir / "03_volume_over_time.png")


def plot_fraud_rate_over_time(time_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Daily fraud rate, with the overall average for reference."""
    daily = time_frame.groupby("date", observed=True)[TARGET_COLUMN].agg(
        transactions="size", frauds="sum"
    )
    daily["fraud_rate"] = daily["frauds"] / daily["transactions"]
    overall = time_frame[TARGET_COLUMN].mean()

    figure, axis = plt.subplots(figsize=(13, 4.5))
    axis.plot(daily.index, daily["fraud_rate"], color=FRAUD_COLOUR, linewidth=1.2)
    axis.axhline(
        overall,
        color=NEUTRAL_COLOUR,
        linestyle="--",
        label=f"Overall rate {overall:.2%}",
    )
    axis.set_ylabel("Daily fraud rate")
    axis.set_title("Fraud rate over the training period")
    axis.legend()
    figure.autofmt_xdate()

    return _save(figure, output_dir / "04_fraud_rate_over_time.png")


def plot_fraud_rate_by_hour(time_frame: pd.DataFrame, output_dir: Path) -> Path:
    """Fraud rate by hour of day, with volume behind it for context."""
    hourly = time_frame.groupby("hour", observed=True)[TARGET_COLUMN].agg(
        transactions="size", frauds="sum"
    )
    hourly["fraud_rate"] = hourly["frauds"] / hourly["transactions"]

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(hourly.index, hourly["fraud_rate"], color=FRAUD_COLOUR, alpha=0.85)
    axis.set_xlabel("Hour of day (derived from TransactionDT)")
    axis.set_ylabel("Fraud rate")
    axis.set_title("Fraud rate by hour of day")
    axis.set_xticks(range(0, 24))

    # A second y-axis sharing the same x-axis, so volume can be overlaid
    # without one series flattening the other.
    volume_axis = axis.twinx()
    volume_axis.plot(
        hourly.index, hourly["transactions"], color=NEUTRAL_COLOUR, linewidth=1.5
    )
    volume_axis.set_ylabel("Transactions (line)")
    volume_axis.grid(False)

    return _save(figure, output_dir / "05_fraud_rate_by_hour.png")


def _horizontal_rate_chart(
    rates: pd.DataFrame, title: str, path: Path, height: float = 5.0
) -> Path:
    """Shared drawing code for the 'fraud rate by category' charts."""
    figure, axis = plt.subplots(figsize=(9, height))

    axis.barh(
        rates["category"].astype(str),
        rates["fraud_rate"],
        color=FRAUD_COLOUR,
        alpha=0.85,
    )
    axis.invert_yaxis()  # highest rate at the top
    axis.set_xlabel("Fraud rate")
    axis.set_title(title)

    for index, row in rates.iterrows():
        axis.text(
            row["fraud_rate"],
            index,
            f"  {row['fraud_rate']:.2%}  (n={int(row['transactions']):,})",
            va="center",
            fontsize=9,
        )

    axis.set_xlim(0, rates["fraud_rate"].max() * 1.45)
    return _save(figure, path)


def plot_fraud_rate_by_product(frame: pd.DataFrame, output_dir: Path) -> Path:
    rates = fraud_rate_by_category(frame, "ProductCD")
    return _horizontal_rate_chart(
        rates, "Fraud rate by product code", output_dir / "06_fraud_rate_by_product.png"
    )


def plot_missing_values(profile: pd.DataFrame, output_dir: Path, top_n: int = 40) -> Path:
    """The columns with the most missing data."""
    worst = profile.nlargest(top_n, "missing_pct").sort_values("missing_pct")

    figure, axis = plt.subplots(figsize=(9, 11))
    axis.barh(worst["column"], worst["missing_pct"], color=NEUTRAL_COLOUR)
    axis.set_xlabel("Percentage of values missing")
    axis.set_title(f"Top {top_n} columns by missing data")
    axis.set_xlim(0, 100)

    return _save(figure, output_dir / "07_missing_values.png")


def plot_identity_coverage(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Fraud rate for transactions with an identity record versus without."""
    grouped = frame.groupby("has_identity", observed=True)[TARGET_COLUMN].agg(
        transactions="size", frauds="sum"
    )
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"]

    labels = ["No identity record", "Has identity record"]
    values = [grouped.loc[0, "fraud_rate"], grouped.loc[1, "fraud_rate"]]
    counts = [grouped.loc[0, "transactions"], grouped.loc[1, "transactions"]]

    figure, axis = plt.subplots(figsize=(7.5, 5))
    bars = axis.bar(labels, values, color=[LEGIT_COLOUR, FRAUD_COLOUR])
    axis.set_ylabel("Fraud rate")
    axis.set_title("Fraud rate by whether an identity record exists")

    for bar, rate, count in zip(bars, values, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            rate,
            f"{rate:.2%}\n({count:,} transactions)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.set_ylim(0, max(values) * 1.3)
    return _save(figure, output_dir / "08_identity_coverage.png")


def plot_fraud_rate_by_card(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Fraud rate by card network and by card type, side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for axis, column, title in zip(
        axes, ["card4", "card6"], ["Card network (card4)", "Card type (card6)"]
    ):
        rates = fraud_rate_by_category(frame, column)
        axis.barh(rates["category"].astype(str), rates["fraud_rate"], color=FRAUD_COLOUR)
        axis.invert_yaxis()
        axis.set_xlabel("Fraud rate")
        axis.set_title(title)
        for position, row in rates.iterrows():
            axis.text(
                row["fraud_rate"],
                position,
                f"  {row['fraud_rate']:.2%}",
                va="center",
                fontsize=9,
            )
        axis.set_xlim(0, rates["fraud_rate"].max() * 1.4)

    figure.suptitle("Fraud rate by card attributes", fontweight="bold")
    return _save(figure, output_dir / "09_fraud_rate_by_card.png")


def plot_v_group_sizes(v_groups: pd.DataFrame, output_dir: Path) -> Path:
    """How the 339 V columns divide into shared-missingness blocks."""
    figure, axis = plt.subplots(figsize=(12, 5))

    axis.bar(
        v_groups["group_id"].astype(str),
        v_groups["n_columns"],
        color=NEUTRAL_COLOUR,
    )
    axis.set_xlabel("Missing-pattern block")
    axis.set_ylabel("Number of V columns in the block")
    axis.set_title(
        f"V columns divide into {len(v_groups)} blocks that share "
        f"an identical missing pattern"
    )
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)

    return _save(figure, output_dir / "10_v_column_groups.png")
```

---

## 13. Create `src/pipelines/eda.py`

### 13.1 What this stage does

It reads the joined training Parquet, runs every analysis, writes four report files and ten charts, and prints the headline findings to the terminal.

The written summary is generated by the code rather than typed by hand. That matters: if the data ever changes, the report changes with it. A hand-written report goes stale silently.

### 13.2 The file

Create `src/pipelines/eda.py`:

```python
"""
EDA stage: profile the joined training data and write reports and charts.

Input:  data/interim/train_joined.parquet
        data/interim/test_joined.parquet  (time range only)
Output: reports/eda_summary.md
        reports/column_profile.csv
        reports/missing_profile.csv
        reports/v_column_missing_groups.csv
        reports/figures/*.png

Run with:
    python run.py --step eda
"""

from __future__ import annotations

import pandas as pd

from config.config import (
    COLUMN_PROFILE_FILE,
    EDA_SUMMARY_FILE,
    FIGURES_DIR,
    HIGH_MISSING_THRESHOLD,
    IDENTITY_FLAG_COLUMN,
    JOINED_TEST_FILE,
    JOINED_TRAIN_FILE,
    MISSING_PROFILE_FILE,
    TARGET_COLUMN,
    TIME_COLUMN,
    V_COLUMNS,
    V_GROUPS_FILE,
    ensure_directories,
)
from src.utils.eda_utils import (
    derive_time_frame,
    family_summary,
    fraud_rate_by_category,
    missing_pattern_groups,
    plot_amount_distribution,
    plot_class_balance,
    plot_fraud_rate_by_card,
    plot_fraud_rate_by_hour,
    plot_fraud_rate_by_product,
    plot_fraud_rate_over_time,
    plot_identity_coverage,
    plot_missing_values,
    plot_v_group_sizes,
    plot_volume_over_time,
    profile_columns,
    time_range_summary,
)
from src.utils.memory_utils import memory_usage_mb


def _load_training_data() -> pd.DataFrame:
    if not JOINED_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"{JOINED_TRAIN_FILE} not found.\n"
            f"Run  python run.py --step ingestion  first."
        )

    print(f"  Loading {JOINED_TRAIN_FILE.name} ...")
    frame = pd.read_parquet(JOINED_TRAIN_FILE)
    print(
        f"    {frame.shape[0]:,} rows x {frame.shape[1]} columns, "
        f"{memory_usage_mb(frame):,.1f} MB in memory"
    )
    return frame


def _class_balance(frame: pd.DataFrame) -> dict:
    total = len(frame)
    fraud = int(frame[TARGET_COLUMN].sum())
    legitimate = total - fraud
    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legitimate,
        "fraud_rate": fraud / total,
        "ratio": legitimate / fraud,
    }


def _identity_breakdown(frame: pd.DataFrame) -> dict:
    grouped = frame.groupby(IDENTITY_FLAG_COLUMN, observed=True)[TARGET_COLUMN].agg(
        transactions="size", frauds="sum"
    )
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"]

    return {
        "without_identity_transactions": int(grouped.loc[0, "transactions"]),
        "without_identity_fraud_rate": float(grouped.loc[0, "fraud_rate"]),
        "with_identity_transactions": int(grouped.loc[1, "transactions"]),
        "with_identity_fraud_rate": float(grouped.loc[1, "fraud_rate"]),
    }


def _identity_by_product(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate identity coverage against product code.

    This is a check on decision D-16. If having an identity record turns out
    to be almost entirely determined by which product was used, then the
    has_identity flag tells the model nothing that ProductCD does not
    already tell it, and it is redundant rather than useful.

    normalize="index" turns the counts into row percentages, so each product
    row sums to 1 and the products can be compared despite very different
    volumes.
    """
    table = pd.crosstab(
        frame["ProductCD"].astype("object"),
        frame[IDENTITY_FLAG_COLUMN],
        normalize="index",
    )
    table.columns = ["no_identity_share", "has_identity_share"]
    return (table * 100).round(1).reset_index()


def _test_time_range() -> dict | None:
    """
    Read only TransactionDT from the test Parquet.

    Parquet stores data column by column, so asking for one column reads
    only that column. This is near instant even though the file is hundreds
    of megabytes.
    """
    if not JOINED_TEST_FILE.exists():
        print("  Test Parquet not found, skipping the train-versus-test time comparison.")
        return None

    test_time = pd.read_parquet(JOINED_TEST_FILE, columns=[TIME_COLUMN])
    return time_range_summary(test_time, "test")


def _write_summary(results: dict) -> None:
    """Assemble the markdown report from the computed results."""
    balance = results["balance"]
    identity = results["identity"]
    train_time = results["train_time"]
    test_time = results["test_time"]

    lines: list[str] = []
    add = lines.append

    add("# EDA Summary: IEEE-CIS Fraud Detection")
    add("")
    add("Generated automatically by `src/pipelines/eda.py`. "
        "Do not edit by hand, it is overwritten on every run.")
    add("")

    # --- shape ---
    add("## 1. Dataset shape")
    add("")
    add(f"- Joined training table: **{results['rows']:,} rows x "
        f"{results['columns']} columns**")
    add(f"- In-memory size after type optimisation: "
        f"**{results['memory_mb']:,.1f} MB**")
    add("")

    # --- balance ---
    add("## 2. Class balance")
    add("")
    add(f"- Total transactions: **{balance['total']:,}**")
    add(f"- Fraudulent: **{balance['fraud']:,}**")
    add(f"- Legitimate: **{balance['legitimate']:,}**")
    add(f"- Fraud rate: **{balance['fraud_rate']:.4%}**")
    add(f"- Roughly 1 fraud per **{balance['ratio']:.0f}** legitimate transactions")
    add("")
    add("A model that predicted \"never fraud\" would score "
        f"**{1 - balance['fraud_rate']:.2%} accuracy** while being useless. "
        "Accuracy is not used as a metric on this project.")
    add("")

    # --- identity ---
    add("## 3. Identity coverage")
    add("")
    add("| Group | Transactions | Fraud rate |")
    add("|-------|--------------|------------|")
    add(f"| No identity record | {identity['without_identity_transactions']:,} | "
        f"{identity['without_identity_fraud_rate']:.4%} |")
    add(f"| Has identity record | {identity['with_identity_transactions']:,} | "
        f"{identity['with_identity_fraud_rate']:.4%} |")
    add("")

    lift = (
        identity["with_identity_fraud_rate"] / identity["without_identity_fraud_rate"]
        if identity["without_identity_fraud_rate"]
        else float("nan")
    )
    add(f"Fraud is **{lift:.2f}x** as likely among transactions that have an "
        "identity record. The presence or absence of that record is therefore "
        "informative in itself, which is why `has_identity` is kept as a feature.")
    add("")
    add("Identity coverage by product code, as a percentage of each product's "
        "transactions:")
    add("")
    add(results["identity_by_product"].to_markdown(index=False))
    add("")

    # --- time ---
    add("## 4. Time coverage")
    add("")
    add("| Split | First | Last | Span (days) |")
    add("|-------|-------|------|-------------|")
    add(f"| train | {train_time['start_date']} | {train_time['end_date']} | "
        f"{train_time['span_days']} |")
    if test_time:
        add(f"| test | {test_time['start_date']} | {test_time['end_date']} | "
            f"{test_time['span_days']} |")
    add("")

    if test_time:
        gap_days = round(
            (test_time["min_seconds"] - train_time["max_seconds"]) / 86400, 1
        )
        add(f"There is a gap of **{gap_days} days** between the last training "
            "transaction and the first test transaction. The test set is "
            "entirely in the future relative to training.")
        add("")
        add("**Consequence:** validation must be a time-based split, never a "
            "random one. A random split would let the model learn from "
            "transactions that happened after the ones it is validated on, "
            "producing a validation score that cannot be reproduced in "
            "production.")
        add("")

    # --- families ---
    add("## 5. Feature families")
    add("")
    add(results["families"].to_markdown(index=False))
    add("")

    if results["unmapped_columns"]:
        add(f"**Warning:** {len(results['unmapped_columns'])} columns did not "
            "match any known family: "
            f"{', '.join(results['unmapped_columns'])}")
        add("")

    # --- missing ---
    add("## 6. Missing data")
    add("")
    add(f"- Columns with no missing values at all: "
        f"**{results['columns_no_missing']}**")
    add(f"- Columns missing more than "
        f"{HIGH_MISSING_THRESHOLD:.0%} of their values: "
        f"**{results['columns_high_missing']}**")
    add("")
    add("The 25 emptiest columns:")
    add("")
    add(results["worst_missing"].to_markdown(index=False))
    add("")

    # --- V groups ---
    add("## 7. V column structure")
    add("")
    add(f"The {len(V_COLUMNS)} V columns fall into "
        f"**{results['v_group_count']} blocks** that share an identical "
        "missing value pattern.")
    add("")
    add("Vesta engineered these features in batches from shared source data. "
        "When a source was unavailable for a transaction, every feature "
        "derived from it went blank together. Columns inside one block are "
        "therefore usually closely related, which gives Step 3 a principled "
        "way to reduce 339 columns to a manageable number: keep a "
        "representative from each block instead of dropping columns "
        "arbitrarily.")
    add("")
    add("The ten largest blocks:")
    add("")
    add(results["v_groups"].head(10)[["group_id", "n_columns", "missing_pct"]]
        .to_markdown(index=False))
    add("")
    add("Full detail in `reports/v_column_missing_groups.csv`.")
    add("")

    # --- categorical rates ---
    add("## 8. Fraud rate by key categorical columns")
    add("")
    for title, table in results["category_rates"].items():
        add(f"### {title}")
        add("")
        display = table.copy()
        display["fraud_rate"] = display["fraud_rate"].map(lambda value: f"{value:.2%}")
        add(display.to_markdown(index=False))
        add("")

    # --- decisions ---
    add("## 9. Decisions carried into Step 3")
    add("")
    add("1. **Primary metric is PR-AUC.** ROC-AUC is reported alongside it, "
        "since it was the competition metric. Accuracy is not used.")
    add("2. **Validation is time-based.** The last 20% of the training period "
        "by `TransactionDT` becomes the validation set. No random shuffling.")
    add("3. **Missing values stay missing.** LightGBM, XGBoost, and CatBoost "
        "all learn a direction for missing values at each split. Filling "
        "blanks with an average would assert something false.")
    add("4. **`has_identity` is kept** as an explicit feature.")
    add("5. **V columns are reduced using the block structure** identified "
        "above, rather than by an arbitrary correlation cutoff.")
    add("")

    EDA_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {EDA_SUMMARY_FILE.name}")


def run_eda() -> dict:
    """Run the full EDA stage and return the computed results."""
    print("=" * 60)
    print("STAGE: EDA")
    print("=" * 60)

    ensure_directories()

    frame = _load_training_data()

    # --- profiling -----------------------------------------------------
    print("\n  Profiling columns ...")
    profile = profile_columns(frame)
    profile.to_csv(COLUMN_PROFILE_FILE, index=False)
    print(f"    wrote {COLUMN_PROFILE_FILE.name}")

    missing_profile = (
        profile[["column", "family", "dtype", "missing_count", "missing_pct"]]
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    missing_profile.to_csv(MISSING_PROFILE_FILE, index=False)
    print(f"    wrote {MISSING_PROFILE_FILE.name}")

    families = family_summary(profile)
    unmapped = profile.loc[profile["family"] == "unmapped", "column"].tolist()
    if unmapped:
        print(f"    WARNING: {len(unmapped)} unmapped columns: {unmapped}")

    # --- V column blocks -----------------------------------------------
    print("\n  Grouping V columns by missing pattern ...")
    v_groups = missing_pattern_groups(frame, V_COLUMNS)
    v_groups.to_csv(V_GROUPS_FILE, index=False)
    print(f"    {len(v_groups)} distinct blocks across {len(V_COLUMNS)} V columns")
    print(f"    wrote {V_GROUPS_FILE.name}")

    # --- headline statistics --------------------------------------------
    print("\n  Computing summary statistics ...")
    balance = _class_balance(frame)
    identity = _identity_breakdown(frame)
    identity_by_product = _identity_by_product(frame)
    train_time = time_range_summary(frame, "train")
    test_time = _test_time_range()

    time_frame = derive_time_frame(frame)

    category_rates = {
        "Product code (ProductCD)": fraud_rate_by_category(frame, "ProductCD"),
        "Card network (card4)": fraud_rate_by_category(frame, "card4"),
        "Card type (card6)": fraud_rate_by_category(frame, "card6"),
        "Device type (DeviceType)": fraud_rate_by_category(frame, "DeviceType"),
        "Purchaser email domain, top 15 by fraud rate": fraud_rate_by_category(
            frame, "P_emaildomain", top_n=15
        ),
    }

    # --- charts -----------------------------------------------------------
    print("\n  Generating charts ...")
    plot_class_balance(frame, FIGURES_DIR)
    plot_amount_distribution(frame, FIGURES_DIR)
    plot_volume_over_time(time_frame, FIGURES_DIR)
    plot_fraud_rate_over_time(time_frame, FIGURES_DIR)
    plot_fraud_rate_by_hour(time_frame, FIGURES_DIR)
    plot_fraud_rate_by_product(frame, FIGURES_DIR)
    plot_missing_values(profile, FIGURES_DIR)
    plot_identity_coverage(frame, FIGURES_DIR)
    plot_fraud_rate_by_card(frame, FIGURES_DIR)
    plot_v_group_sizes(v_groups, FIGURES_DIR)

    # --- report ------------------------------------------------------------
    print("\n  Writing summary report ...")
    results = {
        "rows": len(frame),
        "columns": frame.shape[1],
        "memory_mb": memory_usage_mb(frame),
        "balance": balance,
        "identity": identity,
        "identity_by_product": identity_by_product,
        "train_time": train_time,
        "test_time": test_time,
        "families": families,
        "unmapped_columns": unmapped,
        "columns_no_missing": int((profile["missing_pct"] == 0).sum()),
        "columns_high_missing": int(
            (profile["missing_pct"] > HIGH_MISSING_THRESHOLD * 100).sum()
        ),
        "worst_missing": missing_profile.head(25),
        "v_groups": v_groups,
        "v_group_count": len(v_groups),
        "category_rates": category_rates,
    }
    _write_summary(results)

    # --- terminal headline --------------------------------------------------
    print("\n" + "=" * 60)
    print("EDA HEADLINES")
    print("=" * 60)
    print(f"  Rows x columns        : {results['rows']:,} x {results['columns']}")
    print(f"  Fraud rate            : {balance['fraud_rate']:.4%} "
          f"({balance['fraud']:,} of {balance['total']:,})")
    print(f"  Fraud rate, no identity : "
          f"{identity['without_identity_fraud_rate']:.4%}")
    print(f"  Fraud rate, has identity: "
          f"{identity['with_identity_fraud_rate']:.4%}")
    print(f"  Train period          : {train_time['start_date']} to "
          f"{train_time['end_date']} ({train_time['span_days']} days)")
    if test_time:
        gap = (test_time["min_seconds"] - train_time["max_seconds"]) / 86400
        print(f"  Test period           : {test_time['start_date']} to "
              f"{test_time['end_date']} ({test_time['span_days']} days)")
        print(f"  Gap between them      : {gap:.1f} days")
    print(f"  V column blocks       : {results['v_group_count']}")
    print(f"  Columns >90% missing  : {results['columns_high_missing']}")
    print(f"\n  Full report: {EDA_SUMMARY_FILE}")

    return results
```

---

## 14. Run the EDA stage

```powershell
python run.py --step eda
```

Roughly 3 to 6 minutes. The slowest parts are counting distinct values across 435 columns and drawing the charts.

**Expected terminal output, abbreviated:**

```
============================================================
STAGE: EDA
============================================================
  Loading train_joined.parquet ...
    590,540 rows x 435 columns, 6xx.x MB in memory

  Profiling columns ...
    wrote column_profile.csv
    wrote missing_profile.csv

  Grouping V columns by missing pattern ...
    xx distinct blocks across 339 V columns
    wrote v_column_missing_groups.csv

  Computing summary statistics ...

  Generating charts ...
    saved 01_class_balance.png
    ... through 10_v_column_groups.png

  Writing summary report ...
  Wrote eda_summary.md

============================================================
EDA HEADLINES
============================================================
  Rows x columns        : 590,540 x 435
  Fraud rate            : 3.4990% (20,663 of 590,540)
  Fraud rate, no identity : ...
  Fraud rate, has identity: ...
  Train period          : 2017-12-01 to ...
  Test period           : ... to ...
  Gap between them      : ~30 days
  V column blocks       : ...
  Columns >90% missing  : ...
```

### 14.1 Confirm the outputs exist

```powershell
Get-ChildItem reports -File | Select-Object Name
Get-ChildItem reports\figures | Select-Object Name
```

**Expected:** five files in `reports` (the four new ones plus `data_inventory.md` from Step 1), and ten PNGs in `reports\figures`.

Open the report and skim the charts:

```powershell
code reports\eda_summary.md
Invoke-Item reports\figures
```

---

## 15. Reading your results

The numbers below are the ones your own run produces. Here is what to look at and what each thing means.

### 15.1 The four checks that must pass

| Check | Expected | If it does not match |
|-------|----------|----------------------|
| Rows and columns | 590,540 x 435 | Stop. Something went wrong in the join. Send me the ingestion output. |
| Fraud rate | 3.4990% | Stop. This must match Step 1 exactly. If it does not, the join dropped or duplicated rows. |
| Unmapped columns | Zero, no warning printed | Send me the column names. A column exists that I did not anticipate. |
| Train span | Roughly 182 to 184 days | A very different figure means `TransactionDT` was damaged during type optimisation. |

### 15.2 What to look for in each chart

**`01_class_balance.png`.** The visual argument for why accuracy is worthless here. Keep this one, it goes in the README.

**`02_amount_distribution.png`.** Look at whether the fraud distribution sits left or right of the legitimate one. If fraudulent transactions cluster at different amounts, `TransactionAmt` and features derived from it will matter in Step 3.

**`03_volume_over_time.png`.** Look for a sharp spike or a sudden drop. Spikes usually mean a shopping event such as Black Friday or Christmas. A drop to near zero means missing days, which would affect any time-window feature built in Step 3.

**`04_fraud_rate_over_time.png`.** The important one for Step 5. If the daily fraud rate wanders substantially rather than sitting flat on the average line, then fraud behaviour is changing over the period. That is exactly the drift a monitoring system exists to catch, and it makes your Step 5 story real rather than theoretical.

**`05_fraud_rate_by_hour.png`.** Note the two series: bars are the fraud rate, the line is the volume. If the fraud rate peaks during the hours when volume is lowest, that is a strong overnight signal and hour becomes an obvious Step 3 feature.

**`06_fraud_rate_by_product.png`.** Product codes have very different fraud rates in this dataset. Look at which one is worst and how large the gap is.

**`07_missing_values.png`.** How many columns sit at or near 100%. Those are candidates for removal in Step 3, though "mostly missing" does not automatically mean useless, since the fact that a value exists at all can be the signal.

**`08_identity_coverage.png`.** The direct test of decision D-16. If the two bars are close, `has_identity` adds little. If one is clearly higher, it is a real feature.

**`09_fraud_rate_by_card.png`.** Whether card network or card type separates fraud. Typically card type separates more strongly than network.

**`10_v_column_groups.png`.** The size of your Step 3 reduction opportunity. A small number of blocks covering 339 columns means large redundancy and a big reduction available.

### 15.3 What the CSVs are for

`column_profile.csv` is the working document for Step 3. Open it in Excel, sort by `missing_pct` descending, then by `unique_count` ascending. Columns with one unique value carry no information at all and get dropped. Columns that are almost entirely missing get reviewed.

`v_column_missing_groups.csv` drives the V column reduction in Step 3.

---

## 16. The metric decision, explained properly

This is decision D-20, and it is the single most important analytical decision in the project. It also happens to be the easiest one to explain badly to a non-technical audience, so it is worth getting your explanation straight now for the PM track.

### 16.1 Why accuracy is useless

Accuracy is the share of predictions that were correct.

With a 3.5% fraud rate, a model that says "not fraud" to absolutely everything is correct 96.5% of the time. It has learned nothing. It catches zero fraud. It scores 96.5%.

Any metric that a do-nothing model can score highly on cannot tell you whether your model is good. Accuracy is off the table.

### 16.2 What ROC-AUC measures, and where it misleads

Think of the model as producing a risk score for each transaction rather than a yes or no.

ROC-AUC answers: if you pick one fraudulent transaction and one legitimate transaction at random, how often does the model give the fraudulent one a higher score? A perfect model scores 1.0. Random guessing scores 0.5.

That is a genuinely useful measure and it was the competition's official metric, which is why we report it.

Its weakness on rare-event problems is specific. ROC-AUC treats the false positive rate as a proportion of all the legitimate transactions. With 569,877 legitimate transactions, flagging 5,000 of them wrongly is a false positive rate of less than 1%, which barely moves ROC-AUC at all. But in the real world, 5,000 wrongly blocked customers against 20,663 actual frauds is a serious operational problem. ROC-AUC is not sensitive to it, because the legitimate class is so enormous that it absorbs any amount of error.

### 16.3 What PR-AUC measures, and why it is primary here

PR-AUC works with two quantities that map directly onto operations.

**Precision:** of the transactions you flagged, what share were actually fraud. This is the cost to your customers and to your review team.

**Recall:** of all the fraud that happened, what share did you catch. This is the money saved.

There is always a trade-off. Flag more transactions and you catch more fraud but annoy more real customers. PR-AUC summarises how well the model manages that trade-off across every possible level of aggressiveness.

The crucial property is the baseline. A random model's PR-AUC equals the fraud rate, so **0.035 for this dataset**. That is your floor. Any real number you produce is measured against it. On NovaPay the baseline was 0.015 and the best model reached about 0.085, which is roughly five times the baseline.

### 16.4 The metric that will actually convince a business audience

Recall at a fixed review capacity.

Phrase it like this: "If the fraud team can manually review the 1% of transactions we consider riskiest, what share of all fraud do we catch?"

That is 5,905 transactions a period, out of 590,540. It is a real constraint every fraud operation lives under, and the answer is a single number anyone can act on. It will be the headline figure in your README and the number you lead with in the PM walkthrough.

### 16.5 What we report

| Metric | Role |
|--------|------|
| PR-AUC | Primary. Model selection is based on this. Baseline 0.035. |
| ROC-AUC | Secondary. Comparable to the Kaggle leaderboard. |
| Recall at 1% review rate | Business headline. |
| Recall at 5% review rate | Secondary business figure. |
| Precision and recall at the chosen threshold | Operating point, once a threshold is chosen in Step 4. |
| Accuracy | Not reported. |

---

## 17. The validation split decision, explained properly

Decision D-21.

### 17.1 What we are doing

Sort the training data by `TransactionDT`. The earliest 80% becomes the training set, the latest 20% becomes the validation set. No shuffling anywhere.

That gives you roughly 472,400 training rows and 118,100 validation rows, containing roughly 16,500 and 4,100 fraud cases respectively. Plenty on both sides.

### 17.2 Why not a random split

Because the real test set is 30 days in the future. Your validation set has to imitate that situation, or it is measuring the wrong thing.

Three ways a random split leaks information in this specific dataset:

**Card level leakage.** The same card appears in many transactions across the six months. Shuffle randomly and the same card lands in both training and validation. The model learns "this particular card is risky" from the training half and is then rewarded for it in the validation half. That is memorisation, and it does not transfer to cards it has never seen.

**Time delta leakage.** The `D` columns measure days since some earlier event. A model trained on later transactions has effectively seen the outcome of earlier ones through those deltas.

**Fraud pattern leakage.** Fraud arrives in bursts. One compromised card, one attack campaign, produces a cluster of related fraudulent transactions close together in time. Split those randomly and near identical frauds sit on both sides. The model recognises the pattern in validation because it saw its twin in training.

Every one of these inflates your validation score. You would look excellent right up to the point where real data arrived.

### 17.3 The honest trade-off

A time-based split does cost you something. Your validation score will look worse than a random split's would. It is also a single split rather than the average of five folds, so it is a noisier estimate.

We accept both, because a pessimistic score you can trust beats an optimistic one you cannot. Step 4 addresses the noise with time-aware cross-validation, where each fold trains on an expanding window of past data and validates on the period immediately following it, which is the same shape as the real problem repeated several times.

---

## 18. Commit, push, merge, tag

### 18.1 Check what Git can see

```powershell
git status
```

You should see the new and modified code files, and the new files in `reports`. You should **not** see anything under `data/interim`. If you do, `.gitignore` is not covering it. Check that the `data/interim/*` line from Step 1 is still there.

### 18.2 Commit in logical pieces

```powershell
# Configuration
git add config/config.py
git commit -m "feat: extend config with feature families, EDA paths, and split settings"

# Ingestion pipeline
git add src/utils/memory_utils.py src/utils/ingestion_utils.py src/pipelines/ingestion.py
git commit -m "feat: add ingestion pipeline with table join and dtype optimisation"

# Entry point
git add run.py
git commit -m "feat: add run.py pipeline entry point with ingestion and eda stages"

# EDA pipeline
git add src/utils/eda_utils.py src/pipelines/eda.py
git commit -m "feat: add eda pipeline with column profiling and ten charts"

# Generated reports and charts
git add reports/
git commit -m "docs: add generated eda summary, column profiles, and figures"

# Step documentation
git add docs/
git commit -m "docs: add step 2 guide and update project state"

git log --oneline
```

### 18.3 Push and merge

```powershell
git push -u origin step-02-eda

gh pr create --base main --head step-02-eda `
  --title "Step 2: EDA and data understanding" `
  --body "Table join, dtype optimisation, feature family profiling, missing value analysis, V column block detection, imbalance profiling, ten charts, auto-generated EDA summary."

gh pr merge --squash --delete-branch
```

Or use the yellow banner on GitHub, then **Squash and merge**, then **Delete branch**.

### 18.4 Tag the milestone

```powershell
git switch main
git pull

git tag -a v0.2.0-step2 -m "Step 2 complete: EDA and data understanding"
git push origin v0.2.0-step2
```

### 18.5 Update the README roadmap

Open `README.md` and tick Step 2:

```markdown
- [x] Step 1: Dataset acquisition, scaffold, repo, environment
- [x] Step 2: Exploratory data analysis and data understanding
- [ ] Step 3: Feature engineering and preprocessing pipeline
```

While you are there, add a link to the EDA report and drop in the class balance chart. Small touches, but the README is what people actually read.

```markdown
## Exploratory analysis

Full findings in [`reports/eda_summary.md`](reports/eda_summary.md).

![Class balance](reports/figures/01_class_balance.png)
```

Commit that directly to main, since it is documentation only:

```powershell
git add README.md
git commit -m "docs: mark step 2 complete and link the eda report"
git push
```

---

## 19. Verification checklist

**Housekeeping**
- [ ] Decided on Option A (moved the project) or Option B (excluded folders from OneDrive sync)
- [ ] If moved: `.venv` deleted and rebuilt from `requirements.lock.txt`
- [ ] If moved: `python scripts/verify_data.py` still passes from the new location
- [ ] Branch `step-02-eda` created

**Configuration**
- [ ] `config/config.py` replaced with the extended version
- [ ] The length check prints `339 38 14`

**Ingestion**
- [ ] All five new code files created
- [ ] Smoke test with `--nrows 5000` completed and the sample output deleted
- [ ] Full ingestion run completed
- [ ] Train shape is 590,540 x 435
- [ ] Test shape is 506,691 x 434
- [ ] Test renamed 38 columns, train renamed 0
- [ ] Identity match count is 144,233, which is 24.4%
- [ ] Memory reduced by roughly 65 to 75%
- [ ] Both Parquet files exist in `data/interim`
- [ ] Reload check shows int32, float64, int8, and category types preserved

**EDA**
- [ ] EDA stage completed with no unmapped column warning
- [ ] Fraud rate reads 3.4990%, matching Step 1
- [ ] Four report files written to `reports`
- [ ] Ten PNG files written to `reports/figures`
- [ ] `eda_summary.md` opens and reads sensibly
- [ ] Train span is roughly 183 days
- [ ] A gap of roughly 30 days between train and test

**Git**
- [ ] `git status` shows no Parquet or CSV files
- [ ] Commits made, branch pushed, pull request merged
- [ ] Tag `v0.2.0-step2` pushed
- [ ] README roadmap updated

---

## 20. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `MergeError: Merge keys are not unique` | `TransactionID` repeated in one of the tables | Should be impossible given Step 1 verification. Re-run `python scripts/verify_data.py` and send me the output. |
| `ModuleNotFoundError: No module named 'src'` | Running from the wrong folder | `Set-Location` to the folder containing `run.py`. |
| `ModuleNotFoundError: No module named 'config'` | Same cause | Same fix. |
| `PermissionError` when writing Parquet | OneDrive holding the file open, or the Parquet file open in another program | Section 2.3 or 2.4. Also close any tool that has the file open. |
| `ArrowInvalid` on write | A column with genuinely mixed types | Send me the full error, it names the column. |
| Memory reduction under 40% | The optimiser skipped columns | Send me the `Column types after optimisation` block. |
| `FutureWarning` about `observed` | A groupby on a category column without `observed=True` | Every groupby in this code passes it. If you see one, tell me which line. |
| `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'` | Removed in matplotlib 3.9 | None of this code uses it. If it appears, it came from elsewhere. |
| Charts render but are blank | Figure closed before saving | `_save` saves before closing. Send me the code if you changed it. |
| `tick_labels` error in the boxplot | matplotlib older than 3.9 | You are on 3.11.1, so this should not occur. If it does, replace `tick_labels` with `labels`. |
| EDA very slow, over 15 minutes | The distinct value count across 435 columns is the slow part | Normal on first run. Tell me if it exceeds 20 minutes. |
| `KeyError: 0` in the identity breakdown | Every transaction matched or none did | The join went wrong. Send me the ingestion output. |

---

## 21. What to send me before Step 3

1. **The full terminal output** of `python run.py --step ingestion` and `python run.py --step eda`
2. **The contents of `reports/eda_summary.md`** (paste it, or attach the file)
3. **`reports/v_column_missing_groups.csv`** as an attachment. This directly determines the V column reduction strategy in Step 3, and I do not want to guess at your block structure.
4. **Which housekeeping option you chose**, A or B, and the final project path
5. **Your Python version**: `python --version`. I have been writing for 3.11 and want to confirm.
6. **Any checklist item that did not tick**, with the error text

If any figure looks odd to you, say so. An unexpected number here is much cheaper to investigate now than after feature engineering is built on top of it.

---

## 22. What Step 3 will cover

- Dropping the columns that carry no information: single-valued columns, and near-duplicates identified through the V block structure
- Reducing 339 V columns to a manageable set, using the block structure rather than an arbitrary cutoff
- Time features: hour, day of week, and where relevant a position within the overall period
- The `TransactionAmt` decomposition, including the cents feature that decision D-18 protected the precision for
- Frequency encoding for the high cardinality columns such as `card1`, which has thousands of distinct values and cannot be one-hot encoded
- Aggregate features: a transaction's amount compared with the average for its card, its address, its email domain
- UID construction, the widely used technique of combining card and address columns to approximate a customer identity, plus an honest account of when it helps and when it overfits
- Email domain cleaning, since the raw values contain variants of the same provider
- Building all of this as a scikit-learn pipeline object, so exactly the same transformations apply at training time and at prediction time. This is the single most common source of production bugs in machine learning, and the pipeline object is the fix
- The time-based split implemented in code
- The DVC decision, which needs your answer to open question Q-03
- Output written to `data/processed/`

---

*End of Step 2. `PROJECT_STATE.md` follows as a separate document.*
