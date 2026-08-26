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

C_COLUMNS = [f"C{i}" for i in range(1, 15)]  # C1 ... C14
D_COLUMNS = [f"D{i}" for i in range(1, 16)]  # D1 ... D15
M_COLUMNS = [f"M{i}" for i in range(1, 10)]  # M1 ... M9
V_COLUMNS = [f"V{i}" for i in range(1, 340)]  # V1 ... V339
IDENTITY_COLUMNS = [f"id_{i:02d}" for i in range(1, 39)]  # id_01 ... id_38

CARD_COLUMNS = [f"card{i}" for i in range(1, 7)]  # card1 ... card6
ADDRESS_COLUMNS = ["addr1", "addr2"]
DISTANCE_COLUMNS = ["dist1", "dist2"]
EMAIL_COLUMNS = ["P_emaildomain", "R_emaildomain"]
DEVICE_COLUMNS = ["DeviceType", "DeviceInfo"]

# Columns that arrive as text rather than numbers. Note this is the
# KNOWN list. The profiling code detects text columns at runtime rather
# than trusting this, and reports any disagreement.
KNOWN_TEXT_COLUMNS = (
    ["ProductCD", "card4", "card6"] + EMAIL_COLUMNS + M_COLUMNS + DEVICE_COLUMNS
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

# A database URL must use forward slashes. Building it from a Windows Path
# produces backslashes, which SQLAlchemy handles inconsistently. .as_posix()
# converts "C:\Users\...\mlflow.db" into "C:/Users/.../mlflow.db", which is
# understood on every platform. This is decision D-42.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
)
MLFLOW_EXPERIMENT_NAME = "ieee-cis-fraud-detection"
REGISTERED_MODEL_NAME = "ieee-cis-fraud-detector"
MODEL_ALIAS_CANDIDATE = "candidate"


# ---------------------------------------------------------
# Splitting and modelling defaults
#
# VALIDATION_FRACTION is the share of TRAINING data held back for
# validation. It is taken from the END of the time range, never at
# random, because the real test set is strictly in the future.
# ---------------------------------------------------------

VALIDATION_FRACTION = 0.2
CV_FOLDS = 5
FRAUD_PROBABILITY_THRESHOLD = 0.5  # placeholder, tuned properly in Step 4


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


# =========================================================
# STEP 3: FEATURE ENGINEERING
# =========================================================

# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

PREPROCESSOR_FILE = MODELS_DIR / "feature_engineer.joblib"
FEATURE_MANIFEST_FILE = REPORTS_DIR / "feature_manifest.csv"
DROPPED_COLUMNS_FILE = REPORTS_DIR / "dropped_columns.csv"
V_REDUCTION_FILE = REPORTS_DIR / "v_column_reduction.csv"
FEATURE_SUMMARY_FILE = REPORTS_DIR / "feature_summary.md"


# ---------------------------------------------------------
# The split column written into the processed files
# ---------------------------------------------------------

SPLIT_COLUMN = "split"
TRAIN_SPLIT_LABEL = "train"
VALID_SPLIT_LABEL = "valid"


# ---------------------------------------------------------
# Columns carried through the pipeline but NEVER used as features.
#
# TransactionID identifies a row and means nothing about fraud.
# TransactionDT is needed for sorting and splitting, but its test values
#   sit entirely outside the training range, so a tree cannot use it.
# isFraud is the answer.
# ---------------------------------------------------------

PASSTHROUGH_COLUMNS = [ID_COLUMN, TIME_COLUMN, TARGET_COLUMN]


# ---------------------------------------------------------
# Column pruning thresholds
# ---------------------------------------------------------

# Drop a column when one single value (blank counts as a value) covers
# this share of all rows or more.
NEAR_CONSTANT_THRESHOLD = 0.99

# A near-constant column is rescued when the rows that do NOT hold the
# dominant value are both numerous enough and unusual enough.
RESCUE_MIN_RARE_ROWS = 500
RESCUE_MIN_FRAUD_LIFT = 2.0

# Two V columns inside the same block are treated as near-duplicates
# when the absolute correlation between them reaches this level.
V_CORRELATION_THRESHOLD = 0.75


# ---------------------------------------------------------
# Text handling
# ---------------------------------------------------------

# Blank values become this label, so that "we do not know" is a real
# category the model can split on rather than a hole.
MISSING_LABEL = "(missing)"

# A value present in test but never seen in training gets this code.
UNSEEN_CATEGORY_CODE = -1


# ---------------------------------------------------------
# Frequency encoding: count how often each value appears in training.
#
# Includes derived columns (uid, card1_addr1, the email and device parts)
# which do not exist in the raw data. The code skips anything missing
# rather than failing, so this list is safe to edit.
# ---------------------------------------------------------

FREQUENCY_ENCODE_COLUMNS = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
    "card1_addr1",
    "uid",
    "P_email_provider",
    "R_email_provider",
    "device_brand",
    "browser_family",
]


# ---------------------------------------------------------
# Aggregate features: (group by this, summarise this).
#
# Each pair produces three columns: the group average, the group spread,
# and the ratio of this row's value to its group average.
# ---------------------------------------------------------

AGGREGATION_SPECS = [
    ("card1", "TransactionAmt"),
    ("addr1", "TransactionAmt"),
    ("card1_addr1", "TransactionAmt"),
    ("uid", "TransactionAmt"),
    ("card1", "D15"),
    ("uid", "D15"),
]


# ---------------------------------------------------------
# Columns used to build the uid customer fingerprint.
# ---------------------------------------------------------

UID_CARD_COLUMN = "card1"
UID_ADDRESS_COLUMN = "addr1"
UID_TIMEDELTA_COLUMN = "D1"


# =========================================================
# STEP 4: MODEL TRAINING
# =========================================================

# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

FINAL_MODEL_FILE = MODELS_DIR / "final_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "final_model_metadata.json"

MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_ANALYSIS_FILE = REPORTS_DIR / "threshold_analysis.csv"
COST_CURVE_FILE = REPORTS_DIR / "cost_curve.csv"
CV_RESULTS_FILE = REPORTS_DIR / "cv_results.csv"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
TRAINING_SUMMARY_FILE = REPORTS_DIR / "training_summary.md"

KAGGLE_SUBMISSION_FILE = PROCESSED_DATA_DIR / "kaggle_submission.csv"


# ---------------------------------------------------------
# The cost model. See step4.md section 3.
#
# THESE ARE ASSUMPTIONS, not figures supplied by a business. Each one has
# stated reasoning behind it. Change any of them and re-run to get a fully
# updated answer.
# ---------------------------------------------------------

# Fully loaded analyst at about $60k a year is roughly $29 an hour. A review
# takes about five minutes, so $2.40. Rounded up for supervision and the
# cases that need a customer call.
COST_REVIEW_PER_CASE = 4.00

# Card networks charge a per-dispute fee on top of the money clawed back.
# Published fees run from roughly $15 to $40.
COST_CHARGEBACK_FEE = 25.00

# Expected cost of holding and releasing a legitimate customer. The softest
# number in the model and the first one to replace with real data.
COST_FALSE_ALARM_FRICTION = 1.00

# Flagging fraud is not the same as stopping it. Some cases are judged
# wrongly and some have already settled.
FRAUD_RECOVERY_RATE = 0.90

# The team can review about one transaction in fifty. The cost model would
# otherwise happily recommend reviewing 15%, which no real team can do.
REVIEW_CAPACITY_RATE = 0.02

# Review rates reported in every summary, for comparison.
HEADLINE_REVIEW_RATES = [0.005, 0.01, 0.02, 0.05]


# ---------------------------------------------------------
# Training settings
# ---------------------------------------------------------

EARLY_STOPPING_ROUNDS = 100
MAX_BOOSTING_ROUNDS = 4000
QUICK_BOOSTING_ROUNDS = 150  # used by run.py --quick

# Expanding-window cross-validation folds, run after a winner is chosen.
CV_N_SPLITS = 4

# Rows sampled for SHAP. Explaining all 118,108 validation rows would take
# far longer and tell you nothing extra.
SHAP_SAMPLE_SIZE = 5000

# Any feature whose name contains one of these belongs to the uid family.
# Quarantined here so the ablation in D-36 can find them by rule rather than
# by a hand-maintained list that would go stale.
UID_FEATURE_MARKERS = ["_by_uid", "_to_uid_", "uid_freq"]

# The ablation decision threshold, set in advance. See D-36.
UID_ABLATION_TOLERANCE = 0.005

# =========================================================
# STEP 5: MONITORING, TESTING, AND PROMOTION
# =========================================================

# ---------------------------------------------------------
# Output locations
# ---------------------------------------------------------

MONITORING_DIR = REPORTS_DIR / "monitoring"

FEATURE_DRIFT_FILE = MONITORING_DIR / "feature_drift.csv"
PERIOD_METRICS_FILE = MONITORING_DIR / "period_metrics.csv"
SCORE_DRIFT_FILE = MONITORING_DIR / "score_drift.csv"
DRIFT_SUMMARY_FILE = MONITORING_DIR / "drift_summary.md"

# Small precomputed file the Step 7 dashboard reads. Per D-45 the dashboard
# must load in under three seconds, so it cannot compute anything from the
# 590,540 row table on page load.
DASHBOARD_DATA_FILE = MONITORING_DIR / "dashboard_data.json"

# The model trained on the training portion only. The final model has seen
# every labelled row, so it cannot score the validation period honestly.
SELECTION_MODEL_FILE = MODELS_DIR / "selection_model.joblib"


# ---------------------------------------------------------
# Drift settings
# ---------------------------------------------------------

# Ten buckets is the convention. More buckets makes PSI jumpier on small
# samples; fewer makes it blind to shifts inside a bucket.
PSI_BINS = 10

PSI_STABLE = 0.10  # below this, no action
PSI_SIGNIFICANT = 0.25  # above this, investigate

# The KS test is slow on very large samples and gains nothing past a point,
# so both sides are subsampled to this size.
KS_SAMPLE_SIZE = 50_000

# How many of the model's most important features get watched closely.
DRIFT_TOP_FEATURES = 20

# A feature needs at least this many usable values in a period before its
# drift number means anything.
DRIFT_MIN_ROWS = 500

# The overall verdict fires on importance-weighted PSI rather than a raw
# count of drifted features. With 284 features, a few will always have
# drifted, and drift in a feature the model ignores does not matter. D-55.
RETRAIN_WEIGHTED_PSI = 0.15
WATCH_WEIGHTED_PSI = 0.05

# How far the alert rate may move from the expected review rate before it
# counts as a problem, as a fraction of the expected rate.
ALERT_RATE_TOLERANCE = 0.50


# ---------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------

MODEL_ALIAS_PRODUCTION = "production"

PROMOTION_MIN_PR_AUC = 0.50
PROMOTION_MAX_CV_SPREAD = 0.05
PROMOTION_REGRESSION_TOLERANCE = 0.01

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
        MONITORING_DIR,
    ]
    for directory in directories:
        # parents=True also creates any missing parent folders
        # exist_ok=True means "do nothing if it is already there"
        directory.mkdir(parents=True, exist_ok=True)
