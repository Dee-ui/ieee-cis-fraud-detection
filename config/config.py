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
