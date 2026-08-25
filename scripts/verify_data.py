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
    ID_COLUMN,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_IDENTITY_FILE,
    TRAIN_IDENTITY_FILE,
    TRAIN_TRANSACTION_FILE,
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
    print(
        f"  Imbalance ratio    : 1 fraud per {legit_count / fraud_count:.0f} legitimate"
    )

    within_tolerance = abs(fraud_rate - EXPECTED_FRAUD_RATE) < FRAUD_RATE_TOLERANCE
    print(
        f"  Expected about {EXPECTED_FRAUD_RATE:.1%}: "
        f"{'as expected' if within_tolerance else 'UNEXPECTED, investigate'}"
    )

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

        print(
            f"  {file_path.name:<28} unique={unique_count:>9,}  duplicates={duplicate_count}"
        )
        if duplicate_count > 0:
            all_ok = False

    # How many transactions actually have an identity record. This drives
    # a design decision in Step 2, so it is worth knowing now.
    transaction_ids = set(
        pd.read_csv(TRAIN_TRANSACTION_FILE, usecols=[ID_COLUMN])[ID_COLUMN]
    )
    identity_ids = set(pd.read_csv(TRAIN_IDENTITY_FILE, usecols=[ID_COLUMN])[ID_COLUMN])

    overlap = len(transaction_ids & identity_ids)  # & is set intersection
    coverage = overlap / len(transaction_ids)

    print(
        f"\n  Transactions with an identity record: {overlap:,} "
        f"({coverage:.1%} of all transactions)"
    )
    print(
        "  The remaining transactions will have missing identity columns "
        "after the join. That is expected."
    )

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
