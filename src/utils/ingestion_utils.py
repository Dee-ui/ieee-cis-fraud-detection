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
