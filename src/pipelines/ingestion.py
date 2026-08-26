"""
Ingestion stage: load the raw CSV tables, join them, and save as Parquet.

Input:  data/raw/{split}_transaction.csv
        data/raw/{split}_identity.csv
Output: data/interim/{split}_joined.parquet

Run with:
    python run.py --step ingestion
"""

from __future__ import annotations

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
