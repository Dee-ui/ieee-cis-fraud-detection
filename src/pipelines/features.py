"""
Feature engineering stage.

Input:  data/interim/train_joined.parquet
        data/interim/test_joined.parquet
        reports/v_column_missing_groups.csv
Output: data/processed/train_features.parquet
        data/processed/test_features.parquet
        models/feature_engineer.joblib
        reports/feature_manifest.csv
        reports/dropped_columns.csv
        reports/v_column_reduction.csv
        reports/feature_summary.md

Run with:
    python run.py --step features
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from config.config import (
    DROPPED_COLUMNS_FILE,
    FEATURE_MANIFEST_FILE,
    FEATURE_SUMMARY_FILE,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    ID_COLUMN,
    JOINED_TEST_FILE,
    JOINED_TRAIN_FILE,
    PREPROCESSOR_FILE,
    REFERENCE_DATETIME,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_SPLIT_LABEL,
    V_GROUPS_FILE,
    V_REDUCTION_FILE,
    VALID_SPLIT_LABEL,
    VALIDATION_FRACTION,
    ensure_directories,
)
from src.features.engineer import FraudFeatureEngineer
from src.utils.column_selection import load_v_groups
from src.utils.memory_utils import memory_usage_mb


def _as_date(seconds: float) -> str:
    """Turn a TransactionDT value into a readable date, for reporting only."""
    reference = pd.Timestamp(REFERENCE_DATETIME)
    return (reference + pd.to_timedelta(int(seconds), unit="s")).date().isoformat()


def _assign_split(frame: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    Label each row train or valid, cutting on time rather than at random.

    The boundary is the TransactionDT value at the 80th percentile. Rows at
    or below it are training, rows above it are validation.

    A random split would put January and June transactions on both sides,
    which lets the model see the future while learning. This dataset makes
    that leak especially bad: the same card appears many times over the six
    months, the D columns measure elapsed time from earlier events, and
    fraud arrives in bursts that a shuffle would scatter across both sides.
    """
    boundary = float(frame[TIME_COLUMN].quantile(1 - VALIDATION_FRACTION))
    labels = np.where(
        frame[TIME_COLUMN] <= boundary, TRAIN_SPLIT_LABEL, VALID_SPLIT_LABEL
    )
    return pd.Series(labels, index=frame.index, dtype="object"), boundary


def _verify(train_features: pd.DataFrame, test_features: pd.DataFrame) -> list[str]:
    """
    Four checks that catch the mistakes which do not raise an error.

    Each of these has silently ruined a model somewhere. None of them shows
    up as a crash, which is exactly why they are worth checking explicitly.
    """
    problems = []

    # 1. The answer must not be sitting in the features.
    if TARGET_COLUMN in train_features.columns:
        problems.append(f"{TARGET_COLUMN} is in the feature table")

    # 2. Train and test must have identical columns in identical order.
    #    Models read positions, not names.
    if list(train_features.columns) != list(test_features.columns):
        only_train = set(train_features.columns) - set(test_features.columns)
        only_test = set(test_features.columns) - set(train_features.columns)
        problems.append(
            f"feature columns differ. only in train: {sorted(only_train)[:5]}, "
            f"only in test: {sorted(only_test)[:5]}"
        )

    # 3. A column that is blank everywhere is dead weight.
    all_blank = [
        column
        for column in train_features.columns
        if train_features[column].isna().all()
    ]
    if all_blank:
        problems.append(
            f"{len(all_blank)} feature columns are entirely blank: {all_blank[:5]}"
        )

    # 4. Infinity breaks most models and is easy to create by dividing.
    numeric = train_features.select_dtypes(include=[np.number])
    infinite = numeric.columns[np.isinf(numeric.to_numpy()).any(axis=0)].tolist()
    if infinite:
        problems.append(f"{len(infinite)} columns contain infinity: {infinite[:5]}")

    return problems


def _write_summary(results: dict) -> None:
    """Write the human-readable summary of what this stage did."""
    lines: list[str] = []
    add = lines.append

    add("# Feature Engineering Summary")
    add("")
    add(
        "Generated automatically by `src/pipelines/features.py`. "
        "Do not edit by hand, it is overwritten on every run."
    )
    add("")

    add("## 1. Column reduction")
    add("")
    add("| Stage | Columns |")
    add("|-------|---------|")
    add(f"| Joined training table | {results['input_columns']} |")
    add(f"| Dropped: single value | {results['constant_dropped']} |")
    add(f"| Dropped: near-constant | {results['near_constant_dropped']} |")
    add(f"| Rescued from near-constant | {results['rescued']} |")
    add(f"| V columns before reduction | {results['v_before']} |")
    add(f"| V columns after reduction | {results['v_after']} |")
    add(f"| **Final feature count** | **{results['feature_count']}** |")
    add("")
    add(
        "Every dropped column, with the evidence behind the decision, is in "
        "`reports/dropped_columns.csv`. The V column mapping is in "
        "`reports/v_column_reduction.csv`."
    )
    add("")

    add("## 2. Feature types")
    add("")
    add(results["kind_counts"].to_markdown(index=False))
    add("")

    add("## 3. The time split")
    add("")
    add("| Portion | Rows | Frauds | Fraud rate | First | Last |")
    add("|---------|------|--------|------------|-------|------|")
    for row in results["split_rows"]:
        add(
            f"| {row['portion']} | {row['rows']:,} | {row['frauds']:,} | "
            f"{row['fraud_rate']:.4%} | {row['start']} | {row['end']} |"
        )
    add("")
    add(
        f"The boundary sits at TransactionDT {results['boundary']:,.0f}, "
        f"which is {results['boundary_date']}."
    )
    add("")
    add(
        "The transformer was fitted on the `train` portion only. The `valid` "
        "portion and the test set were transformed using what was learned "
        "there, and contributed nothing to it. Any frequency count or group "
        "average attached to a validation row was computed without that row."
    )
    add("")

    add("## 4. Test set")
    add("")
    add(f"- Rows: **{results['test_rows']:,}**")
    add(
        f"- Features: **{results['feature_count']}**, identical to training "
        "and in the same order"
    )
    add(
        f"- Values never seen during training, across all counted columns: "
        f"**{results['unseen_share']:.2%}** of lookups returned zero"
    )
    add("")

    add("## 5. Verification")
    add("")
    if results["problems"]:
        for problem in results["problems"]:
            add(f"- PROBLEM: {problem}")
    else:
        add("- The target is not present in the feature table")
        add("- Training and test features match exactly, in name and order")
        add("- No feature column is blank on every row")
        add("- No feature column contains infinity")
    add("")

    add("## 6. Carried into Step 4")
    add("")
    add(
        "1. Read the `split` column rather than recomputing the split, so "
        "every experiment is scored on exactly the same rows."
    )
    add(
        "2. `TransactionID` and `TransactionDT` are present in the files but "
        "are not features. Drop them before training."
    )
    add(
        "3. Load `models/feature_engineer.joblib` for scoring, never rebuild "
        "the transformations by hand."
    )
    add(
        "4. PR-AUC is primary, baseline 0.035. ROC-AUC secondary. Recall at a "
        "1% review rate is the business headline."
    )
    add("")

    FEATURE_SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {FEATURE_SUMMARY_FILE.name}")


def run_features() -> dict:
    """Run the whole feature engineering stage."""
    print("=" * 60)
    print("STAGE: FEATURE ENGINEERING")
    print("=" * 60)

    ensure_directories()

    if not JOINED_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"{JOINED_TRAIN_FILE} not found.\n"
            f"Run  python run.py --step ingestion  first."
        )
    if not V_GROUPS_FILE.exists():
        raise FileNotFoundError(
            f"{V_GROUPS_FILE} not found.\n" f"Run  python run.py --step eda  first."
        )

    # --- load and split ----------------------------------------------------
    print(f"\n  Loading {JOINED_TRAIN_FILE.name} ...")
    train = pd.read_parquet(JOINED_TRAIN_FILE)
    print(
        f"    {train.shape[0]:,} rows x {train.shape[1]} columns, "
        f"{memory_usage_mb(train):,.1f} MB"
    )

    split_labels, boundary = _assign_split(train)
    fit_mask = split_labels == TRAIN_SPLIT_LABEL

    print(f"\n  Time split at TransactionDT {boundary:,.0f} ({_as_date(boundary)})")
    print(f"    train portion: {int(fit_mask.sum()):,} rows")
    print(f"    valid portion: {int((~fit_mask).sum()):,} rows")

    # --- fit on the training portion only ----------------------------------
    print("\n  Fitting the feature engineer on the train portion only ...")
    engineer = FraudFeatureEngineer(v_groups=load_v_groups(V_GROUPS_FILE))
    engineer.fit(train.loc[fit_mask], train.loc[fit_mask, TARGET_COLUMN])

    # --- transform training -------------------------------------------------
    print("\n  Transforming the training table ...")
    train_features = engineer.transform(train)
    train_features[ID_COLUMN] = train[ID_COLUMN].to_numpy()
    train_features[TIME_COLUMN] = train[TIME_COLUMN].to_numpy()
    train_features[TARGET_COLUMN] = train[TARGET_COLUMN].to_numpy()
    train_features[SPLIT_COLUMN] = split_labels.to_numpy()
    print(
        f"    {train_features.shape[0]:,} rows x "
        f"{len(engineer.feature_names_)} features"
    )

    # Statistics needed for the report, gathered before the raw table goes.
    split_rows = []
    for portion in (TRAIN_SPLIT_LABEL, VALID_SPLIT_LABEL):
        mask = split_labels == portion
        subset_target = train.loc[mask, TARGET_COLUMN]
        split_rows.append(
            {
                "portion": portion,
                "rows": int(mask.sum()),
                "frauds": int(subset_target.sum()),
                "fraud_rate": float(subset_target.mean()),
                "start": _as_date(train.loc[mask, TIME_COLUMN].min()),
                "end": _as_date(train.loc[mask, TIME_COLUMN].max()),
            }
        )

    input_columns = train.shape[1]
    del train

    print(f"\n  Writing {FEATURES_TRAIN_FILE.name} ...")
    train_features.to_parquet(FEATURES_TRAIN_FILE, index=False, engine="pyarrow")
    print(f"    {FEATURES_TRAIN_FILE.stat().st_size / 1024 ** 2:,.1f} MB on disk")

    # --- transform test -----------------------------------------------------
    print(f"\n  Loading {JOINED_TEST_FILE.name} ...")
    test = pd.read_parquet(JOINED_TEST_FILE)
    print(f"    {test.shape[0]:,} rows x {test.shape[1]} columns")

    print("  Transforming the test table ...")
    test_features = engineer.transform(test)
    test_features[ID_COLUMN] = test[ID_COLUMN].to_numpy()
    test_features[TIME_COLUMN] = test[TIME_COLUMN].to_numpy()

    test_rows = len(test)
    del test

    # How much of the test set holds values that training never saw. A high
    # number here is an early warning that the two periods differ, which
    # matters for the drift work in Step 5.
    frequency_columns = [
        f"{column}_freq"
        for column in engineer.frequency_maps_
        if f"{column}_freq" in test_features.columns
    ]
    if frequency_columns:
        unseen_share = float((test_features[frequency_columns] == 0).to_numpy().mean())
    else:
        unseen_share = 0.0
    print(f"    unseen-value lookups in test: {unseen_share:.2%}")

    print(f"\n  Writing {FEATURES_TEST_FILE.name} ...")
    test_features.to_parquet(FEATURES_TEST_FILE, index=False, engine="pyarrow")
    print(f"    {FEATURES_TEST_FILE.stat().st_size / 1024 ** 2:,.1f} MB on disk")

    # --- verify --------------------------------------------------------------
    print("\n  Verifying ...")
    feature_only_train = train_features[engineer.feature_names_]
    feature_only_test = test_features[engineer.feature_names_]
    problems = _verify(feature_only_train, feature_only_test)

    if problems:
        for problem in problems:
            print(f"    PROBLEM: {problem}")
    else:
        print("    all four checks passed")

    # --- save the fitted transformer -------------------------------------------
    print(f"\n  Saving {PREPROCESSOR_FILE.name} ...")
    joblib.dump(engineer, PREPROCESSOR_FILE)
    print(f"    {PREPROCESSOR_FILE.stat().st_size / 1024 ** 2:,.1f} MB on disk")

    # --- reports -----------------------------------------------------------------
    manifest = engineer.manifest_.copy()
    manifest["missing_pct_train"] = [
        round(float(feature_only_train[name].isna().mean()) * 100, 2)
        for name in manifest["feature"]
    ]
    manifest["missing_pct_test"] = [
        round(float(feature_only_test[name].isna().mean()) * 100, 2)
        for name in manifest["feature"]
    ]
    manifest.to_csv(FEATURE_MANIFEST_FILE, index=False)
    print(f"  Wrote {FEATURE_MANIFEST_FILE.name}")

    engineer.column_decisions_.to_csv(DROPPED_COLUMNS_FILE, index=False)
    print(f"  Wrote {DROPPED_COLUMNS_FILE.name}")

    if not engineer.v_reduction_.empty:
        engineer.v_reduction_.to_csv(V_REDUCTION_FILE, index=False)
        print(f"  Wrote {V_REDUCTION_FILE.name}")

    decisions = engineer.column_decisions_
    v_after = (
        int(engineer.v_reduction_["kept_column"].nunique())
        if not engineer.v_reduction_.empty
        else 0
    )
    v_before = (
        int(engineer.v_reduction_["n_represented"].sum())
        if not engineer.v_reduction_.empty
        else 0
    )

    results = {
        "input_columns": input_columns,
        "constant_dropped": int((decisions["reason"] == "single distinct value").sum()),
        "near_constant_dropped": int(
            decisions["reason"].str.startswith("near-constant").sum()
        ),
        "rescued": int(decisions["reason"].str.startswith("rescued").sum()),
        "v_before": v_before,
        "v_after": v_after,
        "feature_count": len(engineer.feature_names_),
        "kind_counts": (
            manifest.groupby("kind", observed=True)
            .size()
            .reset_index(name="features")
            .sort_values("features", ascending=False)
        ),
        "split_rows": split_rows,
        "boundary": boundary,
        "boundary_date": _as_date(boundary),
        "test_rows": test_rows,
        "unseen_share": unseen_share,
        "problems": problems,
    }
    _write_summary(results)

    # --- headline ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING HEADLINES")
    print("=" * 60)
    print(f"  Input columns         : {input_columns}")
    print(f"  Final features        : {results['feature_count']}")
    print(f"  V columns             : {v_before} -> {v_after}")
    print(f"  Dropped, single value : {results['constant_dropped']}")
    print(f"  Dropped, near-constant: {results['near_constant_dropped']}")
    print(f"  Rescued               : {results['rescued']}")
    print(f"  Split boundary        : {results['boundary_date']}")
    print(
        f"  Train / valid rows    : {split_rows[0]['rows']:,} / {split_rows[1]['rows']:,}"
    )
    print(
        f"  Verification          : {'PASSED' if not problems else 'SEE PROBLEMS ABOVE'}"
    )
    print(f"\n  Full report: {FEATURE_SUMMARY_FILE}")

    return results
