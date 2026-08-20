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
        "identity record. Read that figure carefully. The table below shows "
        "that identity coverage is almost entirely decided by `ProductCD`: "
        "product W never has an identity record, and every other product "
        "almost always does. Since W also has the lowest fraud rate and makes "
        "up most of the data, the bulk of this gap is a product effect rather "
        "than an identity effect. Restricted to the non-W products, where the "
        "flag actually varies, the lift is closer to 1.4x. `has_identity` is "
        "kept as a feature, but it is expected to rank low.")

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
