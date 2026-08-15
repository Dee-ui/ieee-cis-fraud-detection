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
