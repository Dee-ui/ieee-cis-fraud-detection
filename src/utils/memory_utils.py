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
