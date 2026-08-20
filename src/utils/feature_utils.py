"""
Small, self-contained functions that build derived columns.

Nothing here learns anything from the data. Each function does the same
thing to any row it is given, so these can be applied to training,
validation, and test with no risk of leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import (
    AMOUNT_COLUMN,
    MISSING_LABEL,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    TIME_COLUMN,
    UID_ADDRESS_COLUMN,
    UID_CARD_COLUMN,
    UID_TIMEDELTA_COLUMN,
)


def as_label_series(series: pd.Series) -> pd.Series:
    """
    Turn any column into plain text labels, with blanks made explicit.

    Why this exists: frequency counting, grouping, and building the uid all
    need text. If each of them converted numbers to text its own way, train
    and test could end up producing different strings for the same value,
    for example "315" against "315.0", and every lookup would silently miss.

    Numbers go through float64 first, so the result does not depend on
    whether a column happened to be stored as int16 in one file and int32
    in another. Rounding to six decimal places removes float noise that
    would otherwise create near-duplicate labels.
    """
    if isinstance(series.dtype, pd.CategoricalDtype):
        values = series.astype("object")
    elif pd.api.types.is_numeric_dtype(series):
        values = series.astype("float64").round(6).astype("object")
    else:
        values = series.astype("object")

    labels = values.where(series.notna(), MISSING_LABEL)
    return labels.astype(str)


def combine_labels(*label_series: pd.Series) -> pd.Series:
    """Join several label columns into one, separated by underscores."""
    combined = label_series[0]
    for extra in label_series[1:]:
        combined = combined + "_" + extra
    return combined


def build_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Build hour of day and day of week from the seconds counter.

    Only cyclical features are produced. Hour runs 0 to 23 in the training
    period and 0 to 23 in the test period, so anything the model learns
    about hour still applies later. An absolute day counter would not:
    every test value sits above every training value, and a tree cannot
    split usefully outside the range it was trained on.

    The day-of-week numbers may not line up with real calendar names,
    because the reference date is a convention rather than a fact. That
    does not matter. What matters is that the same real weekday always
    produces the same number, and it does, because both come from the same
    continuous seconds counter.
    """
    seconds = frame[TIME_COLUMN].astype("int64")

    day_number = seconds // SECONDS_PER_DAY

    return pd.DataFrame(
        {
            "hour": ((seconds // SECONDS_PER_HOUR) % 24).astype("int8"),
            "day_of_week": (day_number % 7).astype("int8"),
        },
        index=frame.index,
    )


def build_amount_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Break the transaction amount into parts the model can use.

    amount_log
        Amounts are heavily skewed: most are small, a few are enormous.
        The logarithm compresses the tail so a handful of huge transactions
        do not dominate every split. log1p is log(1 + x), which behaves
        properly when the amount is zero.

    amount_cents
        The part after the decimal point. This is the feature that decision
        D-18 in Step 2 kept float64 precision for. Legitimate purchases
        cluster on particular cent values, currency conversions produce
        distinctive ones, and fraud distributes differently.

    amount_is_round
        Whether the amount has no cents at all.
    """
    amount = frame[AMOUNT_COLUMN].astype("float64")

    cents = (amount - np.floor(amount)).round(4)

    return pd.DataFrame(
        {
            "amount_log": np.log1p(amount).astype("float32"),
            "amount_cents": cents.astype("float32"),
            "amount_is_round": (cents == 0).astype("int8"),
        },
        index=frame.index,
    )


def split_email_domain(labels: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Split an email domain into a provider part and a suffix part.

    "gmail.com" becomes provider "gmail" and suffix "com".
    "yahoo.co.uk" becomes provider "yahoo" and suffix "co.uk".

    The point is that yahoo.com, yahoo.co.uk, and yahoo.com.mx are the same
    provider. Left as whole strings they are three unrelated labels and the
    model has to learn about each separately, with far fewer examples each.

    The blank label has no dot in it, so it survives the split intact and
    stays its own category on both sides.
    """
    parts = labels.str.split(".", n=1, expand=True)

    provider = parts[0]
    if parts.shape[1] > 1:
        suffix = parts[1].fillna(MISSING_LABEL)
    else:
        suffix = pd.Series(MISSING_LABEL, index=labels.index)

    return pd.DataFrame(
        {
            f"{prefix}_email_provider": provider,
            f"{prefix}_email_suffix": suffix,
        },
        index=labels.index,
    )


def build_screen_features(labels: pd.Series) -> pd.DataFrame:
    """
    Turn a screen resolution like "1920x1080" into three numbers.

    errors="coerce" means anything that does not parse becomes blank rather
    than raising. The blank label does not contain an "x", so it lands in
    the first part and fails to convert, which is exactly what we want.
    """
    parts = labels.str.split("x", n=1, expand=True)

    width = pd.to_numeric(parts[0], errors="coerce")
    if parts.shape[1] > 1:
        height = pd.to_numeric(parts[1], errors="coerce")
    else:
        height = pd.Series(np.nan, index=labels.index)

    return pd.DataFrame(
        {
            "screen_width": width.astype("float32"),
            "screen_height": height.astype("float32"),
            "screen_area": (width * height).astype("float32"),
        },
        index=labels.index,
    )


def first_token(labels: pd.Series) -> pd.Series:
    """
    Take the first word of a messy text column.

    "SAMSUNG SM-G892A Build/NRD90M" becomes "samsung".
    "chrome 62.0" becomes "chrome".

    This collapses hundreds of near-identical values into a handful of
    useful groups. Splitting on "/" as well catches strings where the first
    word already carries a build path.
    """
    token = labels.str.split(" ", n=1).str[0]
    token = token.str.split("/", n=1).str[0]
    return token.str.lower()


def build_match_features(frame: pd.DataFrame, match_columns: list[str]) -> pd.DataFrame:
    """
    Summarise the M columns, which are agreement checks.

    M1 to M3 and M5 to M9 hold T or F. M4 is different, holding M0, M1, or
    M2, so it is left out of the "how many said T" count but still counted
    towards how many are blank.

    Two summaries are produced: how many checks passed, and how many were
    not available. The second matters because the M family averages just
    under 50% missing in this dataset, so the amount of missingness varies
    a lot from row to row.
    """
    available = [column for column in match_columns if column in frame.columns]
    if not available:
        return pd.DataFrame(index=frame.index)

    true_count = pd.Series(0, index=frame.index, dtype="int8")
    missing_count = pd.Series(0, index=frame.index, dtype="int8")

    for column in available:
        labels = as_label_series(frame[column])
        if column != "M4":
            true_count = true_count + (labels == "T").astype("int8")
        missing_count = missing_count + (labels == MISSING_LABEL).astype("int8")

    return pd.DataFrame(
        {"match_true_count": true_count, "match_missing_count": missing_count},
        index=frame.index,
    )


def build_uid(frame: pd.DataFrame) -> pd.Series:
    """
    Build a rough customer fingerprint.

    The dataset has no customer identifier. But D1 measures days since the
    card was first seen, so:

        day of this transaction  minus  D1  =  the day the card first appeared

    That subtraction gives the same answer for every transaction on the
    same card, whenever it happens. Combined with card1 and addr1 it is a
    reasonably stable fingerprint for one customer.

    Two warnings, both taken seriously elsewhere in the code.

    It is a guess, not a fact. Two customers can collide on one fingerprint,
    and one customer can split in two if their address changes.

    More importantly, it must never become a feature in its own right. A
    model given the fingerprint memorises individual customers, which looks
    excellent in validation, where the same customers appear on both sides
    of the split, and is worth nothing on customers it has never met. So
    the uid is used only for grouping and counting. That is decision D-29.
    """
    missing_columns = [
        column
        for column in (UID_CARD_COLUMN, UID_ADDRESS_COLUMN, UID_TIMEDELTA_COLUMN)
        if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(f"cannot build uid, missing columns: {missing_columns}")

    day_number = (frame[TIME_COLUMN].astype("int64") // SECONDS_PER_DAY)
    first_seen_day = day_number - frame[UID_TIMEDELTA_COLUMN].astype("float64")

    return combine_labels(
        as_label_series(frame[UID_CARD_COLUMN]),
        as_label_series(frame[UID_ADDRESS_COLUMN]),
        as_label_series(first_seen_day),
    )
