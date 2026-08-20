"""
Deciding which columns to keep.

Three jobs:
  1. Find columns with no information at all
  2. Find columns where one value dominates, and rescue the ones whose
     rare values are strongly linked to fraud
  3. Reduce the 339 V columns using their block structure

Everything here runs on TRAINING ROWS ONLY. Deciding what to keep is a
form of learning from the data, so it obeys the same rule as every other
fitted step.
"""

from __future__ import annotations

import pandas as pd


def top_value_share(series: pd.Series) -> tuple[object, float]:
    """
    Find the most common value in a column and what share of rows it covers.

    dropna=False makes blank count as a value. That is deliberate: a column
    that is 99% blank is just as uninformative as one that is 99% zero, and
    we want the same rule to catch both.

    Returns the value and its share, as a number between 0 and 1.
    """
    if len(series) == 0:
        return None, 1.0

    counts = series.value_counts(dropna=False, normalize=True)
    if counts.empty:
        return None, 1.0

    return counts.index[0], float(counts.iloc[0])


def find_constant_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    Find columns holding a single distinct value, or nothing at all.

    nunique(dropna=True) ignores blanks, so a column that is entirely blank
    returns 0 and a column holding one repeated value returns 1. Both are
    useless: there is nothing for the model to split on.
    """
    constant = []
    for column in columns:
        if frame[column].nunique(dropna=True) <= 1:
            constant.append(column)
    return constant


def assess_near_constant_columns(
    frame: pd.DataFrame,
    target: pd.Series,
    columns: list[str],
    threshold: float,
    min_rare_rows: int,
    min_fraud_lift: float,
) -> pd.DataFrame:
    """
    Judge every column against the near-constant rule, with a rescue check.

    For each column we find the dominant value and its share. If the share
    is below the threshold, the column is kept and nothing more happens.

    If the share is at or above the threshold, we look at the rows that do
    NOT hold the dominant value and measure the fraud rate among them. A
    column is rescued when there are enough of those rows and their fraud
    rate is far from the overall rate, in either direction. A rare value
    strongly linked to fraud is a signal. A rare value strongly linked to
    safety is also a signal.

    Returns one row per column with the full evidence, so that every
    decision can be inspected afterwards rather than taken on trust.
    """
    base_rate = float(target.mean())
    records = []

    for column in columns:
        series = frame[column]
        dominant_value, share = top_value_share(series)

        if share < threshold:
            records.append(
                {
                    "column": column,
                    "dominant_value": str(dominant_value),
                    "dominant_share": round(share, 5),
                    "rare_rows": int(len(series) - round(share * len(series))),
                    "rare_fraud_rate": None,
                    "fraud_lift": None,
                    "decision": "keep",
                    "reason": "not near-constant",
                }
            )
            continue

        # Rows that do not hold the dominant value. Note that when the
        # dominant value is a real value, blank rows count as "rare",
        # because a comparison against blank is always False in pandas.
        if pd.isna(dominant_value):
            rare_mask = series.notna()
        else:
            rare_mask = series != dominant_value

        rare_rows = int(rare_mask.sum())

        if rare_rows < min_rare_rows:
            records.append(
                {
                    "column": column,
                    "dominant_value": str(dominant_value),
                    "dominant_share": round(share, 5),
                    "rare_rows": rare_rows,
                    "rare_fraud_rate": None,
                    "fraud_lift": None,
                    "decision": "drop",
                    "reason": f"near-constant, only {rare_rows} rare rows",
                }
            )
            continue

        rare_fraud_rate = float(target[rare_mask].mean())
        lift = rare_fraud_rate / base_rate if base_rate else 0.0

        # Rescue when the rare group is much riskier OR much safer than
        # average. Both directions are useful information.
        rescued = (lift >= min_fraud_lift) or (lift <= 1 / min_fraud_lift)

        records.append(
            {
                "column": column,
                "dominant_value": str(dominant_value),
                "dominant_share": round(share, 5),
                "rare_rows": rare_rows,
                "rare_fraud_rate": round(rare_fraud_rate, 5),
                "fraud_lift": round(lift, 3),
                "decision": "keep" if rescued else "drop",
                "reason": (
                    "rescued: rare values linked to fraud"
                    if rescued
                    else "near-constant, rare values unremarkable"
                ),
            }
        )

    return pd.DataFrame(records)


def load_v_groups(path) -> list[list[str]]:
    """
    Read the V block structure written by the Step 2 EDA stage.

    The CSV stores each block's members as one comma-separated string, so
    we split it back into a list and strip the spaces.
    """
    table = pd.read_csv(path)
    groups = []
    for _, row in table.iterrows():
        members = [name.strip() for name in str(row["columns"]).split(",")]
        groups.append([name for name in members if name])
    return groups


def cluster_by_correlation(
    correlations: pd.DataFrame, threshold: float
) -> list[list[str]]:
    """
    Group columns that move together, using a simple greedy pass.

    Take the first unassigned column and open a cluster with it. Pull in
    every other unassigned column whose absolute correlation with it
    reaches the threshold. Close the cluster, move to the next unassigned
    column, repeat.

    We use the ABSOLUTE correlation because a column that moves exactly
    opposite to another carries the same information, just with the sign
    flipped, and a tree can flip a sign for free.

    This is not the mathematically best clustering available. A full
    hierarchical clustering would group slightly better. Greedy is used
    because it always gives the same answer, it runs in seconds, and you
    can read it and know exactly what it did.
    """
    columns = list(correlations.columns)
    assigned: set[str] = set()
    clusters: list[list[str]] = []

    for column in columns:
        if column in assigned:
            continue

        cluster = [column]
        assigned.add(column)

        for other in columns:
            if other in assigned:
                continue
            value = correlations.loc[column, other]
            # A correlation is blank when one of the columns never varies
            # over the rows being compared. Blank is not a match.
            if pd.notna(value) and abs(value) >= threshold:
                cluster.append(other)
                assigned.add(other)

        clusters.append(cluster)

    return clusters


def choose_representative(frame: pd.DataFrame, cluster: list[str]) -> str:
    """
    Pick one column to stand for a cluster of near-duplicates.

    We keep the one with the most distinct values. More distinct values
    means finer resolution, which means more places a tree can split, which
    means more of the original information survives.
    """
    if len(cluster) == 1:
        return cluster[0]

    distinct_counts = {column: frame[column].nunique() for column in cluster}
    return max(distinct_counts, key=lambda column: distinct_counts[column])


def reduce_v_columns(
    frame: pd.DataFrame,
    groups: list[list[str]],
    threshold: float,
    verbose: bool = True,
) -> tuple[list[str], pd.DataFrame]:
    """
    Cut the V columns down using their block structure.

    For each block:
      1. Keep only the rows where the block is actually present. Every
         column in a block goes blank on the same rows, so dropna gives
         exactly those rows. Correlating rows that are entirely blank
         would tell us nothing.
      2. Correlate every column against every other column in the block.
      3. Group them greedily.
      4. Keep the column with the most distinct values from each group.

    Returns the surviving column names and a table recording which columns
    each survivor stands in for.
    """
    kept: list[str] = []
    records = []

    for group_index, group in enumerate(groups, start=1):
        available = [column for column in group if column in frame.columns]

        if not available:
            continue

        if len(available) == 1:
            kept.append(available[0])
            records.append(
                {
                    "block": group_index,
                    "cluster": 1,
                    "kept_column": available[0],
                    "n_represented": 1,
                    "represents": available[0],
                }
            )
            continue

        present_rows = frame[available].dropna()

        # If a block is blank on every training row there is nothing to
        # correlate, so keep the first column and move on.
        if len(present_rows) < 2:
            kept.append(available[0])
            records.append(
                {
                    "block": group_index,
                    "cluster": 1,
                    "kept_column": available[0],
                    "n_represented": len(available),
                    "represents": ", ".join(available),
                }
            )
            continue

        correlations = present_rows.corr()
        clusters = cluster_by_correlation(correlations, threshold)

        for cluster_index, cluster in enumerate(clusters, start=1):
            representative = choose_representative(present_rows, cluster)
            kept.append(representative)
            records.append(
                {
                    "block": group_index,
                    "cluster": cluster_index,
                    "kept_column": representative,
                    "n_represented": len(cluster),
                    "represents": ", ".join(cluster),
                }
            )

        if verbose:
            print(
                f"    block {group_index:>2}: {len(available):>3} columns -> "
                f"{len(clusters):>3} kept"
            )

    detail = pd.DataFrame(records)
    return kept, detail
