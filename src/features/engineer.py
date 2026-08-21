"""
FraudFeatureEngineer: the fitted transformer that turns the joined table
into a model-ready feature table.

Why this is a class and not a script
------------------------------------
Some transformations need to know things that can only be worked out from
training data. How often card1 = 16075 appeared. What number gmail.com maps
to. What the average transaction amount is for a given card.

In Step 6 a single transaction arrives at a web service and has to be
scored. None of those facts can be worked out from one row. They have to
have been saved.

So: fit() learns them and stores them on the object. transform() applies
what was stored. joblib.dump() writes the whole object to disk. Step 4 and
Step 6 load it and get transformations identical to training.

The class inherits from scikit-learn's BaseEstimator and TransformerMixin,
which costs almost nothing and means the object drops straight into a
scikit-learn Pipeline alongside a model.

The leakage rule
----------------
fit() must only ever see training rows. The pipeline stage in
src/pipelines/features.py enforces that by splitting on time first and
passing only the earlier portion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config.config import (
    AGGREGATION_SPECS,
    FREQUENCY_ENCODE_COLUMNS,
    MISSING_LABEL,
    M_COLUMNS,
    NEAR_CONSTANT_THRESHOLD,
    PASSTHROUGH_COLUMNS,
    RESCUE_MIN_FRAUD_LIFT,
    RESCUE_MIN_RARE_ROWS,
    UNSEEN_CATEGORY_CODE,
    V_CORRELATION_THRESHOLD,
)
from src.utils.column_selection import (
    assess_near_constant_columns,
    find_constant_columns,
    reduce_v_columns,
)
from src.utils.feature_utils import (
    as_label_series,
    build_amount_features,
    build_match_features,
    build_screen_features,
    build_time_features,
    build_uid,
    combine_labels,
    first_token,
    split_email_domain,
)

# Derived text columns that are built inside this class rather than read
# from the raw data. They are treated as categories from then on.
DERIVED_LABEL_COLUMNS = [
    "P_email_provider",
    "P_email_suffix",
    "R_email_provider",
    "R_email_suffix",
    "device_brand",
    "browser_family",
    "card1_addr1",
]

# Built and used, but never handed to the model on its own. See D-29.
GROUPING_ONLY_COLUMNS = ["uid"]


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Learns feature transformations from training data and applies them."""

    def __init__(
        self,
        v_groups: list[list[str]] | None = None,
        near_constant_threshold: float = NEAR_CONSTANT_THRESHOLD,
        rescue_min_rare_rows: int = RESCUE_MIN_RARE_ROWS,
        rescue_min_fraud_lift: float = RESCUE_MIN_FRAUD_LIFT,
        v_correlation_threshold: float = V_CORRELATION_THRESHOLD,
        verbose: bool = True,
    ):
        # scikit-learn requires that __init__ only stores its arguments and
        # does no work. Anything computed here would be lost when the object
        # is cloned, which Pipeline and cross-validation both do.
        self.v_groups = v_groups
        self.near_constant_threshold = near_constant_threshold
        self.rescue_min_rare_rows = rescue_min_rare_rows
        self.rescue_min_fraud_lift = rescue_min_fraud_lift
        self.v_correlation_threshold = v_correlation_threshold
        self.verbose = verbose

    # -----------------------------------------------------
    # Small internal helpers
    # -----------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _tag(self, name: str, kind: str, source: str) -> None:
        """
        Record what a feature is and where it came from.

        Only runs while fitting. The result becomes reports/feature_manifest.csv,
        which is what lets you answer "where did this column come from" in
        four months without rereading the code.
        """
        if getattr(self, "_recording", False):
            self.feature_tags_.append({"feature": name, "kind": kind, "source": source})

    # -----------------------------------------------------
    # Building the label columns (text versions used for counting)
    # -----------------------------------------------------

    def _build_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Build the text version of every column that needs counting or grouping.

        These are intermediate working columns. Most of them never reach the
        final feature table directly; they get turned into codes and counts
        first.
        """
        labels = pd.DataFrame(index=frame.index)

        # Raw columns that survived pruning and are either text or an
        # identifier-style number such as card1.
        for column in self.label_source_columns_:
            if column in frame.columns:
                labels[column] = as_label_series(frame[column])
            else:
                labels[column] = MISSING_LABEL

        # Email domains split into provider and suffix.
        for prefix, column in (("P", "P_emaildomain"), ("R", "R_emaildomain")):
            if column in labels.columns:
                parts = split_email_domain(labels[column], prefix)
                labels[parts.columns] = parts
            else:
                labels[f"{prefix}_email_provider"] = MISSING_LABEL
                labels[f"{prefix}_email_suffix"] = MISSING_LABEL

        # Device brand and browser family, from the first word of each.
        labels["device_brand"] = (
            first_token(labels["DeviceInfo"])
            if "DeviceInfo" in labels.columns
            else MISSING_LABEL
        )
        labels["browser_family"] = (
            first_token(labels["id_31"])
            if "id_31" in labels.columns
            else MISSING_LABEL
        )

        # A combined card and address key. Two customers can share a card
        # code, and two can share an address, but sharing both is rarer, so
        # the pair is a sharper grouping than either alone.
        if "card1" in labels.columns and "addr1" in labels.columns:
            labels["card1_addr1"] = combine_labels(labels["card1"], labels["addr1"])
        else:
            labels["card1_addr1"] = MISSING_LABEL

        # The customer fingerprint. Grouping only.
        if self.can_build_uid_:
            labels["uid"] = build_uid(frame)
        else:
            labels["uid"] = MISSING_LABEL

        return labels

    # -----------------------------------------------------
    # fit
    # -----------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """
        Learn everything that has to be remembered.

        X must be TRAINING ROWS ONLY. y is the fraud label for those rows,
        and is required, because the rescue rule in the pruning step needs
        to compare fraud rates.
        """
        if y is None:
            raise ValueError(
                "fit needs the target. The near-constant rescue rule compares "
                "fraud rates, so it cannot run without labels."
            )

        y = pd.Series(y).reset_index(drop=True)
        frame = X.reset_index(drop=True)

        self._log("  Selecting columns ...")

        # --- 1. work out which columns are even candidates ---------------
        candidates = [
            column for column in frame.columns if column not in PASSTHROUGH_COLUMNS
        ]

        # --- 2. drop columns with a single value -------------------------
        constant_columns = find_constant_columns(frame, candidates)
        self._log(f"    {len(constant_columns)} columns hold a single value")

        remaining = [c for c in candidates if c not in constant_columns]

        # --- 3. near-constant rule, with the rescue check ----------------
        assessment = assess_near_constant_columns(
            frame,
            y,
            remaining,
            threshold=self.near_constant_threshold,
            min_rare_rows=self.rescue_min_rare_rows,
            min_fraud_lift=self.rescue_min_fraud_lift,
        )

        near_constant_dropped = assessment.loc[
            assessment["decision"] == "drop", "column"
        ].tolist()
        rescued = assessment.loc[
            assessment["reason"].str.startswith("rescued"), "column"
        ].tolist()

        self._log(
            f"    {len(near_constant_dropped)} columns dropped as near-constant, "
            f"{len(rescued)} rescued because their rare values track fraud"
        )

        survivors = [c for c in remaining if c not in near_constant_dropped]

        # Build the constant-column records as a list of dictionaries and only
        # turn them into a table when there is something to put in it.
        #
        # Building an empty DataFrame from an empty list produces a table with
        # no rows and no usable column types. Concatenating that is deprecated
        # in pandas, and in a future release it will silently change the types
        # of the result rather than warning. Skipping the concat entirely when
        # there is nothing to add avoids the problem instead of suppressing it.
        constant_records = [
            {
                "column": column,
                "dominant_value": None,
                "dominant_share": 1.0,
                "rare_rows": 0,
                "rare_fraud_rate": None,
                "fraud_lift": None,
                "decision": "drop",
                "reason": "single distinct value",
            }
            for column in constant_columns
        ]

        if constant_records:
            self.column_decisions_ = pd.concat(
                [pd.DataFrame(constant_records), assessment], ignore_index=True
            )
        else:
            self.column_decisions_ = assessment.reset_index(drop=True)

        # --- 4. reduce the V columns using their blocks ------------------
        survivor_set = set(survivors)
        v_survivors = [
            column
            for group in (self.v_groups or [])
            for column in group
            if column in survivor_set
        ]
        non_v_survivors = [c for c in survivors if c not in set(v_survivors)]

        if v_survivors and self.v_groups:
            self._log(f"  Reducing {len(v_survivors)} surviving V columns ...")
            groups_after_pruning = [
                [column for column in group if column in survivor_set]
                for group in self.v_groups
            ]
            v_kept, v_detail = reduce_v_columns(
                frame,
                [group for group in groups_after_pruning if group],
                threshold=self.v_correlation_threshold,
                verbose=self.verbose,
            )
            self._log(f"    {len(v_survivors)} V columns -> {len(v_kept)} kept")
        else:
            v_kept, v_detail = [], pd.DataFrame()

        self.v_reduction_ = v_detail
        self.base_columns_ = non_v_survivors + v_kept

        # --- 5. work out which columns need text handling ----------------
        # Anything stored as a category, plus the identifier-style numbers
        # named in the frequency list, plus the columns the uid needs.
        category_like = [
            column
            for column in self.base_columns_
            if isinstance(frame[column].dtype, pd.CategoricalDtype)
        ]
        frequency_sources = [
            column for column in FREQUENCY_ENCODE_COLUMNS if column in self.base_columns_
        ]
        uid_sources = [
            column
            for column in ("card1", "addr1", "D1", "id_31", "DeviceInfo")
            if column in self.base_columns_
        ]

        self.label_source_columns_ = sorted(
            set(category_like) | set(frequency_sources) | set(uid_sources)
        )
        self.category_columns_ = category_like
        self.can_build_uid_ = all(
            column in self.base_columns_ for column in ("card1", "addr1", "D1")
        )
        if not self.can_build_uid_:
            self._log("    WARNING: uid cannot be built, a source column was pruned")

        # --- 6. learn the encodings --------------------------------------
        self._log("  Learning encodings ...")
        labels = self._build_labels(frame)

        # Frequency maps: the share of training rows holding each value.
        # A share rather than a raw count, so that a training set of 472,000
        # rows and a test set of 506,000 rows produce comparable numbers.
        self.frequency_maps_ = {}
        for column in FREQUENCY_ENCODE_COLUMNS:
            if column in labels.columns:
                self.frequency_maps_[column] = labels[column].value_counts(
                    normalize=True
                )
        self._log(f"    {len(self.frequency_maps_)} frequency maps")

        # Category maps: every text column gets an integer code. The blank
        # label gets a code like any other value, because "we do not know"
        # is real information. Values never seen in training get -1 at
        # transform time.
        self.category_maps_ = {}
        for column in self.category_columns_ + DERIVED_LABEL_COLUMNS:
            if column in labels.columns:
                distinct = sorted(labels[column].dropna().unique())
                self.category_maps_[column] = {
                    value: code for code, value in enumerate(distinct)
                }
        self._log(f"    {len(self.category_maps_)} category maps")

        # Aggregate maps: the average and spread of a value within each group.
        self.aggregate_maps_ = {}
        for group_column, value_column in AGGREGATION_SPECS:
            if group_column not in labels.columns:
                continue
            if value_column not in frame.columns:
                continue
            values = frame[value_column].astype("float64")
            summary = values.groupby(labels[group_column], observed=True).agg(
                ["mean", "std"]
            )
            self.aggregate_maps_[(group_column, value_column)] = summary
        self._log(f"    {len(self.aggregate_maps_)} aggregate maps")

        # --- 7. run one transform to fix the feature list ----------------
        # Running the real transform on the fit data is the only reliable
        # way to know exactly which columns come out and in what order.
        # Guessing the list separately is how the two drift apart.
        self.feature_tags_ = []
        self._recording = True
        produced = self._transform_frame(frame, labels)
        self._recording = False

        self.feature_names_ = list(produced.columns)
        self.manifest_ = pd.DataFrame(self.feature_tags_)

        self._log(f"  Fitted. {len(self.feature_names_)} features produced.")
        return self

    # -----------------------------------------------------
    # transform
    # -----------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply everything learned in fit. Safe on training, validation, or test."""
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("call fit before transform")

        frame = X.reset_index(drop=True)
        labels = self._build_labels(frame)
        produced = self._transform_frame(frame, labels)

        # Force the exact same columns in the exact same order as training.
        # A model does not read column names, it reads positions. A column
        # that arrives in a different place is silently the wrong number.
        return produced.reindex(columns=self.feature_names_)

    def _transform_frame(
        self, frame: pd.DataFrame, labels: pd.DataFrame
    ) -> pd.DataFrame:
        """Build the feature table. Shared by fit and transform, so they cannot differ."""
        pieces: list[pd.DataFrame] = []

        # --- numeric base columns, passed through as they are -------------
        numeric_base = [
            column
            for column in self.base_columns_
            if column in frame.columns and column not in self.category_columns_
        ]
        if numeric_base:
            pieces.append(frame[numeric_base])
            for column in numeric_base:
                self._tag(column, "base_numeric", column)

        # --- derived numeric features -------------------------------------
        time_features = build_time_features(frame)
        pieces.append(time_features)
        for column in time_features.columns:
            self._tag(column, "derived_time", "TransactionDT")

        amount_features = build_amount_features(frame)
        pieces.append(amount_features)
        for column in amount_features.columns:
            self._tag(column, "derived_amount", "TransactionAmt")

        match_features = build_match_features(frame, M_COLUMNS)
        if not match_features.empty:
            pieces.append(match_features)
            for column in match_features.columns:
                self._tag(column, "derived_match", "M1-M9")

        if "id_33" in labels.columns:
            screen_features = build_screen_features(labels["id_33"])
            pieces.append(screen_features)
            for column in screen_features.columns:
                self._tag(column, "derived_screen", "id_33")

        # Do the purchaser and recipient email domains match? A classic
        # fraud signal. Kept as three states: no, yes, and unknown.
        if "P_emaildomain" in labels.columns and "R_emaildomain" in labels.columns:
            both_known = (labels["P_emaildomain"] != MISSING_LABEL) & (
                labels["R_emaildomain"] != MISSING_LABEL
            )
            same = (labels["P_emaildomain"] == labels["R_emaildomain"]).astype("float32")
            pieces.append(
                pd.DataFrame(
                    {"email_domains_match": same.where(both_known, np.nan)},
                    index=frame.index,
                )
            )
            self._tag("email_domains_match", "derived_email", "P/R_emaildomain")

        # --- category codes -------------------------------------------------
        encoded = {}
        for column, mapping in self.category_maps_.items():
            if column not in labels.columns:
                continue
            name = f"{column}_code"
            encoded[name] = (
                labels[column]
                .map(mapping)
                .fillna(UNSEEN_CATEGORY_CODE)
                .astype("int32")
            )
            self._tag(name, "category_code", column)
        if encoded:
            pieces.append(pd.DataFrame(encoded, index=frame.index))

        # --- frequency counts -----------------------------------------------
        frequencies = {}
        for column, mapping in self.frequency_maps_.items():
            if column not in labels.columns:
                continue
            name = f"{column}_freq"
            # A value never seen in training gets 0. That is truthful:
            # as far as the training data knows, it does not exist.
            frequencies[name] = (
                labels[column].map(mapping).fillna(0.0).astype("float32")
            )
            self._tag(name, "frequency", column)
        if frequencies:
            pieces.append(pd.DataFrame(frequencies, index=frame.index))

        # --- group aggregates -------------------------------------------------
        aggregates = {}
        for (group_column, value_column), summary in self.aggregate_maps_.items():
            if group_column not in labels.columns or value_column not in frame.columns:
                continue

            keys = labels[group_column]
            values = frame[value_column].astype("float64")

            mean_name = f"{value_column}_mean_by_{group_column}"
            std_name = f"{value_column}_std_by_{group_column}"
            ratio_name = f"{value_column}_ratio_to_{group_column}_mean"

            group_mean = keys.map(summary["mean"])
            group_std = keys.map(summary["std"])

            aggregates[mean_name] = group_mean.astype("float32")
            aggregates[std_name] = group_std.astype("float32")

            # Dividing by zero gives infinity, which no model handles well.
            # Turn those into blanks so the tree routes them like any other
            # missing value.
            ratio = values / group_mean
            aggregates[ratio_name] = (
                ratio.replace([np.inf, -np.inf], np.nan).astype("float32")
            )

            for name in (mean_name, std_name, ratio_name):
                self._tag(name, "aggregate", f"{value_column} by {group_column}")

        if aggregates:
            pieces.append(pd.DataFrame(aggregates, index=frame.index))

        result = pd.concat(pieces, axis=1)

        # Guard against a duplicated column name, which would make the
        # feature list ambiguous and is easy to introduce by accident when
        # adding a new feature later.
        duplicated = result.columns[result.columns.duplicated()].tolist()
        if duplicated:
            raise ValueError(f"duplicate feature names produced: {duplicated}")

        return result
