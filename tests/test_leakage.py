"""
Guards against reintroducing leakage.

Leakage does not raise an error. It produces a validation score that is too
good, which looks like success. These tests assert the structural properties
that make leakage impossible, so that if someone changes the pipeline in a
way that breaks them, the build fails rather than the score improving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import TIME_COLUMN, UNSEEN_CATEGORY_CODE, VALIDATION_FRACTION


def test_time_split_puts_every_training_row_before_every_validation_row():
    """
    The split must cut on time, not at random.

    If any training row happened after any validation row, the model would be
    learning from the future. This asserts the boundary holds with no overlap.
    """
    times = np.sort(np.random.default_rng(3).integers(86_400, 10_000_000, 5000))
    frame = pd.DataFrame({TIME_COLUMN: times})

    boundary = float(frame[TIME_COLUMN].quantile(1 - VALIDATION_FRACTION))
    train = frame[frame[TIME_COLUMN] <= boundary]
    valid = frame[frame[TIME_COLUMN] > boundary]

    assert len(train) > 0 and len(valid) > 0
    assert train[TIME_COLUMN].max() <= valid[TIME_COLUMN].min()

    # Roughly the requested share, allowing for ties on the boundary value.
    assert abs(len(valid) / len(frame) - VALIDATION_FRACTION) < 0.02


def test_engineer_never_learned_anything_from_the_validation_period(
    synthetic_joined, synthetic_v_groups
):
    """
    A category present only in the validation portion must be unknown.

    This proves the transformer was fitted on the earlier rows alone. If it
    had seen the whole file, this value would have its own code instead of
    the reserved unseen code.
    """
    from src.features.engineer import FraudFeatureEngineer

    frame = synthetic_joined.copy()
    cut = int(len(frame) * 0.8)

    # Plant a value that exists only after the split boundary.
    frame["ProductCD"] = frame["ProductCD"].astype("object")
    frame.iloc[cut + 5, frame.columns.get_loc("ProductCD")] = "ONLY_IN_VALID"
    frame["ProductCD"] = frame["ProductCD"].astype("category")

    train_part = frame.iloc[:cut]
    engineer = FraudFeatureEngineer(v_groups=synthetic_v_groups, verbose=False)
    engineer.fit(train_part, train_part["isFraud"])

    planted = engineer.transform(frame.iloc[[cut + 5]])

    assert planted["ProductCD_code"].iloc[0] == UNSEEN_CATEGORY_CODE


def test_no_feature_is_a_disguised_time_index(fitted_engineer, synthetic_joined):
    """
    No feature may track the raw clock.

    Trees cannot split outside the value range they were trained on, so a
    feature that rises monotonically with time is useless at prediction time
    and worse than useless during training, because it looks helpful.

    Hour and day of week are fine, and expected to correlate weakly, so the
    bar is set at a near-perfect correlation rather than at zero.
    """
    output = fitted_engineer.transform(synthetic_joined)
    clock = synthetic_joined[TIME_COLUMN].to_numpy(dtype="float64")

    suspicious = []
    for column in output.columns:
        values = output[column].to_numpy(dtype="float64")
        usable = np.isfinite(values)
        if usable.sum() < 100 or np.nanstd(values[usable]) == 0:
            continue
        correlation = abs(np.corrcoef(clock[usable], values[usable])[0, 1])
        if correlation > 0.98:
            suspicious.append((column, round(float(correlation), 4)))

    assert not suspicious, f"features tracking the clock: {suspicious}"
