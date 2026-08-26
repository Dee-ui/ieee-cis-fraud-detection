"""
Tests for the feature engineer.

Two of these guard things that would otherwise fail silently in production:
the joblib round-trip, because Step 6 loads the transformer inside a
container, and row independence, because the API scores one transaction at
a time.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from config.config import TARGET_COLUMN, TIME_COLUMN, UNSEEN_CATEGORY_CODE


def test_transform_produces_the_fitted_feature_list(fitted_engineer, synthetic_joined):
    """Output columns must match the list fixed during fit, in the same order."""
    output = fitted_engineer.transform(synthetic_joined)

    assert list(output.columns) == fitted_engineer.feature_names_
    assert len(output) == len(synthetic_joined)


def test_target_and_passthrough_never_become_features(fitted_engineer):
    """
    The answer must not be in the features, and neither must the time column.

    TransactionDT is excluded because test values sit entirely above training
    values, so a tree cannot split on it usefully. That is decision D-26.
    """
    names = fitted_engineer.feature_names_

    assert TARGET_COLUMN not in names
    assert TIME_COLUMN not in names
    assert "TransactionID" not in names
    assert "uid" not in names  # grouping only, never a feature. D-29.


def test_joblib_round_trip_changes_nothing(fitted_engineer, synthetic_joined, tmp_path):
    """
    Saving and reloading the transformer must not change a single value.

    Step 6 loads this file inside a container. If the round trip altered
    anything, production predictions would differ from training predictions
    and nothing would error. This is the test that makes deployment safe.

    tmp_path is a pytest fixture giving a fresh temporary folder that is
    cleaned up afterwards, so the test leaves nothing behind.
    """
    before = fitted_engineer.transform(synthetic_joined)

    path = tmp_path / "engineer.joblib"
    joblib.dump(fitted_engineer, path)
    reloaded = joblib.load(path)

    after = reloaded.transform(synthetic_joined)

    assert list(after.columns) == list(before.columns)
    pd.testing.assert_frame_equal(before, after)


def test_transform_is_row_independent(fitted_engineer, synthetic_joined):
    """
    Transforming one row must give the same answer as transforming a batch.

    This is the leakage guard that matters most. Every transformation in the
    engineer is either row-wise or a lookup into state stored during fit, so
    a row's features cannot depend on which other rows travel with it.

    If someone later adds a groupby inside transform, thinking it harmless,
    this test fails immediately. It is also exactly the property the Step 6
    API depends on, because it scores one transaction at a time.
    """
    batch = fitted_engineer.transform(synthetic_joined)

    for position in (0, 17, len(synthetic_joined) - 1):
        single = fitted_engineer.transform(synthetic_joined.iloc[[position]])

        assert list(single.columns) == list(batch.columns)
        np.testing.assert_allclose(
            single.to_numpy(dtype="float64"),
            batch.iloc[[position]].to_numpy(dtype="float64"),
            rtol=1e-9,
            equal_nan=True,
        )


def test_unseen_category_maps_to_the_reserved_code(fitted_engineer, synthetic_joined):
    """
    A value the transformer never saw during fit must map to -1.

    This proves two things at once: that the mapping is fixed at fit time
    rather than rebuilt on each call, and that a new value at prediction time
    degrades gracefully instead of raising.
    """
    row = synthetic_joined.iloc[[0]].copy()
    row["ProductCD"] = pd.Series(
        ["A_PRODUCT_THAT_NEVER_EXISTED"], dtype="object", index=row.index
    )

    output = fitted_engineer.transform(row)

    assert output["ProductCD_code"].iloc[0] == UNSEEN_CATEGORY_CODE


def test_unseen_value_gets_zero_frequency(fitted_engineer, synthetic_joined):
    """
    A card number never seen in training must get a frequency of zero.

    Zero is the truthful answer: as far as the training data knows, this
    value does not exist. It is also the behaviour that makes uid_freq
    collapse on the real test set, which is why Section 8.3 exists.
    """
    row = synthetic_joined.iloc[[0]].copy()
    row["card1"] = 999_999

    output = fitted_engineer.transform(row)

    assert output["card1_freq"].iloc[0] == pytest.approx(0.0)


def test_fit_requires_the_target(synthetic_joined, synthetic_v_groups):
    """
    Fitting without labels must fail loudly.

    The near-constant rescue rule compares fraud rates, so it cannot run
    without the target. Failing clearly is better than silently skipping
    the rescue and quietly dropping useful columns.
    """
    from src.features.engineer import FraudFeatureEngineer

    engineer = FraudFeatureEngineer(v_groups=synthetic_v_groups, verbose=False)

    with pytest.raises(ValueError):
        engineer.fit(synthetic_joined)
