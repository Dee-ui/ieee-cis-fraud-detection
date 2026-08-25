"""
Shared fixtures for the test suite.

Every test runs on synthetic data built here. The real dataset is 1.3 GB and
is not in the repository, so tests that needed it could not run in CI, and
tests that do not run automatically do not get run. That is decision D-50.

The synthetic frame mirrors the shape of the joined table: the same column
names, the same dtypes, the same mixture of numeric, text, and blank values.
It has none of the real data's signal, which is fine, because these tests
check that the machinery is correct rather than that the model is good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.config import SECONDS_PER_DAY

N_ROWS = 3000
SEED = 7


@pytest.fixture
def rng() -> np.random.Generator:
    """One seeded random generator, so every test run is identical."""
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_joined(rng) -> pd.DataFrame:
    """
    A small stand-in for data/interim/train_joined.parquet.

    Includes every column the feature engineer needs to build its derived
    features: the time and amount columns, the uid sources (card1, addr1,
    D1), text columns for the category and email handling, id_31 and id_33
    for the device features, M columns for the match features, and a couple
    of small V blocks so the V reduction has something to reduce.
    """
    n = N_ROWS

    # Time runs forward across 120 days so a time split has something to cut.
    time_seconds = np.sort(rng.integers(SECONDS_PER_DAY, SECONDS_PER_DAY * 120, n))

    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(1_000_000, 1_000_000 + n, dtype="int32"),
            "TransactionDT": time_seconds.astype("int32"),
            "isFraud": rng.binomial(1, 0.05, n).astype("int8"),
            "TransactionAmt": np.round(rng.gamma(2.0, 60.0, n), 2),
            "has_identity": rng.binomial(1, 0.25, n).astype("int8"),
            # Identifier-style numbers: many repeats, so frequency encoding
            # and group aggregates have something to work with.
            "card1": rng.integers(1000, 1100, n).astype("int16"),
            "card2": rng.integers(100, 160, n).astype("float32"),
            "addr1": rng.integers(200, 240, n).astype("float32"),
            "D1": rng.integers(0, 90, n).astype("float32"),
            "D15": rng.integers(0, 200, n).astype("float32"),
            "C1": rng.integers(0, 20, n).astype("float32"),
            "dist1": rng.integers(0, 500, n).astype("float32"),
        }
    )

    # Text columns, stored as category exactly as the real pipeline does.
    frame["ProductCD"] = pd.Series(
        rng.choice(["W", "C", "R", "H", "S"], n), dtype="category"
    )
    frame["card4"] = pd.Series(
        rng.choice(["visa", "mastercard", "discover"], n), dtype="category"
    )
    frame["card6"] = pd.Series(rng.choice(["credit", "debit"], n), dtype="category")
    frame["P_emaildomain"] = pd.Series(
        rng.choice(["gmail.com", "yahoo.com", "hotmail.co.uk"], n), dtype="category"
    )
    frame["R_emaildomain"] = pd.Series(
        rng.choice(["gmail.com", "aol.com"], n), dtype="category"
    )
    frame["DeviceInfo"] = pd.Series(
        rng.choice(["SAMSUNG SM-G892A Build/NRD90M", "Windows", "iOS Device"], n),
        dtype="category",
    )
    frame["id_31"] = pd.Series(
        rng.choice(["chrome 62.0", "chrome 63.0", "safari generic"], n),
        dtype="category",
    )
    frame["id_33"] = pd.Series(
        rng.choice(["1920x1080", "1334x750", "2208x1242"], n), dtype="category"
    )
    frame["M1"] = pd.Series(rng.choice(["T", "F"], n), dtype="category")
    frame["M4"] = pd.Series(rng.choice(["M0", "M1", "M2"], n), dtype="category")

    # Two small V blocks. Inside each block the columns are correlated, so
    # the correlation clustering has near-duplicates to collapse.
    base_a = rng.normal(0, 1, n)
    base_b = rng.normal(0, 1, n)
    for index in range(1, 4):
        frame[f"V{index}"] = (base_a + rng.normal(0, 0.05, n)).astype("float32")
    for index in range(4, 7):
        frame[f"V{index}"] = (base_b + rng.normal(0, 0.05, n)).astype("float32")

    # Blanks in the same block-wise pattern the real V columns show.
    blank_a = rng.random(n) < 0.20
    frame.loc[blank_a, ["V1", "V2", "V3"]] = np.nan

    return frame


@pytest.fixture
def synthetic_v_groups() -> list[list[str]]:
    """The V block structure for the synthetic frame."""
    return [["V1", "V2", "V3"], ["V4", "V5", "V6"]]


@pytest.fixture
def fitted_engineer(synthetic_joined, synthetic_v_groups):
    """
    A feature engineer fitted on the first 80% of the synthetic frame.

    Fitted on the earlier portion only, exactly as the real pipeline does,
    so the leakage tests have something honest to check against.
    """
    from src.features.engineer import FraudFeatureEngineer

    cut = int(len(synthetic_joined) * 0.8)
    train_part = synthetic_joined.iloc[:cut]

    engineer = FraudFeatureEngineer(
        v_groups=synthetic_v_groups,
        verbose=False,
    )
    engineer.fit(train_part, train_part["isFraud"])
    return engineer
