"""
Drift detection.

Three measures, each catching something the others miss:

  PSI          how far a distribution has moved, bucket by bucket. The
               primary signal, because it catches a collapse onto one value
               that a missingness check cannot see.
  KS           the largest gap between two cumulative curves. A second
               opinion that needs no buckets.
  missingness  the share of blanks, and how much it changed.

Only the KS statistic is used, never its p-value. On 100,000 rows every
difference is statistically significant, so the p-value would flag all 284
features every month and tell you nothing. Decision D-54.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config.config import (
    DRIFT_MIN_ROWS,
    KS_SAMPLE_SIZE,
    PSI_BINS,
    PSI_SIGNIFICANT,
    PSI_STABLE,
    RANDOM_SEED,
)

# Stops a bucket that emptied completely from producing infinity.
EPSILON = 1e-6


def _usable(values) -> np.ndarray:
    """Drop blanks and infinities, returning a plain float array."""
    array = np.asarray(values, dtype="float64")
    return array[np.isfinite(array)]


def missing_rate(values) -> float:
    """Share of values that are blank or infinite."""
    array = np.asarray(values, dtype="float64")
    if array.size == 0:
        return float("nan")
    return float((~np.isfinite(array)).mean())


def population_stability_index(reference, current, bins: int = PSI_BINS) -> float:
    """
    How far has this distribution moved?

    Cut the reference into equal-sized buckets, then see what share of the
    current data lands in each. If nothing changed, each bucket still holds
    about the same share and the answer is near zero.

        PSI = sum over buckets of (new share - old share) x ln(new / old)

    The multiplication is what gives it teeth. A bucket that moved from 10%
    to 12% barely registers. One that emptied from 10% to 0.5% contributes
    heavily, because the logarithm punishes proportional collapse. That is
    exactly the failure mode we need to catch: a feature that stops varying
    without ever going blank.

    Reading it:
        under 0.10   stable
        0.10 to 0.25 moderate, worth watching
        over 0.25    significant, investigate
    """
    reference_values = _usable(reference)
    current_values = _usable(current)

    if len(reference_values) < DRIFT_MIN_ROWS or len(current_values) < DRIFT_MIN_ROWS:
        return float("nan")

    # Bucket edges come from the reference, so the reference is 10% per
    # bucket by construction and the current data is what moves.
    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))

    # A column with only one or two distinct values cannot be bucketed.
    if len(edges) < 3:
        return float("nan")

    # Open the outer edges so values beyond the training range are counted
    # rather than dropped. Those are exactly the ones worth noticing.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_share = np.histogram(reference_values, bins=edges)[0] / len(
        reference_values
    )
    current_share = np.histogram(current_values, bins=edges)[0] / len(current_values)

    reference_share = np.clip(reference_share, EPSILON, None)
    current_share = np.clip(current_share, EPSILON, None)

    return float(
        np.sum(
            (current_share - reference_share) * np.log(current_share / reference_share)
        )
    )


def kolmogorov_smirnov(reference, current) -> float:
    """
    The largest vertical gap between two cumulative distribution curves.

    Runs from 0, identical, to 1, no overlap at all. Needs no buckets, so it
    cannot be fooled by an unlucky bucket choice, and it is more sensitive
    than PSI to a shift in the middle of a distribution.

    Both sides are subsampled, because the statistic settles down long before
    100,000 rows and the test is slow on large inputs.
    """
    reference_values = _usable(reference)
    current_values = _usable(current)

    if len(reference_values) < DRIFT_MIN_ROWS or len(current_values) < DRIFT_MIN_ROWS:
        return float("nan")

    rng = np.random.default_rng(RANDOM_SEED)
    if len(reference_values) > KS_SAMPLE_SIZE:
        reference_values = rng.choice(reference_values, KS_SAMPLE_SIZE, replace=False)
    if len(current_values) > KS_SAMPLE_SIZE:
        current_values = rng.choice(current_values, KS_SAMPLE_SIZE, replace=False)

    # .statistic only. The p-value is deliberately ignored, see D-54.
    return float(ks_2samp(reference_values, current_values).statistic)


def drift_band(psi: float) -> str:
    """Turn a PSI number into a word a human can act on."""
    if not np.isfinite(psi):
        return "unknown"
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


def compare_features(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    period_label: str,
) -> pd.DataFrame:
    """Run all three measures on every feature, for one period."""
    records = []

    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue

        reference_values = reference[feature].to_numpy(dtype="float64")
        current_values = current[feature].to_numpy(dtype="float64")

        psi = population_stability_index(reference_values, current_values)
        missing_reference = missing_rate(reference_values)
        missing_current = missing_rate(current_values)

        records.append(
            {
                "period": period_label,
                "feature": feature,
                "psi": psi,
                "ks": kolmogorov_smirnov(reference_values, current_values),
                "band": drift_band(psi),
                "missing_reference": missing_reference,
                "missing_current": missing_current,
                "missing_change": missing_current - missing_reference,
                "mean_reference": (
                    float(np.nanmean(reference_values))
                    if np.isfinite(reference_values).any()
                    else float("nan")
                ),
                "mean_current": (
                    float(np.nanmean(current_values))
                    if np.isfinite(current_values).any()
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(records)


def weighted_drift_score(drift: pd.DataFrame, importance: pd.DataFrame) -> float:
    """
    One number for the whole period, weighted by how much the model cares.

    With 284 features a few will always have drifted. A raw count fires every
    month and gets ignored, which is worse than no alarm at all. Weighting by
    SHAP importance means drift in C13, the model's top feature, dominates,
    while drift in has_identity, which the model never uses, contributes
    nothing. Decision D-55.
    """
    weights = importance.set_index("feature")["mean_abs_shap"]
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return float("nan")

    merged = drift.copy()
    merged["weight"] = merged["feature"].map(weights).fillna(0.0)
    merged = merged[np.isfinite(merged["psi"])]

    if merged.empty:
        return float("nan")

    return float((merged["psi"] * merged["weight"]).sum() / total_weight)
