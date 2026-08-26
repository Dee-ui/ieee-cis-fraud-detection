"""
Tests for the drift detectors.

These matter because a broken drift detector fails in the worst possible
way: it reports that everything is fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from config.config import PSI_SIGNIFICANT, PSI_STABLE
from src.monitoring.drift import (
    kolmogorov_smirnov,
    missing_rate,
    population_stability_index,
)


def test_psi_of_a_distribution_against_itself_is_near_zero():
    """No change must produce no signal."""
    rng = np.random.default_rng(11)
    sample = rng.normal(0, 1, 20_000)

    psi = population_stability_index(sample, sample.copy())

    assert psi < 0.01


def test_psi_flags_a_clear_shift():
    """A distribution moved by two standard deviations must trip the alarm."""
    rng = np.random.default_rng(12)
    reference = rng.normal(0, 1, 20_000)
    current = rng.normal(2, 1, 20_000)

    psi = population_stability_index(reference, current)

    assert psi > PSI_SIGNIFICANT


def test_psi_catches_a_collapse_onto_one_value():
    """
    The uid_freq failure, in miniature.

    When most rows collapse onto a single value, nothing goes blank, so a
    missingness check sees a healthy feature. PSI must catch it, because
    that is the whole reason PSI is the primary signal. Decision D-53.
    """
    rng = np.random.default_rng(13)
    reference = rng.uniform(0, 1, 20_000)

    current = rng.uniform(0, 1, 20_000)
    current[: int(0.82 * len(current))] = 0.0  # 82% collapse onto zero

    assert missing_rate(current) == pytest.approx(0.0)  # nothing is blank
    assert population_stability_index(reference, current) > PSI_SIGNIFICANT


def test_psi_tolerates_a_small_wobble():
    """Random sampling noise must not look like drift."""
    rng = np.random.default_rng(14)
    reference = rng.normal(0, 1, 20_000)
    current = rng.normal(0.02, 1, 20_000)

    assert population_stability_index(reference, current) < PSI_STABLE


def test_ks_ranges_from_zero_to_one():
    """Identical samples give roughly zero, disjoint samples give roughly one."""
    rng = np.random.default_rng(15)
    sample = rng.normal(0, 1, 5000)

    assert kolmogorov_smirnov(sample, sample.copy()) < 0.05
    assert kolmogorov_smirnov(sample, sample + 100) > 0.95


def test_psi_returns_nan_when_there_is_not_enough_data():
    """A handful of rows cannot support a distribution comparison."""
    assert np.isnan(population_stability_index(np.array([1.0, 2.0]), np.array([1.0])))


def test_missing_rate_counts_blanks():
    values = np.array([1.0, np.nan, 3.0, np.nan])
    assert missing_rate(values) == pytest.approx(0.5)
