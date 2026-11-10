"""Tests for descriptive phase-probability calibration diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantclock.calibration import (
    brier_score, phase_diagnostics, realized_quadrant, temperature_sensitivity,
)
from quantclock.signals import regime_probabilities

REGIMES = ["recovery", "overheat", "stagflation", "reflation"]
DATES = pd.date_range("2010-01-31", periods=60, freq="ME")


def _features(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"growth_accel": rng.normal(0, 1, len(DATES)),
                         "inflation_accel": rng.normal(0, 1, len(DATES))}, index=DATES)


def _perfect_probabilities(features: pd.DataFrame) -> pd.DataFrame:
    pg = (features["growth_accel"].shift(-1) > 0).astype(float)
    pi = (features["inflation_accel"].shift(-1) > 0).astype(float)
    return pd.DataFrame({
        "recovery": pg * (1 - pi), "overheat": pg * pi,
        "stagflation": (1 - pg) * pi, "reflation": (1 - pg) * (1 - pi),
    }, index=features.index)


def test_brier_perfect_and_worst():
    p = pd.Series([1.0, 0.0, 1.0], index=[1, 2, 3])
    assert brier_score(p, p) == pytest.approx(0.0)
    assert brier_score(p, 1 - p) == pytest.approx(1.0)


def test_brier_ignores_missing_pairs():
    p = pd.Series([0.5, np.nan], index=[1, 2])
    y = pd.Series([1.0, 1.0], index=[1, 2])
    assert brier_score(p, y) == pytest.approx(0.25)
    assert np.isnan(brier_score(pd.Series(dtype=float), pd.Series(dtype=float)))


@pytest.mark.parametrize("g,i,expected", [
    (True, False, "recovery"), (True, True, "overheat"),
    (False, True, "stagflation"), (False, False, "reflation"),
])
def test_realized_quadrant_mapping(g, i, expected):
    assert realized_quadrant(g, i) == expected


def test_perfect_probabilities_hit_every_quadrant_with_zero_brier():
    features = _features()
    result = phase_diagnostics(_perfect_probabilities(features), features)
    assert result["n_periods"] == len(DATES) - 1
    assert result["quadrant_hit_rate"] == pytest.approx(1.0)
    assert result["brier_growth_up"] == pytest.approx(0.0)
    assert result["brier_inflation_up"] == pytest.approx(0.0)
    assert result["hit_rate_minus_baseline"] > 0
    assert result["mean_normalized_entropy"] == pytest.approx(0.0, abs=1e-9)
    assert sum(result["regime_share"].values()) == pytest.approx(1.0)
    assert result["caveats"]


def test_uniform_probabilities_have_maximum_entropy():
    features = _features()
    uniform = pd.DataFrame(0.25, index=DATES, columns=REGIMES)
    result = phase_diagnostics(uniform, features)
    assert result["mean_normalized_entropy"] == pytest.approx(1.0)
    assert result["brier_growth_up"] == pytest.approx(0.25)


def test_constant_baseline_is_most_frequent_realized_quadrant():
    features = pd.DataFrame({"growth_accel": np.ones(len(DATES)),
                             "inflation_accel": -np.ones(len(DATES))}, index=DATES)
    uniform = pd.DataFrame(0.25, index=DATES, columns=REGIMES)
    result = phase_diagnostics(uniform, features)
    assert result["constant_baseline_label"] == "recovery"
    assert result["constant_baseline_hit_rate"] == pytest.approx(1.0)


def test_mean_dominant_regime_duration_is_finite():
    features = _features()
    probs = regime_probabilities(features, window=12, temperature=1.0)
    result = phase_diagnostics(probs, features)
    assert result["mean_dominant_regime_duration_periods"] >= 1.0


def test_missing_columns_raise_clear_errors():
    features = _features()
    probs = _perfect_probabilities(features)
    with pytest.raises(ValueError, match="features must contain"):
        phase_diagnostics(probs, features[["growth_accel"]])
    with pytest.raises(ValueError, match="missing regime columns"):
        phase_diagnostics(probs.drop(columns=["recovery"]), features)


def test_too_few_observations_raise():
    features = _features().iloc[:2]
    with pytest.raises(ValueError, match="three aligned"):
        phase_diagnostics(pd.DataFrame(0.25, index=features.index, columns=REGIMES), features)


def test_temperature_sensitivity_is_monotone_in_decisiveness():
    features = _features()
    result = temperature_sensitivity(features, temperatures=(0.5, 1.0, 2.0), window=12)
    by = result["by_temperature"]
    assert set(by) == {"0.5", "1.0", "2.0"}
    assert by["0.5"]["mean_max_probability"] > by["1.0"]["mean_max_probability"]
    assert by["1.0"]["mean_max_probability"] > by["2.0"]["mean_max_probability"]
    assert result["window"] == 12
