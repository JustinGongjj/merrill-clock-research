"""Tests for composite-factor weighting methods."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorlab.synthesis import (
    METHODS, composite_factor, half_life_weights, synthesize_weights, weight_stability,
)

DATES = pd.date_range("2018-01-31", periods=30, freq="ME")


def _ic_history(seed: int = 4, columns=("f1", "f2", "f3")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.02, 0.05, size=(len(DATES), len(columns)))
    base[:, 0] += 0.05  # f1 has the strongest mean IC
    return pd.DataFrame(base, index=DATES, columns=list(columns))


def test_half_life_weights_halve_each_half_life_period():
    weights = half_life_weights(24, half_life=12)
    assert weights[-1] / weights[-13] == pytest.approx(2.0, rel=1e-9)
    assert weights.sum() == pytest.approx(1.0)


def test_half_life_weights_validation():
    with pytest.raises(ValueError, match="n_periods"):
        half_life_weights(0)
    with pytest.raises(ValueError, match="half_life"):
        half_life_weights(10, half_life=0)


def test_equal_weights_are_uniform():
    result = synthesize_weights(_ic_history(), method="equal")
    values = list(result["weights"].values())
    assert values == pytest.approx([1 / 3] * 3)
    assert result["weight_sum"] == pytest.approx(1.0)
    assert result["warnings"] == []


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_return_nonnegative_normalized_weights(method):
    result = synthesize_weights(_ic_history(), method=method, window=12)
    weights = np.array(list(result["weights"].values()))
    assert np.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum() == pytest.approx(1.0)
    assert result["window_used"] == 12


def test_ic_half_life_favors_factor_with_higher_ic():
    result = synthesize_weights(_ic_history(), method="ic_half_life", window=12)
    assert result["weights"]["f1"] > result["weights"]["f3"]


def test_max_ic_ir_allow_short_can_produce_negative_weights():
    history = _ic_history(seed=8)
    result = synthesize_weights(history, method="max_ic_ir", window=12, allow_short=True)
    assert result["allow_short"] is True
    assert sum(abs(w) for w in result["weights"].values()) == pytest.approx(1.0)
    assert abs(sum(result["weights"].values())) <= 1.0 + 1e-9


def test_short_window_warns_on_covariance_estimation():
    result = synthesize_weights(_ic_history(columns=("f1", "f2", "f3", "f4")),
                                method="max_ic_ir", window=3)
    assert any("协方差" in w for w in result["warnings"])


def test_empty_or_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        synthesize_weights(pd.DataFrame())
    with pytest.raises(ValueError, match="method"):
        synthesize_weights(_ic_history(), method="magic")
    with pytest.raises(ValueError, match="window"):
        synthesize_weights(_ic_history(), window=1)


def test_all_nan_column_gets_zero_weight_and_warning():
    history = _ic_history(columns=("f1", "f2", "f3"))
    history["f3"] = np.nan
    result = synthesize_weights(history, method="equal")
    assert result["weights"]["f3"] == 0.0
    assert any("f3" in w for w in result["warnings"])


def test_composite_factor_is_weighted_sum():
    f1 = pd.DataFrame([[1.0, 2.0]], index=DATES[:1], columns=["A", "B"])
    f2 = pd.DataFrame([[3.0, 4.0]], index=DATES[:1], columns=["A", "B"])
    result = composite_factor({"f1": f1, "f2": f2}, {"f1": 0.25, "f2": 0.75})
    assert result.loc[DATES[0], "A"] == pytest.approx(0.25 * 1 + 0.75 * 3)


def test_composite_factor_uses_only_overlapping_cells():
    f1 = pd.DataFrame([[1.0, 2.0]], index=DATES[:1], columns=["A", "B"])
    f2 = pd.DataFrame([[3.0]], index=DATES[:1], columns=["A"])
    result = composite_factor({"f1": f1, "f2": f2}, {"f1": 0.5, "f2": 0.5})
    assert list(result.columns) == ["A"]


def test_composite_factor_rejects_zero_weights():
    f1 = pd.DataFrame([[1.0]], index=DATES[:1], columns=["A"])
    with pytest.raises(ValueError, match="sum to zero"):
        composite_factor({"f1": f1}, {"f1": 0.0})


def test_weight_stability_zero_for_constant_weights():
    weights = [{"a": 0.6, "b": 0.4}] * 4
    assert weight_stability(weights)["mean_l1_turnover"] == pytest.approx(0.0)
    assert weight_stability(weights[:1])["n_changes"] == 0


def test_weight_stability_detects_changes():
    weights = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}]
    assert weight_stability(weights)["mean_l1_turnover"] == pytest.approx(2.0)
