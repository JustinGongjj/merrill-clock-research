"""Unit and functional tests for cross-sectional single-factor testing."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from factorlab.crosssection import (
    _label, evaluate_factor, fama_macbeth, ic_summary, layer_monotonicity,
    layered_returns, neutralize, rank_ic, winsorize_mad, zscore,
)

ASSETS = [f"A{i}" for i in range(8)]
DATES = pd.date_range("2015-01-31", periods=36, freq="ME")


def _perfect(rng: np.random.Generator, sign: float = 1.0):
    """Factor at t ranks the return from t to t+1 exactly."""
    factor = pd.DataFrame(rng.normal(size=(len(DATES), len(ASSETS))),
                          index=DATES, columns=ASSETS)
    forward = sign * factor.rank(axis=1) / 100.0
    return factor, forward


def test_label_handles_timestamp_and_plain_index():
    assert _label(pd.Timestamp("2020-01-31")) == "2020-01-31"
    assert _label(7) == "7"
    assert _label("2020-01") == "2020-01"


def test_winsorize_mad_clips_outliers():
    frame = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 1000.0]], columns=ASSETS[:5])
    clipped = winsorize_mad(frame, n_mad=5)
    assert clipped.iloc[0, -1] < 1000.0
    assert clipped.iloc[0, 0] == 1.0


def test_winsorize_mad_leaves_constant_row_untouched():
    frame = pd.DataFrame([[2.0] * 5], columns=ASSETS[:5])
    pd.testing.assert_frame_equal(winsorize_mad(frame), frame)


def test_winsorize_mad_rejects_nonpositive_n_mad():
    with pytest.raises(ValueError, match="n_mad"):
        winsorize_mad(pd.DataFrame([[1.0, 2.0]]), n_mad=0)


def test_zscore_centers_and_fills_missing_as_zero():
    frame = pd.DataFrame([[1.0, 2.0, 3.0, np.nan]], columns=ASSETS[:4])
    scaled = zscore(frame)
    assert scaled.iloc[0, 3] == 0.0
    assert abs(scaled.iloc[0, :3].mean()) < 1e-12


def test_zscore_can_keep_missing():
    frame = pd.DataFrame([[1.0, 2.0, np.nan]], columns=ASSETS[:3])
    assert math.isnan(zscore(frame, fill_missing=None).iloc[0, 2])


def test_neutralize_removes_size_exposure():
    rng = np.random.default_rng(11)
    size = pd.DataFrame(rng.normal(size=(10, len(ASSETS))), index=DATES[:10], columns=ASSETS)
    factor = 0.8 * size + pd.DataFrame(rng.normal(0, 0.01, size=(10, len(ASSETS))),
                                       index=DATES[:10], columns=ASSETS)
    residual = neutralize(factor, size=size)
    row = residual.iloc[0]
    assert abs(np.corrcoef(row.to_numpy(), size.iloc[0].to_numpy())[0, 1]) < 0.05


def test_neutralize_without_controls_is_identity():
    frame = pd.DataFrame(np.arange(12, dtype=float).reshape(3, 4))
    pd.testing.assert_frame_equal(neutralize(frame), frame)


def test_rank_ic_perfect_positive_and_negative():
    rng = np.random.default_rng(3)
    factor, forward = _perfect(rng, sign=1.0)
    assert rank_ic(factor, forward).dropna().mean() == pytest.approx(1.0, abs=1e-9)
    _, forward_neg = _perfect(rng, sign=-1.0)
    # rebuild with identical factor rows so the sign flip is isolated
    forward_neg = -forward
    assert rank_ic(factor, forward_neg).dropna().mean() == pytest.approx(-1.0, abs=1e-9)


def test_rank_ic_skips_dates_with_too_few_assets():
    factor = pd.DataFrame([[1.0, 2.0, np.nan, np.nan]], columns=ASSETS[:4])
    forward = pd.DataFrame([[0.1, 0.2, np.nan, np.nan]], columns=ASSETS[:4])
    result = rank_ic(factor, forward, min_assets=3)
    assert math.isnan(result.iloc[0])


def test_rank_ic_is_nan_for_constant_cross_section():
    factor = pd.DataFrame([[1.0] * 6], columns=ASSETS[:6])
    forward = pd.DataFrame(np.arange(6, dtype=float).reshape(1, 6), columns=ASSETS[:6])
    assert math.isnan(rank_ic(factor, forward).iloc[0])


def test_rank_ic_rejects_min_assets_below_two():
    with pytest.raises(ValueError, match="min_assets"):
        rank_ic(pd.DataFrame([[1.0, 2.0]]), pd.DataFrame([[1.0, 2.0]]), min_assets=1)


def test_ic_summary_constant_series_has_infinite_ir_as_nan():
    summary = ic_summary(pd.Series([0.1] * 12))
    assert summary["ic_mean"] == pytest.approx(0.1)
    assert summary["positive_ratio"] == 1.0
    assert math.isnan(summary["icir"])
    assert summary["n_periods"] == 12


def test_ic_summary_empty_series():
    summary = ic_summary(pd.Series([], dtype=float))
    assert summary["n_periods"] == 0 and math.isnan(summary["ic_mean"])


def test_ic_summary_known_values():
    series = pd.Series([0.1, -0.1] * 10)
    summary = ic_summary(series)
    assert summary["ic_mean"] == 0.0
    assert summary["positive_ratio"] == 0.5
    assert summary["ic_std"] > 0


def test_layered_returns_are_monotone_for_perfect_factor():
    rng = np.random.default_rng(5)
    factor, forward = _perfect(rng)
    layers = layered_returns(factor, forward, n_groups=4, min_assets=8)
    means = [layers[f"g{i}"].mean() for i in range(1, 5)]
    assert means == sorted(means)
    assert layer_monotonicity(layers)["is_monotonic"] is True


def test_layer_monotonicity_detects_nonlinear_middle_best():
    layers = pd.DataFrame({"g1": [0.0] * 4, "g2": [0.5] * 4, "g3": [0.1] * 4})
    marker = layer_monotonicity(layers)
    assert marker["middle_best"] is True
    assert marker["shape"] == "nonlinear_middle_best"
    assert marker["is_monotonic"] is False
    assert "非线性" in marker["note"]


def test_layer_monotonicity_insufficient_data():
    assert layer_monotonicity(pd.DataFrame())["shape"] == "insufficient_data"


def test_fama_macbeth_slope_positive_for_predictive_factor():
    rng = np.random.default_rng(9)
    factor, forward = _perfect(rng)
    result = fama_macbeth(factor, forward)
    assert result["slope_mean"] > 0
    assert result["positive_ratio"] == 1.0
    assert result["n_periods"] == len(DATES)


def test_evaluate_factor_report_is_json_serializable():
    rng = np.random.default_rng(21)
    factor, forward = _perfect(rng)
    report = evaluate_factor(factor, forward, n_groups=4, min_assets=8)
    json.dumps(report)  # must not contain numpy scalars / Timestamp keys
    assert report["ic_summary"]["ic_mean"] == pytest.approx(1.0, abs=1e-6)
    assert report["layer_monotonicity"]["is_monotonic"] is True
    assert len(report["preprocessing"]) >= 3
    assert report["sample_start"] == "2015-01-31"
    assert report["caveats"]


def test_evaluate_factor_without_preprocessing_still_works():
    rng = np.random.default_rng(22)
    factor, forward = _perfect(rng)
    report = evaluate_factor(factor, forward, n_groups=4, min_assets=8, preprocess=False)
    assert report["preprocessing"] == []
    assert report["ic_summary"]["ic_mean"] == pytest.approx(1.0, abs=1e-6)


def test_evaluate_factor_with_industry_and_size_controls():
    rng = np.random.default_rng(23)
    factor, forward = _perfect(rng)
    size = pd.DataFrame(rng.normal(size=(len(DATES), len(ASSETS))), index=DATES, columns=ASSETS)
    industry = pd.DataFrame(
        [["X" if i < 4 else "Y" for i in range(len(ASSETS))]] * len(DATES),
        index=DATES, columns=ASSETS)
    report = evaluate_factor(factor, forward, industry=industry, size=size,
                             n_groups=4, min_assets=8)
    assert any("neutralization" in step for step in report["preprocessing"])
    assert math.isfinite(report["ic_summary"]["ic_mean"])


def test_misaligned_inputs_raise_clear_error():
    factor = pd.DataFrame([[1.0, 2.0]], columns=["A", "B"], index=DATES[:1])
    forward = pd.DataFrame([[1.0, 2.0]], columns=["C", "D"], index=DATES[:1])
    with pytest.raises(ValueError, match="overlapping"):
        rank_ic(factor, forward)


def test_forward_returns_are_used_not_same_period_returns():
    """Shifting the factor forward must destroy the perfect rank correlation."""
    rng = np.random.default_rng(31)
    factor, forward = _perfect(rng)
    shifted = factor.shift(1)
    assert rank_ic(factor, forward).dropna().mean() == pytest.approx(1.0, abs=1e-9)
    assert rank_ic(shifted, forward).dropna().mean() < 0.5
