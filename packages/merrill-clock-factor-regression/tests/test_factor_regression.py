import numpy as np
import pandas as pd
import pytest

from factorlab.regression import factor_regression
from examples_factor_regression import synthetic_demo


def test_factor_regression_runs_with_risk_free_adjustment():
    rng = np.random.default_rng(12)
    idx = pd.date_range("2010-01-31", periods=80, freq="ME")
    factors = pd.DataFrame({"Mkt-RF": rng.normal(0, .04, 80), "SMB": rng.normal(0, .02, 80)}, index=idx)
    rf = pd.Series(.001, index=idx)
    strategy = .001 + .7 * factors["Mkt-RF"] + .1 * factors["SMB"] + rf + rng.normal(0, .01, 80)
    result = factor_regression(strategy, factors, maxlags=3, risk_free_returns=rf)
    assert int(result.nobs) == 80
    assert "const" in result.params.index
    assert np.isfinite(result.tvalues).all()


def test_offline_demo_is_reproducible_and_regressible():
    strategy, factors, rf = synthetic_demo(seed=44, periods=90)
    result = factor_regression(strategy, factors, maxlags=3, risk_free_returns=rf)
    assert int(result.nobs) == 90
    assert np.isfinite(result.params).all()


def test_regression_rejects_too_few_complete_rows():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.Series([.01, .02, .01, .00], index=idx)
    x = pd.DataFrame({"factor": [.01, .00, .02, -.01]}, index=idx)
    with pytest.raises(ValueError, match="Too few"):
        factor_regression(y, x)


def _monthly_sample(periods=80):
    rng = np.random.default_rng(5)
    factor_idx = pd.date_range("2015-01-31", periods=periods, freq="ME")
    factors = pd.DataFrame({
        "Mkt-RF": rng.normal(0, .04, periods),
        "SMB": rng.normal(0, .02, periods),
    }, index=factor_idx)
    rf = pd.Series(.001, index=factor_idx, name="RF")
    strategy_idx = pd.date_range("2015-01-01", periods=periods, freq="MS")
    strategy = pd.Series(
        .002 + .7 * factors["Mkt-RF"].to_numpy() + .1 * factors["SMB"].to_numpy()
        + rng.normal(0, .01, periods), index=strategy_idx, name="net_return"
    )
    return strategy, factors, rf


def test_month_start_strategy_dates_align_to_month_end_factors():
    strategy, factors, rf = _monthly_sample()
    result = factor_regression(strategy, factors, maxlags=3, risk_free_returns=rf)
    assert int(result.nobs) == 80
    assert "calendar month" in result.alignment_note
    assert result.sample_start == "2015-01-31"
    assert result.sample_end == "2021-08-31"


def test_rf_column_is_risk_free_rate_not_a_factor():
    strategy, factors, rf = _monthly_sample()
    with_rf = factors.assign(RF=rf.to_numpy())
    automatic = factor_regression(strategy, with_rf)
    explicit = factor_regression(strategy, factors, risk_free_returns=rf)
    assert "RF" not in automatic.params.index
    assert np.allclose(automatic.params.to_numpy(), explicit.params.to_numpy())


def test_two_risk_free_sources_are_rejected():
    strategy, factors, rf = _monthly_sample()
    with pytest.raises(ValueError, match="double"):
        factor_regression(strategy, factors.assign(RF=rf.to_numpy()), risk_free_returns=rf)


def test_disjoint_months_report_the_alignment_problem():
    strategy, factors, _ = _monthly_sample(60)
    shifted = factors.copy()
    shifted.index = shifted.index + pd.DateOffset(years=20)
    with pytest.raises(ValueError, match="overlapping calendar months"):
        factor_regression(strategy, shifted)


def test_hac_small_sample_correction_is_optional():
    strategy, factors, rf = _monthly_sample()
    corrected = factor_regression(strategy, factors, risk_free_returns=rf, use_correction=True)
    assert np.isfinite(corrected.tvalues).all()
    assert int(corrected.nobs) == 80
