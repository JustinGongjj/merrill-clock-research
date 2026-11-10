import numpy as np
import pandas as pd

from quantclock.factors import factor_regression


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
