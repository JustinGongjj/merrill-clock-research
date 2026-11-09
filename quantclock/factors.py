"""Factor attribution using OLS with Newey-West/HAC standard errors."""
from __future__ import annotations

import pandas as pd


def factor_regression(strategy_returns: pd.Series, factor_returns: pd.DataFrame, maxlags: int = 3,
                      risk_free_returns: pd.Series | None = None):
    """Regress strategy excess returns on factor returns with HAC errors.

    Inputs must share a periodicity (e.g. monthly) and use returns, not price
    levels. Alpha is only a historical regression intercept, not a forecast.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError("Install statsmodels: pip install '.[analysis]'") from exc
    if maxlags < 0:
        raise ValueError("maxlags must be non-negative")
    y = strategy_returns.astype(float).copy()
    if risk_free_returns is not None:
        y = y - risk_free_returns.reindex(y.index).astype(float)
    frame = pd.concat([y.rename("strategy_excess"), factor_returns], axis=1).dropna()
    if len(frame) <= frame.shape[1] + 2:
        raise ValueError("Too few aligned observations for factor regression")
    y = frame.pop("strategy_excess")
    x = sm.add_constant(frame, has_constant="add")
    return sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
