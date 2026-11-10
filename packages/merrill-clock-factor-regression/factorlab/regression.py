"""Factor attribution using OLS with Newey-West/HAC standard errors."""
from __future__ import annotations

import pandas as pd


def _month_keys(index: pd.Index) -> pd.PeriodIndex | None:
    """Return one key per calendar month, or None for non-monthly/multi-obs data."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    keys = pd.PeriodIndex(idx.to_period("M"), freq="M")
    return None if keys.has_duplicates else keys


def _align_inputs(
    strategy: pd.Series,
    factors: pd.DataFrame,
    risk_free: pd.Series | None,
) -> tuple[pd.Series, pd.DataFrame, pd.Series | None, str]:
    """Align strategy, factors and RF on common dates.

    Exact date matches are preferred. When one side uses month-start labels and
    the other month-end labels (a very common CSV / Fama-French mismatch), align
    on the calendar month instead of silently returning an empty regression.
    """
    exact = strategy.index.intersection(factors.index)
    if len(exact) >= 3:
        rf = risk_free.reindex(exact) if risk_free is not None else None
        return strategy.loc[exact], factors.loc[exact], rf, "exact date match"

    y_keys = _month_keys(strategy.index)
    f_keys = _month_keys(factors.index)
    if y_keys is None or f_keys is None:
        raise ValueError(
            "No overlapping dates, and at least one input has multiple observations per month. "
            "Supply series with matching timestamps. "
            f"Strategy: {strategy.index.min()} .. {strategy.index.max()} ({len(strategy)} rows); "
            f"factors: {factors.index.min()} .. {factors.index.max()} ({len(factors)} rows)."
        )
    y = strategy.set_axis(y_keys)
    f = factors.set_axis(f_keys)
    common = y.index.intersection(f.index)
    if len(common) < 3:
        raise ValueError(
            "No overlapping calendar months between strategy returns and factor returns. "
            f"Strategy covers {y.index.min()} .. {y.index.max()}, factors cover {f.index.min()} .. {f.index.max()}. "
            "Check frequency, date labels and return units before regressing."
        )
    rf_aligned = None
    if risk_free is not None:
        rf_keys = _month_keys(risk_free.index)
        if rf_keys is None:
            raise ValueError("risk_free_returns must have at most one observation per month")
        rf_aligned = pd.Series(risk_free.to_numpy(dtype=float), index=rf_keys, name="RF").reindex(common)
    dates = common.to_timestamp(how="end").normalize()
    y = y.loc[common].set_axis(dates)
    f = f.loc[common].set_axis(dates)
    if rf_aligned is not None:
        rf_aligned = rf_aligned.set_axis(dates)
    return y, f, rf_aligned, "aligned by calendar month (month-start / month-end normalized to month-end)"


def factor_regression(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    maxlags: int = 3,
    risk_free_returns: pd.Series | None = None,
    use_correction: bool = False,
):
    """Regress strategy excess returns on factor returns with HAC errors.

    Inputs must share a periodicity (e.g. monthly) and use decimal returns, not
    price levels or percentage points. Alpha is only a historical regression
    intercept, not a forecast.

    An ``RF`` column inside ``factor_returns`` is treated as the risk-free
    series; passing ``risk_free_returns`` at the same time raises an error to
    avoid subtracting the risk-free rate twice. ``use_correction`` enables the
    small-sample degrees-of-freedom correction of the HAC covariance.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise RuntimeError("Install statsmodels: pip install -e '.[analysis]'") from exc
    if maxlags < 0:
        raise ValueError("maxlags must be non-negative")
    if not isinstance(factor_returns, pd.DataFrame):
        raise TypeError("factor_returns must be a DataFrame with one column per factor")
    if "strategy_excess" in factor_returns.columns:
        raise ValueError("'strategy_excess' is a reserved column name for the dependent variable")

    factors = factor_returns.copy()
    rf = risk_free_returns
    if "RF" in factors.columns:
        if rf is not None:
            raise ValueError(
                "factor_returns already contains an 'RF' column and risk_free_returns was supplied; "
                "choose one source to avoid double-subtracting the risk-free rate"
            )
        rf = factors.pop("RF")

    y_raw = pd.Series(strategy_returns).astype(float).copy()
    y, factors, rf, alignment_note = _align_inputs(y_raw, factors, rf)
    if rf is not None:
        y = y - rf.astype(float)
    frame = pd.concat([y.rename("strategy_excess"), factors], axis=1).dropna()
    aligned_rows = len(frame)
    if aligned_rows == 0:
        raise ValueError(
            "No complete rows after date alignment and NaN removal. Check that returns are decimals "
            "and that strategy, factors and RF share the same observation dates."
        )
    dependent = frame.pop("strategy_excess")
    x = sm.add_constant(frame, has_constant="add")
    if aligned_rows <= x.shape[1] + 2:
        raise ValueError(
            f"Too few aligned observations for factor regression: {aligned_rows} complete rows "
            f"for {x.shape[1]} parameters. ({alignment_note})"
        )
    fit = sm.OLS(dependent, x).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": bool(use_correction)}
    )
    # Diagnostics travel with the result so callers can audit the sample window.
    try:
        fit.alignment_note = alignment_note
        fit.sample_start = str(dependent.index.min().date())
        fit.sample_end = str(dependent.index.max().date())
        fit.dropped_incomplete_rows = int(len(y_raw) - aligned_rows)
    except AttributeError:
        pass
    return fit
