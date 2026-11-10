"""Small, auditable portfolio backtest primitives (long-only, unlevered by default)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def prices_to_monthly_returns(
    prices: pd.DataFrame,
    drop_incomplete_period: bool = True,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convert adjusted daily prices to month-end returns.

    ``resample('ME')`` labels a still-running month with its future month-end
    date. By default the last row is dropped when that label lies in the future,
    so a partial month is not booked as a completed monthly observation. Set
    ``drop_incomplete_period=False`` when feeding a deliberately truncated or
    simulated calendar. ``as_of`` overrides the calendar date used for that
    check, which keeps deterministic tests possible.
    """
    p = prices.sort_index().copy()
    p.index = pd.to_datetime(p.index).tz_localize(None)
    monthly = p.resample("ME").last().dropna(how="all")
    returns = monthly.div(monthly.shift(1)).sub(1.0).dropna(how="all")
    cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    if drop_incomplete_period and len(returns) and returns.index[-1] > cutoff:
        returns = returns.iloc[:-1]
    return returns


def run_backtest(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 10.0,
    initial_cash: float = 1.0,
) -> pd.DataFrame:
    """Apply decisions after close t to returns starting at t+1.

    Weights must be target holdings indexed by decision date. The one-period
    shift is intentional. Transaction cost is charged on turnover at the prior
    decision date. Missing weights are treated as zero, not forward-filled;
    missing asset returns are also treated as zero, but raise a warning so a
    short price history cannot silently become a zero-return asset.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    common = asset_returns.index.intersection(target_weights.index)
    assets = asset_returns.columns.intersection(target_weights.columns)
    if len(common) == 0 or len(assets) == 0:
        raise ValueError("Returns and weights need overlapping dates and assets")
    raw_returns = asset_returns.loc[common, assets].sort_index()
    missing_returns = int(raw_returns.isna().sum().sum())
    if missing_returns:
        warnings.warn(
            f"{missing_returns} missing asset-return cells were treated as zero returns; "
            "check instrument listing dates before interpreting results.",
            RuntimeWarning,
            stacklevel=2,
        )
    r = raw_returns.fillna(0.0)
    w = target_weights.loc[common, assets].fillna(0.0).sort_index()
    sums = w.sum(axis=1)
    w = w.div(sums.where(sums > 0), axis=0).fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * r).sum(axis=1)
    turnover_at_decision = w.sub(w.shift(1).fillna(0.0)).abs().sum(axis=1)
    charged_turnover = turnover_at_decision.shift(1).fillna(0.0)
    net = gross - charged_turnover * cost_bps / 10_000.0
    equity = initial_cash * (1 + net).cumprod()
    return pd.DataFrame({"gross_return": gross, "turnover": charged_turnover, "cost_return": charged_turnover * cost_bps / 10_000.0,
                         "net_return": net, "equity": equity})


def performance_metrics(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
    """Report descriptive historical metrics; no metric is a forecast.

    Drawdown is measured from the initial unit of capital, so a loss in the
    first period is not understated by starting the wealth path after that loss.
    """
    x = pd.Series(returns).dropna().astype(float)
    if x.empty:
        return {k: float("nan") for k in ("total_return", "cagr", "annualized_volatility", "sharpe_zero_rf", "max_drawdown")}
    wealth = (1 + x).cumprod()
    years = len(x) / periods_per_year
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1) if years > 0 and wealth.iloc[-1] > 0 else float("nan")
    vol = float(x.std(ddof=1) * np.sqrt(periods_per_year)) if len(x) > 1 else float("nan")
    sharpe = float(x.mean() / x.std(ddof=1) * np.sqrt(periods_per_year)) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")
    wealth_with_initial = pd.concat([pd.Series([1.0], dtype=float), wealth], ignore_index=True)
    dd = wealth_with_initial / wealth_with_initial.cummax() - 1
    return {"total_return": float(wealth.iloc[-1] - 1), "cagr": cagr, "annualized_volatility": vol,
            "sharpe_zero_rf": sharpe, "max_drawdown": float(dd.min())}


def walk_forward_splits(n_rows: int, min_train: int = 60, test_size: int = 12, step: int | None = None):
    """Yield chronological positional (train_end, test_start, test_end) slices."""
    if min_train < 1 or test_size < 1:
        raise ValueError("min_train and test_size must be positive")
    stride = test_size if step is None else step
    if stride < 1:
        raise ValueError("step must be positive")
    test_start = min_train
    while test_start < n_rows:
        test_end = min(test_start + test_size, n_rows)
        yield slice(0, test_start), slice(test_start, test_end)
        test_start += stride


def buy_and_hold_weights(asset_returns: pd.DataFrame, assets: list[str] | None = None) -> pd.DataFrame:
    """Equal-weight static benchmark over selected assets."""
    columns = assets or list(asset_returns.columns)
    if not columns:
        raise ValueError("At least one asset required")
    return pd.DataFrame(1 / len(columns), index=asset_returns.index, columns=columns)
