"""Small, auditable portfolio backtest primitives (long-only, unlevered by default)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def prices_to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert adjusted daily prices to month-end returns."""
    p = prices.sort_index().copy()
    p.index = pd.to_datetime(p.index).tz_localize(None)
    monthly = p.resample("ME").last().dropna(how="all")
    return monthly.pct_change(fill_method=None).dropna(how="all")


def run_backtest(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 10.0,
    initial_cash: float = 1.0,
) -> pd.DataFrame:
    """Apply decisions after close t to returns starting at t+1.

    Weights must be target holdings indexed by decision date. The one-period
    shift is intentional. Transaction cost is charged on turnover at the prior
    decision date. Missing assets/weights are treated as zero, not forward-filled.
    """
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    common = asset_returns.index.intersection(target_weights.index)
    assets = asset_returns.columns.intersection(target_weights.columns)
    if len(common) == 0 or len(assets) == 0:
        raise ValueError("Returns and weights need overlapping dates and assets")
    r = asset_returns.loc[common, assets].fillna(0.0).sort_index()
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
    """Report descriptive historical metrics; no metric is a forecast."""
    x = pd.Series(returns).dropna().astype(float)
    if x.empty:
        return {k: float("nan") for k in ("total_return", "cagr", "annualized_volatility", "sharpe_zero_rf", "max_drawdown")}
    wealth = (1 + x).cumprod()
    years = len(x) / periods_per_year
    cagr = float(wealth.iloc[-1] ** (1 / years) - 1) if years > 0 and wealth.iloc[-1] > 0 else float("nan")
    vol = float(x.std(ddof=1) * np.sqrt(periods_per_year)) if len(x) > 1 else float("nan")
    sharpe = float(x.mean() / x.std(ddof=1) * np.sqrt(periods_per_year)) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")
    dd = wealth / wealth.cummax() - 1
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
