"""Leakage-conscious walk-forward linear forecasting baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd


def walk_forward_ridge(
    features: pd.DataFrame,
    asset_returns: pd.DataFrame,
    min_train: int = 60,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Predict each asset's next row return using only information available at t.

    Features at t predict return at t+1. At each origin i, only samples strictly
    before i are fit. Standardization parameters are recalculated on that train
    window, intercept is not penalized, and target scales are not standardized.
    The result is indexed by decision date, so prediction at t maps to the next
    row in asset_returns.
    """
    if min_train < 2 or alpha < 0:
        raise ValueError("min_train must be >=2 and alpha non-negative")
    dates = features.index.intersection(asset_returns.index)
    x = features.loc[dates].astype(float)
    y = asset_returns.loc[dates].astype(float).shift(-1)
    cols = list(asset_returns.columns)
    output = pd.DataFrame(np.nan, index=dates, columns=cols, dtype=float)
    for i in range(min_train, len(dates)):
        x_train = x.iloc[:i]
        y_train = y.iloc[:i]
        valid = x_train.notna().all(axis=1) & y_train.notna().all(axis=1)
        if int(valid.sum()) < min_train:
            continue
        xt = x_train.loc[valid].to_numpy()
        yt = y_train.loc[valid].to_numpy()
        mean = xt.mean(axis=0)
        scale = xt.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        z = (xt - mean) / scale
        z0 = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(z0.shape[1]) * alpha
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(z0.T @ z0 + penalty, z0.T @ yt)
        current = x.iloc[i].to_numpy()
        if np.isfinite(current).all():
            output.iloc[i] = np.r_[1.0, (current - mean) / scale] @ beta
    return output


def forecast_metrics(predictions: pd.DataFrame, realized_next_returns: pd.DataFrame) -> dict[str, float]:
    """Summarize pooled direction accuracy and rank correlation, not strategy P&L."""
    idx = predictions.index.intersection(realized_next_returns.index)
    p = predictions.loc[idx]
    y = realized_next_returns.loc[idx, p.columns]
    mask = p.notna() & y.notna()
    pv = p.where(mask).stack()
    yv = y.where(mask).stack()
    if len(pv) == 0:
        return {"n_forecasts": 0, "direction_accuracy": float("nan"), "pearson_ic": float("nan")}
    accuracy = float(((pv > 0) == (yv > 0)).mean())
    ic = float(pv.corr(yv)) if len(pv) > 1 else float("nan")
    return {"n_forecasts": int(len(pv)), "direction_accuracy": accuracy, "pearson_ic": ic}
