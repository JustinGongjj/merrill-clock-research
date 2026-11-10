"""Factor synthesis / weighting methods, kept explicit and reproducible.

Public research typically compares equal weight, historical-IC half-life
weighting, maximized IC_IR and PCA; the short-weight solution ``Sigma^-1 IC``
often produces negative weights, so a non-negativity projection is available
and is the default. Window length is a real research parameter, not a hidden
constant -- short windows cannot estimate an IC covariance matrix reliably.

This module only constructs weights. It does not claim any of them is optimal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

METHODS = ("equal", "ic_half_life", "max_ic_ir", "pca")


def half_life_weights(n_periods: int, half_life: float = 12.0) -> np.ndarray:
    """``w_t = 2^((t-T-1)/H)`` for t=1..T, normalized to sum to one.

    Moving back H periods halves the weight. H > 0, n_periods >= 1.
    """
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    t = np.arange(1, n_periods + 1, dtype=float)
    weights = 2.0 ** ((t - n_periods - 1) / half_life)
    total = weights.sum()
    return weights / total if total > 0 else np.full(n_periods, 1.0 / n_periods)


def _project_non_negative(weights: np.ndarray) -> np.ndarray:
    """Clip short weights to zero and renormalize (documented simplification)."""
    clipped = np.clip(weights, 0.0, None)
    total = clipped.sum()
    if total <= 0:
        return np.full(len(weights), 1.0 / len(weights))
    return clipped / total


def _sign_align(vector: np.ndarray) -> np.ndarray:
    return vector if vector.sum() >= 0 else -vector


def synthesize_weights(
    ic_history: pd.DataFrame,
    method: str = "equal",
    *,
    window: int = 12,
    half_life: float = 12.0,
    allow_short: bool = False,
) -> dict[str, Any]:
    """Compute composite-factor weights from a history of per-factor IC values.

    ``ic_history`` is date x factor. Only the latest ``window`` rows are used.
    Columns with no finite history get zero weight (and are listed in warnings).
    """
    if not isinstance(ic_history, pd.DataFrame) or ic_history.empty:
        raise ValueError("ic_history must be a non-empty DataFrame (date x factor)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if window < 2:
        raise ValueError("window must be >= 2")
    history = ic_history.tail(window).astype(float)
    names = list(history.columns)
    n = len(names)
    warnings: list[str] = []
    finite = history.notna().sum(axis=0)

    if method == "equal":
        raw = np.ones(n, dtype=float)

    elif method == "ic_half_life":
        w = half_life_weights(len(history), half_life)
        # Treat a missing factor as zero expected IC rather than equal weight.
        raw = np.array([
            float(np.dot(np.nan_to_num(history[c].to_numpy(), nan=0.0), w))
            if finite[c] else 0.0 for c in names
        ], dtype=float)
        raw = np.where(raw >= 0, raw, 0.0) if not allow_short else raw
        if np.allclose(raw, 0.0):
            raw = np.ones(n, dtype=float)
            warnings.append("所有因子半衰 IC 均非正，退化为等权")

    elif method == "max_ic_ir":
        filled = history.fillna(0.0).to_numpy(dtype=float)
        mu = filled.mean(axis=0)
        centered = filled - mu
        cov = centered.T @ centered / max(len(history) - 1, 1)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        raw = np.linalg.pinv(cov) @ mu
        if len(history) < n + 2:
            warnings.append("窗口长度不足，IC 协方差矩阵估计不稳，最大化 IC_IR 权重需谨慎解释")

    else:  # pca
        filled = history.fillna(0.0).to_numpy(dtype=float)
        std = filled.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        standardized = (filled - filled.mean(axis=0)) / std
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        raw = _sign_align(vt[0]) if len(vt) else np.ones(n, dtype=float)

    raw = np.nan_to_num(np.asarray(raw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    for name, count in finite.items():
        if count == 0:
            raw[names.index(name)] = 0.0
            warnings.append(f"因子 {name} 在窗口内没有有效 IC 观测，权重为 0")
    if not allow_short:
        raw = _project_non_negative(raw)
    else:
        total = float(np.abs(raw).sum())
        raw = raw / total if total > 0 else np.full(n, 1.0 / n)
    weights = {name: float(w) for name, w in zip(names, raw)}
    return {
        "method": method,
        "window_used": int(len(history)),
        "half_life": float(half_life) if method == "ic_half_life" else None,
        "allow_short": bool(allow_short),
        "weights": weights,
        "weight_sum": float(sum(weights.values())),
        "warnings": warnings,
    }


def composite_factor(
    factor_frames: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Weighted sum of aligned factor frames (missing cells treated as 0)."""
    if not factor_frames:
        raise ValueError("factor_frames must not be empty")
    frames = [f.astype(float) for f in factor_frames.values() if f is not None]
    if not frames:
        raise ValueError("factor_frames contains no usable frames")
    dates = frames[0].index
    columns = frames[0].columns
    for frame in frames[1:]:
        dates = dates.intersection(frame.index)
        columns = columns.intersection(frame.columns)
    if len(dates) == 0 or len(columns) == 0:
        raise ValueError("factor frames need overlapping dates and assets")
    total = sum(abs(v) for v in weights.values())
    if total <= 0:
        raise ValueError("weights sum to zero")
    out = pd.DataFrame(0.0, index=dates, columns=columns, dtype=float)
    for name, frame in factor_frames.items():
        out += frame.loc[dates, columns].fillna(0.0).astype(float) * float(weights.get(name, 0.0))
    return out


def weight_stability(weight_sequence: list[dict[str, float]]) -> dict[str, float]:
    """Average L1 turnover between consecutive weight vectors."""
    if len(weight_sequence) < 2:
        return {"n_changes": 0, "mean_l1_turnover": float("nan")}
    names = sorted(set().union(*(set(w) for w in weight_sequence)))
    changes: list[float] = []
    for prev, cur in zip(weight_sequence[:-1], weight_sequence[1:]):
        a = np.array([prev.get(k, 0.0) for k in names], dtype=float)
        b = np.array([cur.get(k, 0.0) for k in names], dtype=float)
        changes.append(float(np.abs(b - a).sum()))
    return {"n_changes": len(changes), "mean_l1_turnover": float(np.mean(changes))}
