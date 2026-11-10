"""Density cloud: a 2-D occupancy field over factor/asset space.

The cloud is the "where the mass is" view of the same data that drives the
node graph. Two constructions are provided:

* :func:`correlation_cloud` -- a kernel-smoothed joint density of two factor
  families (e.g. momentum vs volatility), so the diagonal ridge and the
  crowded corners become visible.
* :func:`regime_density_cloud` -- the trajectory of the market through
  growth/inflation space, drawn as a fading heat trail with the current
  position marked.

Both return a plain dict so the HTML renderer can draw them without any
computation of its own.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _grid(x: np.ndarray, y: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an (height, width) meshgrid covering the data extent."""
    if len(x) == 0:
        x_lo, x_hi = -1.0, 1.0
    else:
        x_lo, x_hi = float(np.nanmin(x)), float(np.nanmax(x))
    if len(y) == 0:
        y_lo, y_hi = -1.0, 1.0
    else:
        y_lo, y_hi = float(np.nanmin(y)), float(np.nanmax(y))
    pad_x = (x_hi - x_lo) * 0.05 or 0.5
    pad_y = (y_hi - y_lo) * 0.05 or 0.5
    xs = np.linspace(x_lo - pad_x, x_hi + pad_x, width)
    ys = np.linspace(y_lo - pad_y, y_hi + pad_y, height)
    return np.meshgrid(xs, ys)


def _edge_taper(n: int, plateau: float = 0.6) -> np.ndarray:
    """One-dimensional window: exactly 1.0 across the centre, 0.0 at the edges.

    The middle ``plateau`` fraction of the axis is left untouched so the core
    of a density cloud keeps its true intensity; only the outer band fades out
    with a raised cosine.
    """
    if n <= 1:
        return np.ones(n)
    x = np.linspace(0.0, 1.0, n)
    d = np.minimum(x, 1.0 - x) * 2.0          # 0 at the border, 1 at the centre
    edge = max(1e-6, (1.0 - plateau) / 2.0)   # normalised distance where the ramp ends
    ramp = np.clip(d / edge, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * ramp)  # 0 at the edge, 1 on the plateau


def _soften_edges(density: np.ndarray) -> np.ndarray:
    """Fade density toward the grid border so the cloud has no hard edge.

    A kernel density evaluated on a truncated grid shows a visible rectangular
    boundary wherever the data does not reach the edge. Multiplying by a
    cosine taper removes that artefact without touching the interior.
    """
    if density.size == 0:
        return density
    out = density.astype(float).copy()
    ny, nx = out.shape
    if nx > 1:
        out *= _edge_taper(nx)[None, :]
    if ny > 1:
        out *= _edge_taper(ny)[:, None]
    return out


def correlation_cloud(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    width: int = 64,
    height: int = 40,
    bandwidth: float | None = None,
    max_points: int = 4000,
) -> dict[str, Any]:
    """Kernel-smoothed 2-D density of two columns.

    Uses a Gaussian KDE evaluated on a regular grid (no SciPy dependency).
    Points are sub-sampled to ``max_points`` for speed; the sampling is
    deterministic (evenly spaced, not random) so the cloud is reproducible.
    """
    if frame is None or frame.empty:
        return {"available": False, "note": "没有可用样本生成云图"}
    if x_col not in frame.columns or y_col not in frame.columns:
        return {"available": False, "note": f"缺少列 {x_col} / {y_col}"}

    sample = frame[[x_col, y_col]].dropna()
    if len(sample) < 8:
        return {"available": False, "note": f"有效样本不足（{len(sample)} 行），无法估计密度"}
    if len(sample) > max_points:
        step = len(sample) // max_points + 1
        sample = sample.iloc[::step]

    x = sample[x_col].to_numpy(dtype=float)
    y = sample[y_col].to_numpy(dtype=float)
    gx, gy = _grid(x, y, width, height)

    # Bandwidth via Silverman's rule on the pooled spread.
    spread = float(np.nanstd(np.concatenate([x, y])))
    if spread <= 0:
        spread = 1.0
    sigma = bandwidth or (spread * (len(x) ** -0.2))
    if sigma <= 0:
        sigma = 1.0

    # Evaluate on a sub-sampled grid of points to keep the O(n*m) kernel bounded.
    point_step = max(1, len(x) // 600)
    xs, ys = x[::point_step], y[::point_step]
    diff_x = (gx.ravel()[:, None] - xs[None, :]) / sigma
    diff_y = (gy.ravel()[:, None] - ys[None, :]) / sigma
    density = np.exp(-0.5 * (diff_x ** 2 + diff_y ** 2)).sum(axis=1)
    density = density.reshape(gx.shape)
    if density.max() > 0:
        density = density / density.max()
    density = _soften_edges(density)
    if density.max() > 0:
        density = density / density.max()

    return {
        "available": True,
        "x_label": x_col,
        "y_label": y_col,
        "width": width,
        "height": height,
        "x_range": [float(gx.min()), float(gx.max())],
        "y_range": [float(gy.min()), float(gy.max())],
        "density": [[round(float(v), 4) for v in row] for row in density],
        "points": [
            [round(float(a), 5), round(float(b), 5)]
            for a, b in zip(x[::max(1, len(x) // 400)], y[::max(1, len(y) // 400)])
        ],
        "n_points": int(len(x)),
        "bandwidth": round(float(sigma), 6),
    }


def regime_density_cloud(
    features: pd.DataFrame,
    *,
    width: int = 56,
    height: int = 40,
    growth_col: str = "growth_accel",
    inflation_col: str = "inflation_accel",
    phase_col: str | None = "phase",
) -> dict[str, Any]:
    """Trajectory of the market through growth/inflation space.

    The heat trail fades with age so the most recent quarters are brightest;
    the current position and its phase are returned separately for the marker.
    """
    if features is None or features.empty:
        return {"available": False, "note": "没有宏观特征序列"}
    cols = [growth_col, inflation_col]
    if any(c not in features.columns for c in cols):
        return {"available": False, "note": f"缺少列 {cols}"}

    sample = features[cols].dropna()
    if len(sample) < 8:
        return {"available": False, "note": f"宏观样本不足（{len(sample)} 行）"}

    gx, gy = _grid(sample[growth_col].to_numpy(float), sample[inflation_col].to_numpy(float),
                   width, height)
    spread = float(np.nanstd(sample[cols].to_numpy(float).ravel()))
    sigma = spread * (len(sample) ** -0.2) or 1.0

    x = sample[growth_col].to_numpy(float)
    y = sample[inflation_col].to_numpy(float)
    dates = list(sample.index)
    n = len(x)

    # Time-weighted density: recent observations contribute more.
    weights = np.linspace(0.25, 1.0, n)
    step = max(1, n // 500)
    xs, ys, ws = x[::step], y[::step], weights[::step]
    diff_x = (gx.ravel()[:, None] - xs[None, :]) / sigma
    diff_y = (gy.ravel()[:, None] - ys[None, :]) / sigma
    density = (np.exp(-0.5 * (diff_x ** 2 + diff_y ** 2)) * ws[None, :]).sum(axis=1)
    density = density.reshape(gx.shape)
    if density.max() > 0:
        density = density / density.max()
    density = _soften_edges(density)
    if density.max() > 0:
        density = density / density.max()

    current = {"date": str(dates[-1].date()) if hasattr(dates[-1], "date") else str(dates[-1]),
               "growth": round(float(x[-1]), 6),
               "inflation": round(float(y[-1]), 6)}
    if phase_col and phase_col in features.columns:
        current["phase"] = str(features[phase_col].dropna().iloc[-1])

    return {
        "available": True,
        "x_label": "增长动能",
        "y_label": "通胀动能",
        "width": width,
        "height": height,
        "x_range": [float(gx.min()), float(gx.max())],
        "y_range": [float(gy.min()), float(gy.max())],
        "density": [[round(float(v), 4) for v in row] for row in density],
        "trajectory": [
            [round(float(a), 6), round(float(b), 6),
             round(float(weights[i]), 4), str(dates[i])[:10]]
            for i, (a, b) in enumerate(zip(x, y))
        ],
        "current": current,
        "n_points": int(n),
    }


def correlation_matrix(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Pearson correlation matrix as a plain dict (for a heat-map overlay)."""
    if frame is None or frame.empty:
        return {"available": False, "note": "没有可用样本"}
    use = [c for c in columns if c in frame.columns]
    if len(use) < 2:
        return {"available": False, "note": "至少需要两列可计算相关性"}
    corr = frame[use].dropna().corr()
    values = [[round(float(v), 4) for v in row] for row in corr.to_numpy()]
    return {
        "available": True,
        "labels": use,
        "matrix": values,
        "min": round(float(np.nanmin(corr.to_numpy())), 4),
        "max": round(float(np.nanmax(corr.to_numpy())), 4),
    }


def phase_transition_matrix(phase_series: pd.Series, *, lag: int = 1) -> dict[str, dict[str, float]]:
    """Observed phase-to-phase transition probabilities.

    ``phase_series`` maps decision dates to phase labels. This is an empirical
    frequency table -- it describes what actually happened, it does not predict
    what will.
    """
    if phase_series is None or len(phase_series) < 2:
        return {}
    seq = phase_series.dropna()
    counts: dict[str, dict[str, float]] = {}
    values = list(seq)
    for i in range(len(values) - lag):
        src, tgt = str(values[i]), str(values[i + lag])
        bucket = counts.setdefault(src, {})
        bucket[tgt] = bucket.get(tgt, 0.0) + 1.0
    matrix: dict[str, dict[str, float]] = {}
    for src, bucket in counts.items():
        total = sum(bucket.values())
        matrix[src] = {tgt: round(v / total, 4) for tgt, v in bucket.items()}
    return matrix


def hexbin_density(values: np.ndarray, bins: int = 24) -> dict[str, Any]:
    """Simple hexbin-style occupancy histogram for a 1-D series."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"available": False, "note": "空序列"}
    counts, edges = np.histogram(arr, bins=bins)
    centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    peak = int(counts.max()) if len(counts) else 0
    return {
        "available": True,
        "centres": [round(float(c), 6) for c in centres],
        "counts": [int(c) for c in counts],
        "peak": peak,
    }


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0
