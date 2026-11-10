"""Cross-sectional single-factor testing: the missing half of this repository.

``factorlab.regression`` answers "which known factors explain my strategy's
return" (time-series attribution). This module answers the different question
that broker factor desks actually test: "does this cross-sectional signal rank
next-period returns?" It implements the three-way check used in public
research -- RankIC/ICIR, Fama-MacBeth slope t-stat, and layered monotonicity --
plus the standard preprocessing chain (MAD winsorize, z-score, industry/size
neutralization, missing-as-neutral).

Conventions
-----------
* ``factor`` and ``forward_returns`` are wide frames: index = decision date t,
  columns = assets, values = factor exposure at t / return from t to t+1.
* All statistics are computed on the cross section available at each date; no
  forward information enters a factor's own row.
* ``min_assets`` guards against meaningless correlations on tiny universes.
* Nothing here is a return forecast or an investment recommendation.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

MAD_TO_SIGMA = 1.4826


def _label(value: Any) -> str:
    """Stable string label for a date index entry (Timestamp, str or int)."""
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def _check_aligned(factor: pd.DataFrame, forward: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(factor, pd.DataFrame) or not isinstance(forward, pd.DataFrame):
        raise TypeError("factor and forward_returns must be wide DataFrames (date x asset)")
    common_dates = factor.index.intersection(forward.index)
    common_assets = factor.columns.intersection(forward.columns)
    if len(common_dates) == 0 or len(common_assets) == 0:
        raise ValueError("factor and forward_returns need overlapping dates and assets")
    return (factor.loc[common_dates, common_assets].astype(float).sort_index(),
            forward.loc[common_dates, common_assets].astype(float).sort_index())


def winsorize_mad(frame: pd.DataFrame, n_mad: float = 5.0) -> pd.DataFrame:
    """Cross-sectional median/MAD winsorization, one row at a time.

    Values beyond ``median +/- n_mad * 1.4826 * MAD`` are pulled back to the
    boundary. A zero MAD (a nearly constant cross section) is left untouched.
    """
    if n_mad <= 0:
        raise ValueError("n_mad must be positive")
    out = frame.copy().astype(float)
    for date in out.index:
        row = out.loc[date]
        values = row.dropna()
        if values.empty:
            continue
        median = float(values.median())
        mad = float((values - median).abs().median()) * MAD_TO_SIGMA
        if mad <= 0 or not math.isfinite(mad):
            continue
        lower, upper = median - n_mad * mad, median + n_mad * mad
        out.loc[date] = row.clip(lower=lower, upper=upper)
    return out


def zscore(frame: pd.DataFrame, fill_missing: float | None = 0.0) -> pd.DataFrame:
    """Cross-sectional standardization. Missing values become the neutral 0.

    ``fill_missing=None`` keeps NaNs, which is useful when the caller needs to
    know exactly which assets had no observation.
    """
    out = frame.copy().astype(float)
    for date in out.index:
        row = out.loc[date]
        values = row.dropna()
        if values.empty:
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        scaled = (row - mean) / std if std > 0 else row - mean
        out.loc[date] = scaled
    if fill_missing is not None:
        out = out.fillna(fill_missing)
    return out


def neutralize(
    factor: pd.DataFrame,
    industry: pd.DataFrame | None = None,
    size: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Remove industry and size exposure with a per-date cross-sectional OLS.

    Industry dummies use one level as the baseline (drop-first), size is a
    numeric column. Rank-deficient designs are handled by ``lstsq``. Dates
    with too few complete observations are returned unchanged (not silently
    filled), and the caller can inspect them via ``neutralize_report``.
    """
    out = factor.copy().astype(float)
    for date in out.index:
        y = out.loc[date].astype(float)
        columns: list[pd.Series] = []
        if size is not None and date in size.index:
            s = size.loc[date].reindex(out.columns).astype(float)
            s = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1.0)
            columns.append(s.rename("size"))
        if industry is not None and date in industry.index:
            labels = industry.loc[date].reindex(out.columns)
            dummies = pd.get_dummies(labels, prefix="ind", dummy_na=False, dtype=float)
            if dummies.shape[1] > 1:
                columns.append(dummies.iloc[:, 1:])
        if not columns:
            continue
        design = pd.concat([pd.Series(1.0, index=out.columns, name="const")] + columns, axis=1)
        valid = y.notna() & design.notna().all(axis=1)
        if int(valid.sum()) <= design.shape[1]:
            continue
        x = design.loc[valid].to_numpy(dtype=float)
        target = y.loc[valid].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(x, target, rcond=None)
        residual = pd.Series(np.nan, index=out.columns, dtype=float)
        residual.loc[valid] = target - x @ beta
        out.loc[date] = residual
    return out


def rank_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_assets: int = 5,
) -> pd.Series:
    """Per-date Spearman rank correlation between factor and next-period return."""
    f, r = _check_aligned(factor, forward_returns)
    if min_assets < 2:
        raise ValueError("min_assets must be >= 2")
    values: list[float] = []
    for date in f.index:
        frow, rrow = f.loc[date], r.loc[date]
        mask = frow.notna() & rrow.notna()
        if int(mask.sum()) < min_assets:
            values.append(float("nan"))
            continue
        fv, rv = frow[mask].rank(method="average"), rrow[mask].rank(method="average")
        if fv.std(ddof=0) == 0 or rv.std(ddof=0) == 0:
            values.append(float("nan"))
            continue
        values.append(float(np.corrcoef(fv.to_numpy(), rv.to_numpy())[0, 1]))
    return pd.Series(values, index=f.index, name="rank_ic")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean, dispersion, ICIR, win rate and a simple t-stat on the IC series.

    The t-stat treats IC observations as independent; serial correlation is not
    corrected here. Use HAC errors on the regression track for formal testing.
    """
    clean = pd.Series(ic).dropna().astype(float)
    if clean.empty:
        return {"n_periods": 0, "ic_mean": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "positive_ratio": float("nan"), "t_stat": float("nan")}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")
    n = len(clean)
    # A constant IC series has a floating-point std near 1e-17, not exactly 0;
    # dividing by it would report a meaningless huge ICIR / t-stat.
    usable_std = bool(std == std and std > 1e-12)
    ir = mean / std if usable_std else float("nan")
    t = mean / (std / math.sqrt(n)) if usable_std else float("nan")
    return {
        "n_periods": int(n),
        "ic_mean": mean,
        "ic_std": std,
        "icir": float(ir) if ir == ir else float("nan"),
        "positive_ratio": float((clean > 0).mean()),
        "t_stat": float(t) if t == t else float("nan"),
    }


def layered_returns(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_groups: int = 5,
    min_assets: int = 5,
) -> pd.DataFrame:
    """Average forward return per factor quantile group.

    Ranking uses ``rank(method='first')`` before bucketing so tied factor
    values cannot collapse a group. Group 1 = lowest factor values.
    """
    f, r = _check_aligned(factor, forward_returns)
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    records: list[dict[str, float]] = []
    for date in f.index:
        frow, rrow = f.loc[date], r.loc[date]
        mask = frow.notna() & rrow.notna()
        if int(mask.sum()) < max(min_assets, n_groups):
            continue
        ranked = frow[mask].rank(method="first")
        groups = pd.qcut(ranked, q=n_groups, labels=False)
        row: dict[str, float] = {"date_group_count": int(n_groups)}
        for g in range(n_groups):
            row[f"g{g + 1}"] = float(rrow[mask][groups == g].mean())
        records.append(row)
    if not records:
        return pd.DataFrame(columns=[f"g{i + 1}" for i in range(n_groups)])
    return pd.DataFrame(records)


def layer_monotonicity(layers: pd.DataFrame) -> dict[str, Any]:
    """Classify the group-return shape: monotone / middle-best / mixed."""
    columns = [c for c in layers.columns if c.startswith("g")]
    if not columns or layers.empty:
        return {"group_means": {}, "spearman": float("nan"),
                "is_monotonic": False, "middle_best": False, "shape": "insufficient_data"}
    means = [float(layers[c].mean()) for c in columns]
    ranks = pd.Series(range(1, len(means) + 1), dtype=float)
    spearman = float(np.corrcoef(ranks, pd.Series(means).rank(method="average"))[0, 1]) \
        if pd.Series(means).std(ddof=0) > 0 else float("nan")
    diffs = np.diff(means)
    increasing = bool((diffs >= 0).all() and (diffs > 0).any())
    decreasing = bool((diffs <= 0).all() and (diffs < 0).any())
    middle_best = bool(max(means) not in (means[0], means[-1]))
    shape = "monotonic_increasing" if increasing else (
        "monotonic_decreasing" if decreasing else (
            "nonlinear_middle_best" if middle_best else "mixed"))
    return {
        "group_means": {c: float(m) for c, m in zip(columns, means)},
        "spearman": spearman,
        "is_monotonic": increasing or decreasing,
        "middle_best": middle_best,
        "shape": shape,
        "note": ("中间组优于头尾说明存在非线性，IC 与线性回归会低估甚至误判这类因子"
                 if middle_best else ""),
    }


def fama_macbeth(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_assets: int = 5,
) -> dict[str, Any]:
    """Per-date cross-sectional regression of forward return on standardized factor.

    Returns the slope series plus its mean, t-stat and positive share. The slope
    is the factor return for that period; the t-stat is computed on the slope
    series (not corrected for autocorrelation).
    """
    f, r = _check_aligned(factor, forward_returns)
    slopes: list[float] = []
    for date in f.index:
        frow, rrow = f.loc[date], r.loc[date]
        mask = frow.notna() & rrow.notna()
        if int(mask.sum()) < min_assets:
            slopes.append(float("nan"))
            continue
        x = frow[mask].to_numpy(dtype=float)
        y = rrow[mask].to_numpy(dtype=float)
        xc = x - x.mean()
        denom = float(xc @ xc)
        slopes.append(float(xc @ (y - y.mean()) / denom) if denom > 0 else float("nan"))
    series = pd.Series(slopes, index=f.index, name="fm_slope").dropna()
    if series.empty:
        return {"n_periods": 0, "slope_mean": float("nan"), "t_stat": float("nan"),
                "positive_ratio": float("nan"), "slopes": {}}
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    t = float(series.mean() / (std / math.sqrt(len(series)))) if std and std > 0 else float("nan")
    return {
        "n_periods": int(len(series)),
        "slope_mean": float(series.mean()),
        "t_stat": t,
        "positive_ratio": float((series > 0).mean()),
        "slopes": {_label(k): float(v) for k, v in series.items()},
    }


def evaluate_factor(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    industry: pd.DataFrame | None = None,
    size: pd.DataFrame | None = None,
    n_mad: float = 5.0,
    n_groups: int = 5,
    min_assets: int = 5,
    preprocess: bool = True,
) -> dict[str, Any]:
    """Run the full three-way single-factor check and return a JSON-ready report."""
    f, r = _check_aligned(factor, forward_returns)
    applied: list[str] = []
    if preprocess:
        f = winsorize_mad(f, n_mad=n_mad)
        applied.append(f"MAD winsorize (n_mad={n_mad})")
        f = zscore(f, fill_missing=None)
        applied.append("cross-sectional z-score (missing kept as NaN)")
        if industry is not None or size is not None:
            f = neutralize(f, industry=industry, size=size)
            applied.append("industry/size neutralization by cross-sectional OLS")
        f = zscore(f, fill_missing=0.0)
        applied.append("post-neutralization z-score; missing set to neutral 0")
    ic = rank_ic(f, r, min_assets=min_assets)
    layers = layered_returns(f, r, n_groups=n_groups, min_assets=min_assets)
    fm = fama_macbeth(f, r, min_assets=min_assets)
    summary = ic_summary(ic)
    return {
        "n_dates": int(len(f.index)),
        "n_assets": int(len(f.columns)),
        "sample_start": _label(f.index.min()),
        "sample_end": _label(f.index.max()),
        "preprocessing": applied,
        "ic_summary": summary,
        "rank_ic_series": {_label(k): (None if pd.isna(v) else float(v))
                           for k, v in ic.items()},
        "layered_returns": ({} if layers.empty else
                            {c: float(layers[c].mean()) for c in layers.columns if c.startswith("g")}),
        "layer_monotonicity": layer_monotonicity(layers),
        "fama_macbeth": fm,
        "caveats": [
            "IC t 值与 Fama-MacBeth t 值未做序列相关校正，仅作描述性统计",
            "所有统计量只描述该历史样本，不是预测值，也不代表因果效应",
        ],
    }
