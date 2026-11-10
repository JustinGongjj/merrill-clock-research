"""Forward-looking projections for the constellation view.

Three projections are produced, each with an explicit statistical basis and an
explicit statement of what it is *not*:

1. :func:`project_phase_path` -- a Markov projection over the Merrill-clock
   phases using the empirical transition matrix. This is a *probability*, not a
   prediction; it says how often the clock moved this way historically.
2. :func:`project_factor_expectations` -- forward factor expectations from the
   measured IC/ICIR statistics, with a confidence band that widens with
   horizon.
3. :func:`project_price_paths` -- a bootstrap of forward price paths built
   from historical monthly returns, so the fan chart is honest about path
   dispersion rather than a single line pretending to know the future.

Every function returns plain dicts and carries a ``basis`` string describing
the data it came from, plus ``caveats`` listing what a reader must not
conclude from it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def project_phase_path(
    current_phase: str,
    transition_matrix: dict[str, dict[str, float]],
    *,
    horizon: int = 6,
) -> dict[str, Any]:
    """Propagate the phase distribution forward with the empirical matrix.

    The result is the probability of each phase at each future step, starting
    from a point mass on ``current_phase``.
    """
    if not transition_matrix:
        return {
            "available": False,
            "note": "没有可用的相位转移矩阵",
            "steps": [],
            "caveats": ["无历史转移样本，不做任何相位外推"],
        }
    phases = sorted({p for row in transition_matrix.values() for p in row})
    if current_phase not in phases:
        phases.append(current_phase)
    dist = {p: (1.0 if p == current_phase else 0.0) for p in phases}

    steps: list[dict[str, Any]] = []
    for h in range(1, horizon + 1):
        nxt: dict[str, float] = {p: 0.0 for p in phases}
        for src, mass in dist.items():
            if mass <= 0:
                continue
            row = transition_matrix.get(src, {})
            total = sum(row.values())
            if total <= 0:
                nxt[src] += mass          # absorbing: stays put if unobserved
                continue
            for tgt, prob in row.items():
                nxt[tgt] = nxt.get(tgt, 0.0) + mass * prob
        dist = nxt
        best = max(dist.items(), key=lambda kv: kv[1])
        steps.append({
            "step": h,
            "distribution": {k: round(float(v), 4) for k, v in sorted(dist.items())},
            "modal_phase": best[0],
            "modal_probability": round(float(best[1]), 4),
            "entropy": round(_entropy(dist), 4),
        })

    return {
        "available": True,
        "current_phase": current_phase,
        "horizon": horizon,
        "steps": steps,
        "basis": "相位转移矩阵由历史月末相位序列的实测转移频率估计得到",
        "caveats": [
            "这是历史频率的外推，不是对未来的预测，不含任何前瞻信息",
            "转移频率假设相位过程平稳，宏观制度切换时该假设不成立",
            "熵随步数上升表示不确定性累积；熵下降不代表预测更准",
        ],
    }


def project_factor_expectations(
    factor_stats: dict[str, dict[str, float]],
    *,
    horizon: int = 6,
    decay: float = 0.85,
) -> dict[str, Any]:
    """Forward IC expectation with a widening confidence band.

    The point forecast decays the measured IC toward zero (a factor's edge is
    assumed to fade); the band widens with the square root of the horizon,
    mirroring the sampling error of an IC estimated from N periods.
    """
    if not factor_stats:
        return {"available": False, "note": "没有因子统计量", "factors": []}
    out = []
    for name, stats in factor_stats.items():
        ic = float(stats.get("ic_mean", stats.get("rank_ic", 0.0)) or 0.0)
        n = int(stats.get("n_periods", 0) or 0)
        icir = float(stats.get("icir", 0.0) or 0.0)
        path = []
        for h in range(1, horizon + 1):
            point = ic * (decay ** h)
            se = (abs(ic) / math.sqrt(n)) if n > 0 else abs(ic)
            band = 1.96 * se * math.sqrt(h)
            path.append({
                "step": h,
                "expected_ic": round(point, 6),
                "lower": round(point - band, 6),
                "upper": round(point + band, 6),
            })
        out.append({
            "factor": name,
            "measured_ic": round(ic, 6),
            "icir": round(icir, 4),
            "n_periods": n,
            "path": path,
        })
    out.sort(key=lambda d: abs(d["measured_ic"]), reverse=True)
    return {
        "available": True,
        "horizon": horizon,
        "decay": decay,
        "factors": out,
        "basis": "以实测 RankIC 为起点，按 decay 逐期衰减，区间宽度按 √h 扩张",
        "caveats": [
            "假设因子 IC 逐期衰减，衰减速度为设定值而非估计值",
            "区间是抽样误差近似，不是预测区间，未覆盖结构突变",
            "IC 的横截面口径与本图的单标的暴露不完全等同",
        ],
    }


def project_price_paths(
    monthly_returns: pd.Series,
    *,
    horizon: int = 12,
    n_paths: int = 200,
    seed: int = 20261110,
    percentile_band: tuple[float, float] = (10.0, 90.0),
) -> dict[str, Any]:
    """Bootstrap forward price paths from historical monthly returns.

    The fan chart is the percentile band of the bootstrap distribution, so the
    widening is a property of the data (volatility clustering), not a cosmetic
    taper.
    """
    if monthly_returns is None or len(monthly_returns) < 12:
        return {"available": False, "note": "历史收益样本不足（至少需要 12 期）", "paths": []}
    series = monthly_returns.dropna()
    values = series.to_numpy(dtype=float)
    if len(values) < 12:
        return {"available": False, "note": "有效收益样本不足", "paths": []}

    rng = np.random.default_rng(seed)
    start_price = float(series.iloc[-1])
    draws = rng.choice(values, size=(n_paths, horizon), replace=True)
    # Compounded paths from the last observed price.
    paths = start_price * np.cumprod(1.0 + draws, axis=1)
    lo_q, hi_q = percentile_band
    lower = np.percentile(paths, lo_q, axis=0)
    upper = np.percentile(paths, hi_q, axis=0)
    median = np.percentile(paths, 50.0, axis=0)

    steps = []
    for h in range(horizon):
        steps.append({
            "step": h + 1,
            "median": round(float(median[h]), 6),
            "lower": round(float(lower[h]), 6),
            "upper": round(float(upper[h]), 6),
            "prob_up": round(float((paths[:, h] > start_price).mean()), 4),
        })

    return {
        "available": True,
        "horizon": horizon,
        "n_paths": n_paths,
        "start_price": round(start_price, 6),
        "sample_start": str(series.index[0])[:10],
        "sample_end": str(series.index[-1])[:10],
        "sample_n": int(len(values)),
        "steps": steps,
        "annualised_vol": round(float(np.std(values, ddof=1) * math.sqrt(12)), 6),
        "mean_monthly": round(float(np.mean(values)), 6),
        "basis": f"对历史 {len(values)} 期月度收益做有放回抽样，{n_paths} 条路径取分位数带",
        "caveats": [
            "自助抽样假设收益分布在未来不变，不含趋势与波动率聚集的动态",
            "分位带是分布区间，不是置信区间",
            "不构成投资建议，也不代表任何具体路径会发生",
        ],
    }


def signal_confidence(
    predictions: pd.Series,
    realized: pd.Series,
) -> dict[str, Any]:
    """Directional accuracy and rank correlation of a forecast vs reality."""
    idx = predictions.index.intersection(realized.index)
    if len(idx) == 0:
        return {"available": False, "note": "预测与实际没有重叠区间"}
    p = predictions.loc[idx].astype(float)
    y = realized.loc[idx].astype(float)
    mask = p.notna() & y.notna()
    p, y = p[mask], y[mask]
    if len(p) == 0:
        return {"available": False, "note": "重叠区间内没有有效值"}
    accuracy = float(((p > 0) == (y > 0)).mean())
    corr = float(p.corr(y)) if len(p) > 1 and p.std() > 0 and y.std() > 0 else float("nan")
    return {
        "available": True,
        "n": int(len(p)),
        "direction_accuracy": round(accuracy, 4),
        "pearson_ic": round(corr, 4) if math.isfinite(corr) else None,
    }


def _entropy(distribution: dict[str, float]) -> float:
    """Shannon entropy of a probability distribution (bits)."""
    values = [v for v in distribution.values() if v > 0]
    if not values:
        return 0.0
    total = sum(values)
    return -sum((v / total) * math.log2(v / total) for v in values)
