"""Descriptive calibration diagnostics for the four-phase probability signal.

``regime_probabilities`` produces a smooth logistic decomposition of two
macro scores. It is explicitly *not* a calibrated forecast. These diagnostics
make that limitation measurable: Brier score against the next observation's
growth/inflation direction, quadrant hit rate versus a naive constant
baseline, mean entropy (how decisive the signal is) and regime persistence.

Important: macro features in the default pipeline come from latest-revised
FRED series. Evaluating them against later observations is a descriptive
consistency check, not a real-time forecast test.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

REGIMES = ("recovery", "overheat", "stagflation", "reflation")


def brier_score(probability: pd.Series, outcome: pd.Series) -> float:
    """Mean squared error of a probability against a 0/1 outcome."""
    p = pd.Series(probability).astype(float)
    y = pd.Series(outcome).astype(float)
    common = p.index.intersection(y.index)
    frame = pd.concat([p.loc[common], y.loc[common]], axis=1).dropna()
    if frame.empty:
        return float("nan")
    return float(((frame.iloc[:, 0] - frame.iloc[:, 1]) ** 2).mean())


def _mean_run_length(labels: list[str]) -> float:
    if not labels:
        return float("nan")
    lengths: list[int] = []
    current, run = labels[0], 0
    for label in labels:
        if label == current:
            run += 1
        else:
            lengths.append(run)
            current, run = label, 1
    lengths.append(run)
    return float(np.mean(lengths))


def realized_quadrant(growth_up: bool, inflation_up: bool) -> str:
    if growth_up and not inflation_up:
        return "recovery"
    if growth_up and inflation_up:
        return "overheat"
    if not growth_up and inflation_up:
        return "stagflation"
    return "reflation"


def phase_diagnostics(
    probabilities: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Compare phase probabilities with the *next* macro observation.

    ``features`` must contain ``growth_accel`` and ``inflation_accel`` on the
    same index as ``probabilities``. The next row's signs form the realized
    quadrant. The last row has no successor and is excluded.
    """
    required = {"growth_accel", "inflation_accel"}
    if not required.issubset(features.columns):
        raise ValueError(f"features must contain {sorted(required)}")
    missing = [r for r in REGIMES if r not in probabilities.columns]
    if missing:
        raise ValueError(f"probabilities missing regime columns: {missing}")
    common = probabilities.index.intersection(features.index)
    p = probabilities.loc[common, list(REGIMES)].astype(float)
    f = features.loc[common, ["growth_accel", "inflation_accel"]].astype(float)
    if len(common) < 3:
        raise ValueError("Need at least three aligned observations for phase diagnostics")

    next_growth = f["growth_accel"].shift(-1)
    next_inflation = f["inflation_accel"].shift(-1)
    valid = next_growth.notna() & next_inflation.notna() & p.notna().all(axis=1)
    p = p.loc[valid]
    growth_up = (next_growth.loc[valid] > 0)
    inflation_up = (next_inflation.loc[valid] > 0)
    if p.empty:
        return {"n_periods": 0, "note": "no complete next-period observations"}

    pg = p["recovery"] + p["overheat"]
    pi = p["overheat"] + p["stagflation"]
    predicted = p.idxmax(axis=1)
    realized = pd.Series(
        [realized_quadrant(bool(g), bool(i)) for g, i in zip(growth_up, inflation_up)],
        index=p.index, dtype=object,
    )
    hit = float((predicted == realized).mean())
    baseline_label = realized.value_counts().idxmax()
    baseline_hit = float((realized == baseline_label).mean())

    row_sums = p.sum(axis=1).replace(0, np.nan)
    normalized = p.div(row_sums, axis=0)
    entropy = -(normalized * np.log(normalized.replace(0, np.nan))).fillna(0.0).sum(axis=1)
    entropy = entropy / math.log(len(REGIMES))
    dominant = list(predicted)

    return {
        "n_periods": int(len(p)),
        "brier_growth_up": brier_score(pg, growth_up.astype(float)),
        "brier_inflation_up": brier_score(pi, inflation_up.astype(float)),
        "quadrant_hit_rate": hit,
        "constant_baseline_label": str(baseline_label),
        "constant_baseline_hit_rate": baseline_hit,
        "hit_rate_minus_baseline": hit - baseline_hit,
        "mean_normalized_entropy": float(entropy.mean()),
        "mean_dominant_regime_duration_periods": _mean_run_length(dominant),
        "regime_share": {k: float((predicted == k).mean()) for k in REGIMES},
        "caveats": [
            "相位概率为 logistic 分解，不是标定过的预测概率；Brier 只作描述性对照",
            "宏观特征来自最新修订值，下一期观测也不是点时数据，不构成实时预测检验",
            "四相均匀分布的命中率参考值为 0.25；与常数基线的差值不代表统计显著",
        ],
    }


def temperature_sensitivity(
    features: pd.DataFrame,
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0),
    *,
    window: int = 60,
    smooth: int = 1,
) -> dict[str, Any]:
    """Show how signal decisiveness changes with the logistic temperature.

    This is a sensitivity diagnostic, not parameter selection: no objective is
    optimized and no best value is declared.
    """
    from .signals import regime_probabilities

    results: dict[str, Any] = {}
    for temperature in temperatures:
        probs = regime_probabilities(features, window=window,
                                     temperature=temperature, smooth=smooth)
        max_prob = probs.max(axis=1).dropna()
        results[str(temperature)] = {
            "mean_max_probability": float(max_prob.mean()) if not max_prob.empty else float("nan"),
            "mean_entropy": float(phase_diagnostics(probs, features).get(
                "mean_normalized_entropy", float("nan"))),
        }
    return {"window": window, "smooth": smooth, "by_temperature": results}
