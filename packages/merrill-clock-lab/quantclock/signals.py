"""Transparent probability-based Merrill-clock quadrant signals."""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ("recovery", "overheat", "stagflation", "reflation")


def _rolling_zscore(s: pd.Series, window: int = 60, min_periods: int = 24) -> pd.Series:
    min_periods = min(window, min_periods)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
    return (s - mean) / std


def regime_probabilities(features: pd.DataFrame, window: int = 60, temperature: float = 1.0, smooth: int = 1) -> pd.DataFrame:
    """Estimate four quadrant probabilities from lagged standardized accelerations.

    Positive growth_accel means improving growth momentum; positive
    inflation_accel means rising inflation momentum. Probabilities are a smooth
    logistic decomposition of two scores, not calibrated forecasts.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    gz = _rolling_zscore(features["growth_accel"], window)
    iz = _rolling_zscore(features["inflation_accel"], window)
    pg = 1.0 / (1.0 + np.exp(-gz.clip(-20, 20) / temperature))
    pi = 1.0 / (1.0 + np.exp(-iz.clip(-20, 20) / temperature))
    probs = pd.DataFrame({
        "recovery": pg * (1 - pi),
        "overheat": pg * pi,
        "stagflation": (1 - pg) * pi,
        "reflation": (1 - pg) * (1 - pi),
    }, index=features.index)
    if smooth > 1:
        probs = probs.ewm(span=smooth, adjust=False, min_periods=1).mean()
    return probs.div(probs.sum(axis=1).replace(0, np.nan), axis=0)


def default_research_tilts() -> dict[str, dict[str, float]]:
    """Illustrative long-only research priors, not investment recommendations.

    Keys are semantic sleeve names; map these to the user's chosen liquid
    instruments and separately model cash, leverage, financing and costs.
    """
    return {
        "recovery": {"equity": .50, "treasury_intermediate": .30, "gold": .10, "broad_commodities": .10},
        "overheat": {"equity": .35, "treasury_intermediate": .15, "gold": .10, "broad_commodities": .40},
        "stagflation": {"cash": .50, "gold": .25, "broad_commodities": .25},
        "reflation": {"equity": .20, "treasury_intermediate": .60, "gold": .15, "broad_commodities": .05},
    }


def map_regimes_to_assets(probabilities: pd.DataFrame, asset_map: dict[str, str], tilts: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Blend regime sleeves into named instrument weights, then normalize.

    asset_map maps semantic sleeves (e.g. 'equity') to columns in returns.
    Missing sleeves are dropped and remaining weights re-normalized.
    """
    templates = tilts or default_research_tilts()
    assets = list(dict.fromkeys(asset_map.values()))
    out = pd.DataFrame(0.0, index=probabilities.index, columns=assets)
    for regime in REGIMES:
        if regime not in probabilities:
            raise ValueError(f"Missing regime probability {regime!r}")
        p = probabilities[regime].fillna(0.0)
        for sleeve, weight in templates[regime].items():
            ticker = asset_map.get(sleeve)
            if ticker in out.columns:
                out[ticker] += p * float(weight)
    denom = out.sum(axis=1).replace(0, np.nan)
    return out.div(denom, axis=0)
