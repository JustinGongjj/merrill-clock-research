"""Offline end-to-end sanity check for the Merrill-clock research pipeline.

No network, no market data and no installed model weights. Synthetic asset
returns are deliberately built from the *previous* period's macro features, so
this script proves the code path (macro -> features -> probabilities -> weights
-> next-period backtest -> metrics), not that the strategy works. Any performance
number printed here is a property of the synthetic generator, not evidence of
predictability or investability.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from quantclock.backtest import performance_metrics, run_backtest
from quantclock.macro import make_clock_features
from quantclock.signals import default_research_tilts, map_regimes_to_assets, regime_probabilities

ASSET_MAP = {
    "equity": "EQ",
    "treasury_intermediate": "BOND",
    "gold": "GOLD",
    "broad_commodities": "COMM",
    "cash": "CASH",
}


def build_synthetic_dataset(seed: int = 2026, periods: int = 240):
    """Return (industrial_production, cpi, asset_returns, clock_features)."""
    rng = np.random.default_rng(seed)
    # Extra leading months feed the 12-month CPI change and the release lag.
    macro_idx = pd.date_range("1998-01-01", periods=periods + 24, freq="MS")
    t = np.arange(len(macro_idx), dtype=float)
    industrial_production = pd.Series(
        100.0 * np.exp(0.0018 * t + 0.020 * np.sin(2 * np.pi * t / 48.0)), index=macro_idx, name="INDPRO"
    )
    cpi = pd.Series(
        100.0 * np.exp(0.0022 * t + 0.012 * np.sin(2 * np.pi * t / 60.0 + 1.0)), index=macro_idx, name="CPIAUCSL"
    )
    decision_idx = pd.date_range("2000-01-31", periods=periods, freq="ME")
    features = make_clock_features(industrial_production, cpi, decision_idx)
    # Lagged features drive the next holding period, matching run_backtest's
    # one-period shift. This is a self-referential generator, not market data.
    g = features["growth_accel"].fillna(0.0).clip(-3, 3).shift(1).fillna(0.0) / 3.0
    i = features["inflation_accel"].fillna(0.0).clip(-3, 3).shift(1).fillna(0.0) / 3.0
    n = len(decision_idx)
    returns = pd.DataFrame({
        "EQ": 0.006 + 0.020 * g + rng.normal(0.0, 0.040, n),
        "BOND": 0.003 - 0.015 * i + rng.normal(0.0, 0.020, n),
        "GOLD": 0.002 + 0.012 * i + rng.normal(0.0, 0.030, n),
        "COMM": 0.004 + 0.018 * i + 0.010 * g + rng.normal(0.0, 0.050, n),
        "CASH": np.full(n, 0.002),
    }, index=decision_idx)
    return industrial_production, cpi, returns, features


def run_demo(seed: int = 2026, periods: int = 240, cost_bps: float = 10.0) -> dict:
    _, _, returns, features = build_synthetic_dataset(seed=seed, periods=periods)
    probabilities = regime_probabilities(features, window=36, smooth=3)
    weights = map_regimes_to_assets(probabilities, ASSET_MAP, default_research_tilts())
    result = run_backtest(returns, weights, cost_bps=cost_bps)
    invested = weights.sum(axis=1).shift(1).fillna(0.0) > 0
    if not invested.any():
        raise RuntimeError("No valid weights: extend the synthetic history or lower the z-window")
    strategy = performance_metrics(result.loc[invested, "net_return"])
    core = ["EQ", "BOND", "GOLD", "COMM"]
    benchmark = performance_metrics(returns[core].loc[invested].mean(axis=1))
    return {
        "synthetic": True,
        "periods": int(periods),
        "evaluated_months": int(invested.sum()),
        "warmup_months_excluded": int((~invested).sum()),
        "sample_start": str(result.loc[invested].index.min().date()),
        "sample_end": str(result.loc[invested].index.max().date()),
        "strategy": strategy,
        "equal_weight_benchmark_no_costs": benchmark,
        "cost_bps_per_one_way_turnover": float(cost_bps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline synthetic Merrill-clock pipeline check")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--periods", type=int, default=240)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    report = run_demo(seed=args.seed, periods=args.periods, cost_bps=args.cost_bps)
    print("Deterministic synthetic data only: no network, no real prices, no strategy claim.")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
