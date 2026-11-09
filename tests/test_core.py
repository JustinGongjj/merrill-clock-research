import numpy as np
import pandas as pd

from quantclock.backtest import run_backtest, performance_metrics, walk_forward_splits
from quantclock.signals import regime_probabilities, map_regimes_to_assets


def test_weights_apply_after_signal_date():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"EQ": [0.10, 0.20, -0.10]}, index=idx)
    weights = pd.DataFrame({"EQ": [1.0, 0.0, 1.0]}, index=idx)
    result = run_backtest(returns, weights, cost_bps=0)
    assert result.loc[idx[0], "net_return"] == 0
    assert result.loc[idx[1], "net_return"] == 0.20
    assert result.loc[idx[2], "net_return"] == 0


def test_regime_probabilities_sum_to_one():
    idx = pd.date_range("2000-01-31", periods=100, freq="ME")
    features = pd.DataFrame({"growth_accel": np.sin(np.arange(100)), "inflation_accel": np.cos(np.arange(100))}, index=idx)
    probs = regime_probabilities(features, window=20)
    valid = probs.dropna()
    assert np.allclose(valid.sum(axis=1), 1.0)
    assert list(probs.columns) == ["recovery", "overheat", "stagflation", "reflation"]


def test_mapping_normalizes_available_sleeves():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    probs = pd.DataFrame(0.25, index=idx, columns=["recovery", "overheat", "stagflation", "reflation"])
    weights = map_regimes_to_assets(probs, {"equity": "EQ", "treasury_intermediate": "BOND"})
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_walk_forward_splits_do_not_overlap_train_test():
    splits = list(walk_forward_splits(100, min_train=60, test_size=12))
    assert splits[0][0].stop == 60
    assert splits[0][1].start == 60
    assert splits[0][1].stop == 72


def test_metrics_are_finite_for_simple_series():
    result = performance_metrics(pd.Series([0.01, -0.01, 0.02]))
    assert np.isfinite(result["total_return"])
    assert result["max_drawdown"] <= 0
