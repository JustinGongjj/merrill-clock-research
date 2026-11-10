import numpy as np
import pandas as pd
import pytest

from quantclock.prediction import forecast_metrics, walk_forward_ridge
from quantclock.rl import train_q_policy, rl_backtest


def test_ridge_prediction_is_nan_until_minimum_train():
    idx = pd.date_range("2000-01-31", periods=20, freq="ME")
    x = pd.DataFrame({"macro": np.arange(20, dtype=float)}, index=idx)
    r = pd.DataFrame({"asset": np.linspace(-.05, .05, 20)}, index=idx)
    pred = walk_forward_ridge(x, r, min_train=10, alpha=1.0)
    assert pred.iloc[:10].isna().all().all()
    assert pred.iloc[10:-1].notna().all().all()
    assert pred.iloc[-1].notna().all()  # valid live forecast; its future outcome is not yet in the sample


def test_q_learning_and_frozen_oos_policy():
    states = np.array([0, 1, 0, 1, 0, 1])
    returns = np.array([[.01, -.01], [-.02, .02], [.01, -.01], [-.01, .01], [.02, -.02], [0., 0.]])
    actions = np.array([[1., 0.], [0., 1.], [.5, .5]])
    policy = train_q_policy(states, returns, actions, n_states=2, episodes=20, seed=2)
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    oos = rl_backtest(policy, states, pd.DataFrame(returns, index=idx, columns=["a", "b"]))
    assert len(oos) == 6
    assert oos.iloc[0].turnover == 1.0
    assert np.isfinite(oos.net_return).all()


def test_forecast_metrics_counts_only_complete_pairs():
    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    predictions = pd.DataFrame({"a": [np.nan, .1, .2, -.1, .3], "b": [.1, np.nan, .2, .1, .0]}, index=idx)
    realized = pd.DataFrame({"a": [.01, .02, -.01, .01, .02], "b": [.01, .02, .01, -.01, .01]}, index=idx)
    metrics = forecast_metrics(predictions, realized)
    mask = predictions.notna().to_numpy() & realized.notna().to_numpy()
    assert metrics["n_forecasts"] == int(mask.sum()) == 8
    expected = float(((predictions.to_numpy()[mask] > 0) == (realized.to_numpy()[mask] > 0)).mean())
    assert metrics["direction_accuracy"] == pytest.approx(expected)
    assert np.isfinite(metrics["pearson_ic"])


def test_forecast_metrics_handles_no_complete_pairs():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    predictions = pd.DataFrame({"a": [np.nan] * 3}, index=idx)
    realized = pd.DataFrame({"a": [.01, .02, -.01]}, index=idx)
    metrics = forecast_metrics(predictions, realized)
    assert metrics["n_forecasts"] == 0
    assert np.isnan(metrics["direction_accuracy"])
