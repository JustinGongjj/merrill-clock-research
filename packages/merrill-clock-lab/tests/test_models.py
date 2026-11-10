import numpy as np
import pandas as pd

from quantclock.prediction import walk_forward_ridge
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
