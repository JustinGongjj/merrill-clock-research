"""A deliberately small offline tabular Q-learning research baseline.

This is educational experimentation, not a robust trading system. It learns on
one historical path, assumes actions are feasible portfolio weights, and cannot
make historical prices representative of future markets. Always compare with
non-RL baselines and an untouched walk-forward test set.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class QPolicy:
    q_values: np.ndarray
    action_weights: np.ndarray
    epsilon_final: float
    seed: int

    def choose(self, states: np.ndarray) -> np.ndarray:
        idx = np.asarray(states, dtype=int)
        return np.argmax(self.q_values[idx], axis=1)


def train_q_policy(
    states: np.ndarray,
    next_period_asset_returns: np.ndarray,
    action_weights: np.ndarray,
    n_states: int,
    episodes: int = 250,
    learning_rate: float = 0.08,
    gamma: float = 0.90,
    epsilon: float = 0.25,
    epsilon_min: float = 0.03,
    transaction_cost: float = 0.001,
    risk_penalty: float = 0.0,
    seed: int = 7,
) -> QPolicy:
    """Train Q-learning on chronological in-sample observations only.

    Row t's state is paired with row t's next-period asset returns. Callers must
    construct these arrays so state/decision precedes the subsequent return.
    Costs use one-way absolute weight turnover. No test data may be passed here.
    """
    s = np.asarray(states, dtype=int)
    r = np.asarray(next_period_asset_returns, dtype=float)
    actions = np.asarray(action_weights, dtype=float)
    if r.ndim != 2 or actions.ndim != 2 or actions.shape[1] != r.shape[1]:
        raise ValueError("returns and action_weights must be 2D with matching asset columns")
    if len(s) != len(r) or len(s) < 2:
        raise ValueError("states and next-period returns need same length >= 2")
    if s.min() < 0 or s.max() >= n_states:
        raise ValueError("state id outside n_states")
    if np.any(actions < 0) or not np.allclose(actions.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("actions must be long-only weights summing to one")
    rng = np.random.default_rng(seed)
    q = np.zeros((n_states, len(actions)), dtype=float)
    eps = epsilon
    for _ in range(episodes):
        prev_action = None
        for t in range(len(s) - 1):
            st = s[t]
            if rng.random() < eps:
                a = int(rng.integers(len(actions)))
            else:
                a = int(np.argmax(q[st]))
            turnover = float(np.abs(actions[a]).sum()) if prev_action is None else float(np.abs(actions[a] - actions[prev_action]).sum())
            portfolio_return = float(actions[a] @ r[t])
            reward = portfolio_return - transaction_cost * turnover - risk_penalty * portfolio_return ** 2
            next_st = s[t + 1]
            td_target = reward + gamma * float(np.max(q[next_st]))
            q[st, a] += learning_rate * (td_target - q[st, a])
            prev_action = a
        eps = max(epsilon_min, eps * 0.985)
    return QPolicy(q_values=q, action_weights=actions, epsilon_final=float(eps), seed=seed)


def rl_backtest(policy: QPolicy, states: np.ndarray, asset_returns: pd.DataFrame, transaction_cost: float = 0.001) -> pd.DataFrame:
    """Evaluate a frozen policy on chronological out-of-sample data only."""
    s = np.asarray(states, dtype=int)
    if len(s) != len(asset_returns):
        raise ValueError("states must align to asset_returns")
    action_idx = policy.choose(s)
    weights = policy.action_weights[action_idx]
    r = asset_returns.to_numpy(dtype=float)
    gross = np.sum(weights * r, axis=1)
    turnover = np.zeros(len(r), dtype=float)
    if len(r):
        turnover[0] = float(np.abs(weights[0]).sum())
    if len(r) > 1:
        turnover[1:] = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    net = gross - transaction_cost * turnover
    return pd.DataFrame({"action": action_idx, "gross_return": gross, "turnover": turnover, "net_return": net}, index=asset_returns.index)
