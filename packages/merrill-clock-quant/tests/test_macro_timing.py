import numpy as np
import pandas as pd

from quantclock.macro import (
    _accelerations_on_observation_grid,
    _monthly_observations,
    delayed_monthly_asof,
    make_clock_features,
)


def test_monthly_observation_only_available_after_configured_release_lag():
    obs = pd.Series([100.0, 110.0, 120.0], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    decisions = pd.to_datetime(["2024-01-31", "2024-02-28", "2024-03-31", "2024-04-30"])
    aligned = delayed_monthly_asof(obs, decisions, release_lag_days=20, name="x")
    assert pd.isna(aligned.iloc[0])
    assert aligned.iloc[1] == 100.0
    assert aligned.iloc[2] == 110.0
    assert aligned.iloc[3] == 120.0


def _sample_macro():
    idx = pd.date_range("2010-01-01", periods=60, freq="MS")
    t = np.arange(60, dtype=float)
    ip = pd.Series(100 * np.exp(0.004 * t + 0.02 * np.sin(t / 4)), index=idx, name="INDPRO")
    cpi = pd.Series(100 * np.exp(0.002 * t + 0.01 * np.cos(t / 6)), index=idx, name="CPIAUCSL")
    return ip, cpi


def test_differencing_happens_on_observation_months_not_decision_periods():
    ip, cpi = _sample_macro()
    daily_decisions = pd.date_range("2013-01-01", periods=365, freq="D")
    features = make_clock_features(ip, cpi, daily_decisions)
    grid = _accelerations_on_observation_grid(
        _monthly_observations(ip), _monthly_observations(cpi), 3, 3
    )
    seen = set(np.round(features["growth_accel"].dropna().to_numpy(), 10))
    expected = set(np.round(grid["growth_accel"].dropna().to_numpy(), 10))
    # A daily-grid diff(6) would invent hundreds of six-day "growth" values.
    assert seen
    assert seen.issubset(expected)
    assert len(seen) < 30


def test_daily_and_monthly_decision_grids_agree_on_the_same_date():
    ip, cpi = _sample_macro()
    date = pd.Timestamp("2014-06-30")
    monthly = make_clock_features(ip, cpi, pd.DatetimeIndex([date]))
    daily = make_clock_features(ip, cpi, pd.date_range("2014-06-01", periods=30, freq="D"))
    for column in ("growth_accel", "inflation_accel"):
        assert abs(float(monthly.iloc[0][column]) - float(daily.loc[date, column])) < 1e-12


def test_feature_values_are_stable_between_release_dates():
    ip, cpi = _sample_macro()
    daily = pd.date_range("2014-03-01", periods=10, freq="D")
    features = make_clock_features(ip, cpi, daily)
    assert features["growth_accel"].notna().all()
    assert features["growth_accel"].nunique() == 1
