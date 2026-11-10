import pandas as pd

from quantclock.macro import delayed_monthly_asof


def test_monthly_observation_only_available_after_configured_release_lag():
    obs = pd.Series([100.0, 110.0, 120.0], index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]))
    decisions = pd.to_datetime(["2024-01-31", "2024-02-28", "2024-03-31", "2024-04-30"])
    aligned = delayed_monthly_asof(obs, decisions, release_lag_days=20, name="x")
    assert pd.isna(aligned.iloc[0])
    assert aligned.iloc[1] == 100.0
    assert aligned.iloc[2] == 110.0
    assert aligned.iloc[3] == 120.0
