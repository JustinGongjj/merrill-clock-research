import pandas as pd

from quantclock.futures import front_month_returns


def test_roll_uses_new_contract_return_not_calendar_spread():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({
        "F1": [100, 101, 102, 103],
        "F2": [110, 111, 112, 113],
    }, index=dates)
    returns, active = front_month_returns(prices, {"F1": "2024-01-04", "F2": "2024-02-04"}, roll_days_before=2, roll_cost_bps=10)
    # F2 becomes active before F1 first notice; roll return is F2's 111/110 - 1, less 10bp.
    assert active.iloc[0] == "F1"
    assert active.iloc[1] == "F2"
    assert abs(returns.iloc[1] - (111 / 110 - 1 - .001)) < 1e-12
