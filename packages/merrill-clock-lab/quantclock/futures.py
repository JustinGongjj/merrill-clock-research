"""Contract-level futures rolling utilities; never infer roll rules from Yahoo tickers."""
from __future__ import annotations
import pandas as pd


def front_month_returns(
    contract_settles: pd.DataFrame,
    first_notice_dates: dict[str, str | pd.Timestamp],
    roll_days_before: int = 5,
    roll_cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Build a front-month total-return proxy from settlement prices by contract.

    Columns must be actual contract identifiers; dates are trading dates.
    Contract selection is the nearest contract whose first-notice date is more
    than ``roll_days_before`` calendar days away. Returns on a roll date use the
    new contract's own prior-day settle, so the calendar-spread price gap is not
    misreported as a profit/loss. A fixed roll cost is charged when the selected
    contract changes. A production backtest should model exact contract sizes,
    margin, commissions, slippage, exchange calendars, and roll execution time.
    """
    if roll_days_before < 0 or roll_cost_bps < 0:
        raise ValueError("roll_days_before and roll_cost_bps must be non-negative")
    prices = contract_settles.sort_index().copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    expiry = {k: pd.Timestamp(v).tz_localize(None) for k, v in first_notice_dates.items()}
    candidates = [c for c in prices.columns if c in expiry]
    if not candidates:
        raise ValueError("No contract columns match first_notice_dates")
    active = []
    for date in prices.index:
        eligible = [c for c in candidates if expiry[c].normalize() > date.normalize() + pd.Timedelta(days=roll_days_before)
                    and pd.notna(prices.at[date, c])]
        if not eligible:
            active.append(None)
            continue
        active.append(min(eligible, key=lambda c: expiry[c]))
    contract = pd.Series(active, index=prices.index, name="active_contract")
    returns = pd.Series(float("nan"), index=prices.index, name="return")
    switched = contract.ne(contract.shift(1)) & contract.notna() & contract.shift(1).notna()
    for i in range(1, len(prices)):
        c = contract.iloc[i]
        if c is None or pd.isna(c):
            continue
        prior = prices[c].iloc[i - 1]
        current = prices[c].iloc[i]
        if pd.notna(prior) and pd.notna(current) and prior != 0:
            returns.iloc[i] = current / prior - 1.0
    returns.loc[switched] -= roll_cost_bps / 10_000.0
    return returns, contract
