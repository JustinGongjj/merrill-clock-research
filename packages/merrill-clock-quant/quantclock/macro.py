"""Point-in-time-aware macro feature preparation.

The default FRED graph files contain latest revised observations. Release-lag
alignment prevents the simplest period-end lookahead, but cannot undo revisions.
Use a point-in-time vintage/release file for publication-grade historical tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _monthly_observations(series: pd.Series) -> pd.Series:
    s = series.dropna().copy()
    s.index = pd.to_datetime(s.index)
    # FRED monthly observations are commonly dated to the first day of the month.
    # Treat them as the value for that observation month, not as available then.
    s.index = s.index.to_period("M")
    return s.groupby(level=0).last().sort_index()


def delayed_monthly_asof(
    series: pd.Series,
    decision_dates: pd.DatetimeIndex,
    release_lag_days: int,
    name: str,
) -> pd.Series:
    """Align monthly observations to decision dates only after assumed release.

    An observation for calendar month M becomes usable at month-end(M) plus
    ``release_lag_days``. This is conservative but only an approximation of the
    actual release calendar; use release timestamps for strict point-in-time work.
    """
    s = _monthly_observations(series)
    obs_month_end = s.index.to_timestamp(how="end").normalize()
    known_at = (obs_month_end + pd.to_timedelta(int(release_lag_days), unit="D")).astype("datetime64[ns]")
    source = pd.DataFrame({"known_at": known_at, name: s.to_numpy()}).sort_values("known_at")
    decision_index = pd.DatetimeIndex(decision_dates).tz_localize(None).astype("datetime64[ns]")
    decisions = pd.DataFrame({"decision_date": decision_index}).sort_values("decision_date")
    aligned = pd.merge_asof(decisions, source, left_on="decision_date", right_on="known_at", direction="backward")
    return pd.Series(aligned[name].to_numpy(), index=pd.DatetimeIndex(decision_dates), name=name)


def _accelerations_on_observation_grid(
    industrial_production: pd.Series,
    cpi: pd.Series,
    growth_accel_lag_months: int,
    inflation_accel_lag_months: int,
) -> pd.DataFrame:
    """Compute growth/inflation acceleration on the monthly observation grid.

    Differencing must happen here, where one row is exactly one observation
    month, before any as-of replication onto decision dates. Computing the same
    ``diff(6)`` on decision dates would silently mean "six decision periods"
    (six days on a daily grid, or a duplicated/zero change whenever two adjacent
    decisions still see the same released observation).
    """
    ip_log = np.log(industrial_production.where(industrial_production > 0))
    ip_growth6 = ip_log.diff(6) * 100.0
    growth_accel = ip_growth6 - ip_growth6.shift(growth_accel_lag_months)
    cpi_yoy = cpi.div(cpi.shift(12)).sub(1.0).mul(100.0)
    inflation_accel = cpi_yoy - cpi_yoy.shift(inflation_accel_lag_months)
    return pd.DataFrame({"growth_accel": growth_accel, "inflation_accel": inflation_accel})


def make_clock_features(
    industrial_production: pd.Series,
    cpi: pd.Series,
    decision_dates: pd.DatetimeIndex,
    ip_release_lag_days: int = 20,
    cpi_release_lag_days: int = 20,
    growth_accel_lag_months: int = 3,
    inflation_accel_lag_months: int = 3,
) -> pd.DataFrame:
    """Return as-of-aligned growth/inflation acceleration features.

    Growth proxy: change in 6-month industrial-production growth over the
    configured lag. Inflation proxy: change in 12-month CPI inflation. Both are
    computed on the observation-month grid, then released to the caller's
    decision calendar with the assumed publication lag, so a daily, weekly or
    monthly decision grid all receive the same point-in-time value.

    These are transparent proxies, not official GDP-cycle labels, and the
    release lag is an approximation of the real publication calendar.
    """
    dates = pd.DatetimeIndex(decision_dates).tz_localize(None)
    ip = _monthly_observations(industrial_production)
    cpi_monthly = _monthly_observations(cpi)
    grid = _accelerations_on_observation_grid(
        ip, cpi_monthly, growth_accel_lag_months, inflation_accel_lag_months
    )
    columns = []
    for column, lag in (("growth_accel", ip_release_lag_days), ("inflation_accel", cpi_release_lag_days)):
        feature = grid[column].copy()
        # The month label is all that matters; delayed_monthly_asof re-derives it.
        feature.index = feature.index.to_timestamp(how="start")
        columns.append(delayed_monthly_asof(feature, dates, lag, column))
    return pd.concat(columns, axis=1).reindex(dates)


def load_vintage_csv(path: str) -> pd.DataFrame:
    """Load long point-in-time macro observations with date, release_date, value.

    Multiple releases/revisions per observation are supported. For each decision
    date, the latest release available by then is selected. This import format
    lets users use ALFRED or another official vintage download without embedding
    keys or silently treating revised data as real-time data.
    """
    frame = pd.read_csv(path, parse_dates=["date", "release_date"])
    required = {"date", "release_date", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Vintage CSV needs columns {sorted(required)}")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["date", "release_date", "value"]).sort_values(["date", "release_date"])


def align_vintage_asof(vintages: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Resolve long-format vintages to the latest value known at each decision date."""
    rows = []
    for decision in pd.DatetimeIndex(decision_dates):
        available = vintages[vintages["release_date"] <= decision]
        latest = available.sort_values("release_date").drop_duplicates("date", keep="last")
        row = latest.set_index("date")["value"].sort_index()
        rows.append(row.rename(decision))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(decision_dates))
