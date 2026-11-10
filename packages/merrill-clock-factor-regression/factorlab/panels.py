"""Build a cross-sectional panel (date x asset) from a local price dataset.

The local financial database stores daily adjusted closes for a small ETF
universe. That is enough to exercise the full single-factor test pipeline, but
it is NOT a stock universe: results are ETF momentum/volatility diagnostics and
must never be compared with broker A-share factor IC numbers.

All factors at date t use information available at t only; the dependent
variable is the return from t to t+1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def to_monthly(prices: pd.DataFrame, drop_incomplete: bool = True,
               as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Resample daily adjusted closes to month-end, dropping a partial month."""
    frame = prices.sort_index().copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    monthly = frame.resample("ME").last().dropna(how="all")
    if drop_incomplete and len(monthly):
        cutoff = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
        # Drop every month-end label beyond the as-of date, not just the last
        # row: an as-of date earlier than the data end must not leak later rows.
        monthly = monthly[monthly.index <= cutoff]
    return monthly


def price_factors(
    monthly_prices: pd.DataFrame,
    momentum_windows: tuple[int, ...] = (1, 3, 6, 12),
    volatility_windows: tuple[int, ...] = (3, 12),
) -> dict[str, pd.DataFrame]:
    """Momentum (past k-month return) and realized volatility factors.

    Volatility uses the trailing standard deviation of monthly returns. A
    ``rev_1m`` mirror is included so the sign convention of short-term reversal
    is explicit rather than implicit in a negative IC.
    """
    if monthly_prices.empty:
        raise ValueError("monthly_prices is empty")
    returns = monthly_prices.pct_change(fill_method=None)
    factors: dict[str, pd.DataFrame] = {}
    for w in momentum_windows:
        if w < 1:
            raise ValueError("momentum windows must be >= 1")
        factors[f"mom_{w}m"] = monthly_prices.pct_change(w, fill_method=None)
    for w in volatility_windows:
        if w < 2:
            raise ValueError("volatility windows must be >= 2")
        factors[f"vol_{w}m"] = returns.rolling(w, min_periods=max(2, w // 2)).std(ddof=0)
    if "mom_1m" in factors:
        factors["rev_1m"] = -factors["mom_1m"]
    return factors


def forward_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Return from month t to month t+1, indexed by the decision month t."""
    return monthly_prices.pct_change(fill_method=None).shift(-1)


def load_prices_from_db(db_path: str | Path, dataset_id: str = "market_prices") -> pd.DataFrame:
    """Read a price dataset from the local clockdata SQLite store (lazy import)."""
    try:
        from clockdata.store import Store
    except ImportError as exc:
        raise RuntimeError(
            "Database access needs the clockdata package installed alongside factorlab"
        ) from exc
    frame = Store(db_path).load_frame(dataset_id)
    if frame.empty:
        raise RuntimeError(f"Dataset {dataset_id!r} is empty or missing in {db_path}")
    return frame


def build_panel(
    prices: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    momentum_windows: tuple[int, ...] = (1, 3, 6, 12),
    volatility_windows: tuple[int, ...] = (3, 12),
    min_history_months: int = 14,
) -> dict[str, Any]:
    """Return factor frames, forward returns and sample metadata for one universe."""
    monthly = to_monthly(prices, as_of=as_of)
    if len(monthly) < min_history_months:
        raise ValueError(
            f"Need at least {min_history_months} monthly observations for the configured factors; "
            f"got {len(monthly)}"
        )
    factors = price_factors(monthly, momentum_windows, volatility_windows)
    fwd = forward_returns(monthly)
    return {
        "monthly_prices": monthly,
        "factors": factors,
        "forward_returns": fwd,
        "universe": list(monthly.columns),
        "sample_start": str(monthly.index.min().date()),
        "sample_end": str(monthly.index.max().date()),
        "n_months": int(len(monthly)),
    }
