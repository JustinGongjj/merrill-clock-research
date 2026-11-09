"""Public-data ingestion adapters for the Merrill-clock research scaffold.

FRED's public graph CSV endpoint does not require an API key. It returns the
latest revised series, not a point-in-time vintage. For investable research,
use ALFRED vintages or provide point-in-time release data; see README.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import io
import warnings

import pandas as pd
import requests

FRED_GRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
DEFAULT_MACRO_SERIES = {"industrial_production": "INDPRO", "cpi": "CPIAUCSL"}
DEFAULT_MARKET_TICKERS = {
    "equity": "SPY", "treasury_intermediate": "IEF", "treasury_long": "TLT",
    "gold": "GLD", "broad_commodities": "DBC", "energy_proxy": "USO",
    "industrial_metals_proxy": "CPER", "agriculture_proxy": "DBA", "cash_proxy": "BIL",
}


def fetch_fred_series(series_id: str, start: str = "1990-01-01", timeout: int = 30) -> pd.Series:
    """Fetch a FRED series as a numeric Series; latest revisions are returned."""
    url = FRED_GRAPH.format(series_id=series_id)
    response = requests.get(url, params={"cosd": start}, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.BytesIO(response.content))
    if frame.shape[1] < 2:
        raise ValueError(f"Unexpected FRED CSV for {series_id}: {frame.columns.tolist()}")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["date", series_id]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    series = frame.dropna(subset=["date"]).set_index("date")[series_id].dropna().sort_index()
    series.name = series_id
    return series


def fetch_fred_panel(series: dict[str, str] | None = None, start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch named FRED series, aligning by observation date."""
    mapping = series or DEFAULT_MACRO_SERIES
    return pd.concat({name: fetch_fred_series(sid, start=start) for name, sid in mapping.items()}, axis=1)


def fetch_market_prices(
    tickers: Iterable[str] | None = None,
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily close proxies from Yahoo Finance via yfinance.

    These are convenient research proxies, not exchange-certified data. Yahoo
    continuous futures tickers are deliberately not included by default because
    their roll construction is not a substitute for contract-level backtests.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install the optional market-data dependency: pip install '.[market]'") from exc
    symbols = list(tickers or DEFAULT_MARKET_TICKERS.values())
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    if raw.empty:
        raise RuntimeError("Market data provider returned no rows. Check network, ticker symbols, or dates.")
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        field = "Close" if "Close" in level0 else level0[0]
        prices = raw[field].copy()
    else:
        field = "Close" if "Close" in raw.columns else raw.columns[-1]
        prices = raw[[field]].rename(columns={field: symbols[0]})
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.apply(pd.to_numeric, errors="coerce").sort_index()
    prices = prices.dropna(how="all")
    warnings.warn("Yahoo Finance data are third-party adjusted-price proxies; verify terms and corporate actions before research use.", RuntimeWarning)
    return prices


def read_price_csv(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """Load a local wide CSV: one date column and one price column per asset."""
    frame = pd.read_csv(path, parse_dates=[date_column])
    if date_column not in frame:
        raise ValueError(f"Missing required date column {date_column!r}")
    frame = frame.set_index(date_column).sort_index()
    frame.index = pd.to_datetime(frame.index)
    return frame.apply(pd.to_numeric, errors="coerce")


def write_csv(frame: pd.DataFrame, path: str | Path, index_label: str = "date") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index_label=index_label)
    return target
