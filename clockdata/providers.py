"""Optional upstream providers beyond the three always-available Python sources.

This module is intentionally dependency-light: ``akshare`` and ``WindPy`` are
imported lazily so the core database keeps working when neither is installed.

  * AkShare  -- free, cross-platform (macOS/Linux/Windows). Used for real
    A-share daily prices and CSI 300 constituents. No token required.
  * WindPy   -- Windows-only, needs a licensed Wind terminal. On macOS it cannot
    be imported, so every entry point raises an informative error instead of
    failing silently.
  * FRED     -- TB3MS (3-month T-bill rate) via the unauthenticated CSV endpoint
    (no key needed); ALFRED vintage (point-in-time) via the keyed API when
    ``FRED_API_KEY`` is set.
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd


# --------------------------------------------------------------------------- #
# AkShare (A-share real prices)
# --------------------------------------------------------------------------- #
def csi300_universe() -> list[str]:
    """Return the current CSI 300 constituent codes (6-digit strings)."""
    import akshare as ak

    cons = ak.index_stock_cons_csindex(symbol="000300")
    codes = [str(c).zfill(6) for c in cons["成分券代码"].tolist()]
    if not codes:
        raise RuntimeError("CSI 300 constituent list came back empty")
    return codes


def fetch_ashare_daily(
    universe: Iterable[str] | None = None,
    start: str = "2010-01-01",
    end: str | None = None,
    *,
    adjust: str = "qfq",
    delay_seconds: float = 0.05,
    max_stocks: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch forward-adjusted daily closes for an A-share universe.

    Returns a wide frame (date x code) of adjusted closes plus the list of
    codes that failed (network/empty). Failures are collected, never fatal, so a
    partial sample is still usable for cross-sectional work.

    Note: this hits Eastmoney through AkShare and is rate-limited; the per-stock
    delay is a courtesy, not a guarantee. Keep ``max_stocks`` small for probes.
    """
    import akshare as ak

    if universe is None:
        universe = csi300_universe()
    codes = [str(c).zfill(6) for c in universe]
    if max_stocks is not None:
        codes = codes[:max_stocks]
    end = end or pd.Timestamp.today().strftime("%Y%m%d")
    start_compact = start.replace("-", "")
    end_compact = end.replace("-", "")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for code in codes:
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_compact, end_date=end_compact, adjust=adjust,
            )
            if df is None or df.empty:
                failed.append(code)
                continue
            series = df.set_index("日期")["收盘"].rename(code)
            series.index = pd.to_datetime(series.index)
            frames.append(series.to_frame().sort_index())
        except Exception:  # noqa: BLE001 - one bad stock must not kill the pull
            failed.append(code)
        if delay_seconds:
            time.sleep(delay_seconds)

    if not frames:
        raise RuntimeError(
            f"No A-share prices fetched for the requested universe "
            f"({len(codes)} codes attempted). Check network and AkShare availability."
        )
    panel = pd.concat(frames, axis=1).sort_index()
    return panel, failed


# --------------------------------------------------------------------------- #
# WindPy (Windows-only; graceful on macOS)
# --------------------------------------------------------------------------- #
def windpy_available() -> bool:
    try:
        import WindPy  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - expected on non-Windows / unlicensed hosts
        return False


def fetch_windpy_daily(
    universe: Iterable[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted closes via WindPy.

    WindPy only runs on Windows with a licensed Wind terminal. On macOS this
    raises immediately so callers can fall back to AkShare or the local DB
    rather than hanging on an unavailable COM/API surface.
    """
    if not windpy_available():
        raise RuntimeError(
            "WindPy is not importable on this host. It requires a Windows machine "
            "with a licensed Wind (万得) terminal. Use the AkShare provider or the "
            "local database on macOS instead."
        )
    from WindPy import w  # imported only after the availability guard above

    codes = list(universe)
    end = end or pd.Timestamp.today().strftime("%Y%m%d")
    start_compact = start.replace("-", "")
    raw = w.wsd(",".join(codes), "close", start_compact, end, "PriceAdj:U;Fill:Previous")
    if raw.ErrorCode != 0:
        raise RuntimeError(f"WindPy wsd returned ErrorCode {raw.ErrorCode}")
    panel = pd.DataFrame(raw.Data, index=raw.Codes, columns=pd.to_datetime(raw.Times)).T
    panel.index.name = "date"
    return panel.sort_index()


# --------------------------------------------------------------------------- #
# FRED TB3MS (3-month T-bill rate) and optional ALFRED vintage
# --------------------------------------------------------------------------- #
def fetch_tb3ms(start: str = "1990-01-01") -> pd.DataFrame:
    """Fetch the 3-month T-bill rate (TB3MS) via FRED's unauthenticated CSV.

    Returns a single-column frame of the annualized rate in percent, indexed by
    date. No API key required. Treat as the latest-vintage value, not
    point-in-time.
    """
    import requests

    start_compact = start.replace("-", "")
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = requests.get(url, params={"id": "TB3MS", "cosd": start_compact}, timeout=30)
    r.raise_for_status()
    frame = pd.read_csv(pd.io.common.StringIO(r.text))
    frame.columns = [c.strip() for c in frame.columns]
    id_col = "Observation Date" if "Observation Date" in frame.columns else frame.columns[0]
    val_col = "TB3MS" if "TB3MS" in frame.columns else frame.columns[1]
    frame[id_col] = pd.to_datetime(frame[id_col])
    out = frame[[id_col, val_col]].set_index(id_col)
    out.index.name = "date"
    out[val_col] = pd.to_numeric(out[val_col], errors="coerce")
    out = out.dropna().sort_index()
    out.columns = ["TB3MS"]
    return out


def tb3ms_to_monthly_return(tb3ms: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Convert a TB3MS rate (annualized percent) to monthly returns aligned to ``index``.

    ``monthly_return = (rate/100) / 12`` using the month-end rate. Missing months
    are forward-filled then zero-filled so a short T-bill history does not abort
    the backtest.
    """
    if tb3ms is None or tb3ms.empty:
        return pd.Series(dtype=float, index=index)
    monthly_rate = tb3ms.resample("ME").last().iloc[:, 0] / 100.0 / 12.0
    aligned = monthly_rate.reindex(index)
    aligned = aligned.interpolate().ffill().fillna(0.0)
    aligned.name = "cash_return"
    return aligned


def fetch_fred_vintage(series: str, start: str, api_key: str) -> pd.DataFrame:
    """Fetch a point-in-time (vintage) observation history from ALFRED.

    Requires a FRED API key (set ``FRED_API_KEY``). Returns a long-format frame
    with columns ``date`` (the period the value refers to), ``vintage`` (when it
    was published) and ``value``. This is the correct tool for macro
    point-in-time research but is optional -- the default pipeline uses the
    latest-vintage CSV instead.
    """
    import requests

    url = "https://api.stlouisfed.org/fred/vintage"
    params = {"series_id": series, "api_key": api_key, "file_type": "json",
              "realtime_start": start}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    records: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    for obs in payload.get("observations", []):
        date = obs.get("date")
        vintage = obs.get("vintage_date") or obs.get("realtime_end")
        val = obs.get("value")
        if date is None or vintage is None or val in (None, "."):
            continue
        try:
            records.append((pd.to_datetime(date), pd.to_datetime(vintage), float(val)))
        except (ValueError, TypeError):
            continue
    frame = pd.DataFrame(records, columns=["date", "vintage", "value"])
    return frame
