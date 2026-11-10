"""Public Fama-French factor data adapter (monthly, percentage points -> decimals)."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import IO

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FF5_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]


def _session_with_retries(total: int = 3) -> requests.Session:
    """Public data endpoints occasionally drop a TLS connection; retry idempotent GETs."""
    session = requests.Session()
    retry = Retry(total=total, connect=total, read=total, backoff_factor=0.6,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def parse_fama_french_monthly(text: str) -> pd.DataFrame:
    """Parse the monthly block of the Fama-French five-factor CSV into decimals.

    Kept separate from the download so the parser can be tested offline and so
    an upstream layout change is easy to localise. The annual block that follows
    the monthly rows is ignored.
    """
    lines = text.splitlines()
    header_line = next((i for i, line in enumerate(lines) if "Mkt-RF" in line and "SMB" in line and "HML" in line), None)
    if header_line is None:
        raise ValueError("Could not locate five-factor header; the source format may have changed")
    rows = []
    for line in lines[header_line + 1:]:
        fields = [part.strip() for part in line.split(",")]
        if not fields or len(fields) < 7 or not (len(fields[0]) == 6 and fields[0].isdigit()):
            if rows:
                break
            continue
        try:
            values = [float(v) for v in fields[1:7]]
        except ValueError:
            continue
        rows.append((fields[0], values))
    if not rows:
        raise ValueError("No monthly factor records parsed")
    dates = pd.PeriodIndex([d for d, _ in rows], freq="M").to_timestamp(how="end").normalize()
    data = pd.DataFrame([v for _, v in rows], index=dates, columns=FF5_COLUMNS)
    return data / 100.0


def fetch_fama_french_5_factors(timeout: int = 30) -> pd.DataFrame:
    """Fetch the public U.S. monthly five-factor file, returning decimal returns.

    Check the data library's terms and definitions before redistributing or using
    outside research. Values are reported in percent in the source file.
    """
    response = _session_with_retries().get(FF5_URL, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next((n for n in archive.namelist() if n.lower().endswith(".csv")), None)
        if name is None:
            raise ValueError("No CSV inside Fama-French ZIP")
        text = archive.read(name).decode("utf-8-sig", errors="replace")
    return parse_fama_french_monthly(text)


def read_factor_csv(path: str | Path | IO, date_column: str = "date") -> pd.DataFrame:
    """Read a caller-provided wide factor-return CSV (returns must be decimals).

    ``path`` may be a local file path or any text file-like object. The file
    must have one date column and one numeric column per factor. An optional
    ``RF`` column is recognized as the risk-free series by the demo and by
    ``factor_regression``. Month-start and month-end date labels are both
    accepted; the regression aligns them by calendar month. This helper does
    not download, cache, or redistribute input data.
    """
    frame = pd.read_csv(path)
    if date_column not in frame.columns:
        raise ValueError(f"Missing date column {date_column!r}")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    if frame.empty:
        raise ValueError("Factor CSV has no valid dated rows")
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
