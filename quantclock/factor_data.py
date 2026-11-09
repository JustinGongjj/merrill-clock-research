"""Public Fama-French factor data adapter (monthly, percentage points -> decimals)."""
from __future__ import annotations
import io
import zipfile
import pandas as pd
import requests

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"


def fetch_fama_french_5_factors(timeout: int = 30) -> pd.DataFrame:
    """Fetch the public U.S. monthly five-factor file, returning decimal returns.

    Check the data library's terms and definitions before redistributing or using
    outside research. Values are reported in percent in the source file.
    """
    response = requests.get(FF5_URL, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next((n for n in archive.namelist() if n.lower().endswith(".csv")), None)
        if name is None:
            raise ValueError("No CSV inside Fama-French ZIP")
        lines = archive.read(name).decode("utf-8-sig", errors="replace").splitlines()
    # Find the monthly header/data start; the source includes notes and an annual section.
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
    data = pd.DataFrame([v for _, v in rows], index=dates, columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    # all source columns are percentages; including RF.
    data = data / 100.0
    return data
