import io

import pandas as pd
import pytest

from factorlab.data import parse_fama_french_monthly, read_factor_csv


def test_factor_csv_parses_dates_and_numeric_columns():
    csv = io.StringIO("date,Mkt-RF,RF\n2024-01-31,0.01,0.001\n2024-02-29,-0.02,0.001\n")
    data = read_factor_csv(csv)
    assert isinstance(data.index, pd.DatetimeIndex)
    assert list(data.columns) == ["Mkt-RF", "RF"]
    assert data.loc[pd.Timestamp("2024-01-31"), "Mkt-RF"] == .01


def test_fama_french_parser_offline_monthly_block():
    sample = """This file was created by CMPT monthly returns.
, Mkt-RF,SMB,HML,RMW,CMA,RF
196307,-0.39,-0.48,-0.84,0.64,-1.15,0.27
196308,5.08,-0.80,1.72,0.40,-0.38,0.25

Annual Factors: January-December
1963,12.40,1.20,2.30,3.40,4.50,2.70
"""
    data = parse_fama_french_monthly(sample)
    assert data.shape == (2, 6)
    assert list(data.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert data.index[0] == pd.Timestamp("1963-07-31")
    assert data.index[1] == pd.Timestamp("1963-08-31")
    assert data.iloc[0]["Mkt-RF"] == pytest.approx(-0.0039)
    assert data.iloc[0]["RF"] == pytest.approx(0.0027)


def test_fama_french_parser_reports_layout_changes():
    with pytest.raises(ValueError, match="header"):
        parse_fama_french_monthly("notes only\n196307,1,2,3,4,5,6\n")
