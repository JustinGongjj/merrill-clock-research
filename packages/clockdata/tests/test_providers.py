"""Tests for the optional upstream providers (AkShare / WindPy / FRED TB3MS).

Live network fetchers are guarded so the suite stays offline and deterministic;
run them explicitly with RUN_LIVE_TESTS=1 when you want to validate upstream.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from clockdata import providers
from clockdata.sources import DATASETS, FETCHERS, SOURCES

RUN_LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"
skip_live = pytest.mark.skipif(not RUN_LIVE, reason="set RUN_LIVE_TESTS=1 to hit upstream")


def test_windpy_unavailable_on_mac():
    # On macOS WindPy cannot be imported; the guard must return False cleanly.
    assert providers.windpy_available() is False


def test_tb3ms_to_monthly_return_conversion():
    idx = pd.date_range("2024-01-31", periods=4, freq="ME")
    tb3ms = pd.DataFrame({"TB3MS": [5.0, 5.0, 5.0, 5.0]}, index=idx)
    out = providers.tb3ms_to_monthly_return(tb3ms, idx)
    # annualized 5% -> monthly (5/100)/12 = 0.004166...
    assert np.isclose(out.iloc[0], 5.0 / 100.0 / 12.0)
    assert out.index.equals(idx)


def test_tb3ms_to_monthly_return_handles_short_history():
    # A short T-bill history must forward-fill then zero-fill, never raise.
    idx = pd.date_range("2024-01-31", periods=6, freq="ME")
    short = pd.DataFrame({"TB3MS": [4.8]}, index=[idx[2]])
    out = providers.tb3ms_to_monthly_return(short, idx)
    assert len(out) == 6
    assert out.iloc[2] == 4.8 / 100.0 / 12.0
    assert (out.iloc[3:] > 0).all()  # forward-filled
    assert (out.iloc[:2] == 0).all()  # zero-filled before history starts


def test_sources_register_ashare_and_tb3ms():
    assert "ashare_prices" in DATASETS
    assert DATASETS["ashare_prices"]["source_id"] == "akshare"
    assert "tb3ms" in DATASETS
    assert DATASETS["tb3ms"]["source_id"] == "fred"
    assert "ashare_prices" in FETCHERS
    assert "tb3ms" in FETCHERS
    source_ids = {s["source_id"] for s in SOURCES}
    assert {"akshare", "windpy", "fred", "yahoo", "fama_french"}.issubset(source_ids)


@skip_live
def test_fetch_tb3ms_live():
    frame = providers.fetch_tb3ms(start="2023-01-01")
    assert not frame.empty
    assert "TB3MS" in frame.columns
    assert (frame["TB3MS"].dropna() > 0).any()


@skip_live
def test_fetch_ashare_daily_small_live():
    panel, failed = providers.fetch_ashare_daily(
        universe=["600519", "000001"], start="2024-01-01",
        end="2024-03-01", delay_seconds=0.0,
    )
    assert panel.shape[1] >= 1
    assert isinstance(failed, list)


@skip_live
def test_csi300_universe_live():
    codes = providers.csi300_universe()
    assert len(codes) == 300
