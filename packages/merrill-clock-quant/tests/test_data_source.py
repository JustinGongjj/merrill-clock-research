"""Regression tests for live/local data-source switching and upstream gaps."""
from __future__ import annotations

import argparse
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest

from quantclock import cli, data


def _fake_yfinance(bulk: pd.DataFrame):
    return types.SimpleNamespace(download=lambda tickers, **kwargs: bulk)


def _bulk(glc_values, spy_values=(1.0, 2.0, 3.0)):
    idx = pd.date_range("2024-01-02", periods=3, freq="D")
    return pd.DataFrame({("Close", "SPY"): list(spy_values),
                         ("Close", "GLD"): list(glc_values)}, index=idx)


def test_empty_ticker_column_is_surfaced_as_warning(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(_bulk([np.nan] * 3)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prices = data.fetch_market_prices(tickers=["SPY", "GLD"],
                                          start="2024-01-01", end="2024-01-10")
    assert prices["GLD"].isna().all()
    assert any("GLD" in str(w.message) and "local" in str(w.message) for w in caught)


def test_all_empty_columns_raise_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance",
                        _fake_yfinance(_bulk([np.nan] * 3, spy_values=[np.nan] * 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="no usable price columns"):
            data.fetch_market_prices(tickers=["SPY", "GLD"],
                                     start="2024-01-01", end="2024-01-10")


def test_auto_mode_falls_back_and_records_provenance(monkeypatch):
    args = argparse.Namespace(data_source="auto")
    seen: list[str] = []

    def fake_run(current):
        seen.append(current.data_source)
        if len(seen) == 1:
            raise RuntimeError("no complete month has all four core sleeves")
        return 0

    monkeypatch.setattr(cli, "yahoo_healthy", lambda: True)
    monkeypatch.setattr(cli, "run_research", fake_run)
    assert cli.run_with_fallback(args) == 0
    assert seen == ["auto", "local-fallback"]


def test_live_mode_does_not_silently_fall_back(monkeypatch):
    args = argparse.Namespace(data_source="live")
    monkeypatch.setattr(cli, "run_research",
                        lambda _a: (_ for _ in ()).throw(RuntimeError("upstream down")))
    with pytest.raises(RuntimeError, match="upstream down"):
        cli.run_with_fallback(args)


def test_auto_mode_skips_live_when_upstream_probe_fails(monkeypatch):
    args = argparse.Namespace(data_source="auto")
    seen: list[str] = []

    def fake_run(current):
        seen.append(current.data_source)
        return 0

    monkeypatch.setattr(cli, "yahoo_healthy", lambda: False)
    monkeypatch.setattr(cli, "run_research", fake_run)
    assert cli.run_with_fallback(args) == 0
    assert seen == ["local-fallback"]


def test_local_mode_reads_from_store(monkeypatch):
    args = argparse.Namespace(data_source="local", start="2000-01-01", db=None)
    sentinel = (pd.DataFrame({"industrial_production": [1.0], "cpi": [2.0]}),
                pd.DataFrame({"SPY": [3.0]}))
    monkeypatch.setattr(cli, "_fetch_live", lambda _a: pytest.fail("should not hit network"))
    monkeypatch.setattr(cli, "_load_local", lambda _a: sentinel)
    assert cli.load_market_inputs(args) is sentinel
