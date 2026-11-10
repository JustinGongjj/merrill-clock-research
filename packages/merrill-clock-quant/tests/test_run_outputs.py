"""Functional tests for the run outputs: manifest, diagnostics and metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantclock import cli

TICKERS = ["SPY", "IEF", "TLT", "GLD", "DBC", "USO", "CPER", "DBA", "BIL"]


def _fixture_data():
    macro_index = pd.date_range("1998-01-01", periods=320, freq="MS")
    t = np.arange(len(macro_index))
    macro = pd.DataFrame({
        "industrial_production": 100 * np.exp(0.002 * t) + 2 * np.sin(t / 6),
        "cpi": 100 * np.exp(0.0018 * t) + np.cos(t / 8),
    }, index=macro_index)
    dates = pd.bdate_range("2000-01-03", periods=5200)
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0003, 0.009, size=(len(dates), len(TICKERS)))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=TICKERS)
    return macro, prices


def _args(out: Path, **overrides) -> argparse.Namespace:
    values = dict(start="2000-01-01", output=str(out), cost_bps=10.0,
                  ip_lag_days=20, cpi_lag_days=20, z_window=60, smooth=3,
                  cash_policy="auto", keep_incomplete_month=False,
                  data_source="local", db=None, temperature=1.0, no_manifest=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_research_writes_manifest_diagnostics_and_metrics(tmp_path: Path, monkeypatch):
    macro, prices = _fixture_data()
    monkeypatch.setattr(cli, "load_market_inputs", lambda _args: (macro, prices))
    assert cli.run_research(_args(tmp_path)) == 0
    for name in ("metrics.json", "clock_backtest.csv", "phase_diagnostics.json",
                 "run_manifest.json", "macro_latest_vintage.csv", "monthly_asset_returns.csv"):
        assert (tmp_path / name).exists(), name

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["observations"] > 100
    assert metrics["temperature"] == 1.0
    assert isinstance(metrics["phase_quadrant_hit_rate"], float)
    assert isinstance(metrics["phase_brier_growth_up"], float)

    diagnostics = json.loads((tmp_path / "phase_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["calibration"]["n_periods"] > 0
    assert set(diagnostics["temperature_sensitivity"]["by_temperature"]) == {"0.5", "1.0", "2.0"}

    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["workstation"] == "merrill-clock-quant"
    assert manifest["dataset_ids"] == ["fred_macro", "market_prices"]
    assert manifest["tested_variants"] == 1
    assert "cost_bps" in manifest["parameter_confirmations"]
    # Parameters left at parser defaults are surfaced instead of hidden.
    assert any("without explicit confirmation" in i for i in manifest["validation_issues"])


def test_confirmation_gate_marks_default_parameters(tmp_path: Path):
    args = _args(tmp_path, temperature=1.0, cost_bps=10.0, z_window=60)
    args.z_window = 60  # same as default -> unconfirmed
    manifest = cli.write_run_manifest(args, tmp_path, sample_start="2000-01-31",
                                      sample_end="2020-12-31", universe=TICKERS,
                                      dataset_ids=["fred_macro"])
    assert manifest["parameter_confirmations"]["temperature"] == "default"
    assert any("FRED" in w for w in manifest["data_warnings"])
    assert any("未显式确认" in w for w in manifest["data_warnings"])


def test_explicit_temperature_is_recorded_as_confirmed(tmp_path: Path):
    args = _args(tmp_path, temperature=0.7)
    manifest = cli.write_run_manifest(args, tmp_path, sample_start="2000-01-31",
                                      sample_end="2020-12-31", universe=TICKERS,
                                      dataset_ids=["fred_macro"])
    assert manifest["parameter_confirmations"]["temperature"] == "explicit"
    assert manifest["temperature"] == 0.7


def test_no_manifest_flag_skips_the_file(tmp_path: Path, monkeypatch):
    macro, prices = _fixture_data()
    monkeypatch.setattr(cli, "load_market_inputs", lambda _args: (macro, prices))
    assert cli.run_research(_args(tmp_path, no_manifest=True)) == 0
    assert not (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "phase_diagnostics.json").exists()


def test_temperature_changes_phase_probabilities_but_not_sample(tmp_path: Path, monkeypatch):
    macro, prices = _fixture_data()
    monkeypatch.setattr(cli, "load_market_inputs", lambda _args: (macro, prices))
    low, high = tmp_path / "low", tmp_path / "high"
    assert cli.run_research(_args(low, temperature=0.5)) == 0
    assert cli.run_research(_args(high, temperature=2.0)) == 0
    low_metrics = json.loads((low / "metrics.json").read_text(encoding="utf-8"))
    high_metrics = json.loads((high / "metrics.json").read_text(encoding="utf-8"))
    assert low_metrics["observations"] == high_metrics["observations"]
    assert low_metrics["cagr"] != high_metrics["cagr"]
