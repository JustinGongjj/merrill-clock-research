"""Functional tests: price panel construction and the two-track CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factorlab import cli
from factorlab.panels import build_panel, forward_returns, price_factors, to_monthly

TICKERS = ["SPY", "IEF", "TLT", "GLD", "DBC", "USO", "CPER", "DBA", "BIL"]


def _daily_prices(periods: int = 700, seed: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=periods)
    returns = rng.normal(0.0003, 0.01, size=(periods, len(TICKERS)))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


def _run_cli(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["merrill-clock-factor"] + argv)
    return cli.main()


def test_to_monthly_drops_partial_month():
    prices = _daily_prices(periods=120)
    monthly_strict = to_monthly(prices, as_of=pd.Timestamp("2015-03-15"))
    assert monthly_strict.index[-1] <= pd.Timestamp("2015-03-15")
    monthly_keep = to_monthly(prices, drop_incomplete=False)
    assert len(monthly_keep) >= len(monthly_strict)


def test_price_factors_have_expected_names_and_no_future_data():
    monthly = to_monthly(_daily_prices(), as_of=pd.Timestamp("2017-10-31"))
    factors = price_factors(monthly)
    assert {"mom_1m", "mom_3m", "mom_6m", "mom_12m", "vol_3m", "vol_12m", "rev_1m"} <= set(factors)
    assert (factors["rev_1m"].dropna() == -factors["mom_1m"].dropna()).all().all()
    assert factors["mom_1m"].index.equals(monthly.index)


def test_forward_returns_are_shifted_one_period():
    monthly = to_monthly(_daily_prices(), as_of=pd.Timestamp("2017-10-31"))
    fwd = forward_returns(monthly)
    assert fwd.iloc[0]["SPY"] == pytest.approx(monthly["SPY"].pct_change().iloc[1])
    assert fwd.iloc[-1].isna().all()


def test_build_panel_reports_universe_and_sample():
    panel = build_panel(_daily_prices(), as_of=pd.Timestamp("2017-10-31"))
    assert panel["universe"] == TICKERS
    assert panel["n_months"] > 14
    assert panel["sample_start"].startswith("2015-01")
    assert all(frame.shape == panel["monthly_prices"].shape for frame in panel["factors"].values())
    assert panel["forward_returns"].shape == panel["monthly_prices"].shape


def test_build_panel_rejects_short_history():
    with pytest.raises(ValueError, match="monthly observations"):
        build_panel(_daily_prices(periods=40), as_of=pd.Timestamp("2015-02-28"))


def test_cli_check_writes_report_and_ic_history(tmp_path: Path, monkeypatch):
    prices_path = tmp_path / "prices.csv"
    _daily_prices().to_csv(prices_path, index_label="date")
    report_path = tmp_path / "cross_section_report.json"
    ic_path = tmp_path / "ic.csv"
    rc = _run_cli(monkeypatch, ["check", "--prices-csv", str(prices_path),
                                "--factors", "mom_1m,mom_3m,vol_3m",
                                "--output-json", str(report_path),
                                "--ic-csv", str(ic_path)])
    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["track"] == "cross_sectional_single_factor_test"
    assert set(payload["factors"]) == {"mom_1m", "mom_3m", "vol_3m"}
    assert payload["universe"] == TICKERS
    assert "ETF" in payload["universe_type"]
    assert ic_path.exists()
    ic_frame = pd.read_csv(ic_path)
    assert list(ic_frame.columns) == ["date", "mom_1m", "mom_3m", "vol_3m"]


def test_cli_check_rejects_unknown_factor(tmp_path: Path, monkeypatch):
    prices_path = tmp_path / "prices.csv"
    _daily_prices().to_csv(prices_path, index_label="date")
    with pytest.raises(SystemExit, match="Unknown factors"):
        _run_cli(monkeypatch, ["check", "--prices-csv", str(prices_path),
                               "--factors", "not_a_factor",
                               "--output-json", str(tmp_path / "x.json")])


def test_cli_synthesize_writes_weights(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(2)
    dates = pd.date_range("2018-01-31", periods=30, freq="ME")
    ic = pd.DataFrame(rng.normal(0.02, 0.05, size=(30, 3)),
                      index=dates, columns=["f1", "f2", "f3"])
    ic_path = tmp_path / "ic.csv"
    ic.to_csv(ic_path, index_label="date")
    out = tmp_path / "synthesis.json"
    assert _run_cli(monkeypatch, ["synthesize", "--ic-csv", str(ic_path),
                                  "--method", "ic_half_life", "--window", "12",
                                  "--output-json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["method"] == "ic_half_life"
    assert sum(payload["weights"].values()) == pytest.approx(1.0)
    assert payload["rolling_weight_stability"]["n_changes"] > 0
    assert payload["caveats"]


def test_cli_attribute_writes_hac_results(tmp_path: Path, monkeypatch):
    rng = np.random.default_rng(12)
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    factors = pd.DataFrame(rng.normal(0.005, 0.03, size=(60, 3)),
                           index=dates, columns=["Mkt-RF", "SMB", "HML"])
    factors["RF"] = 0.001
    factors_path = tmp_path / "factors.csv"
    factors.to_csv(factors_path, index_label="date")
    strategy = 0.002 + 0.8 * factors["Mkt-RF"] + rng.normal(0, 0.01, 60)
    returns_path = tmp_path / "returns.csv"
    pd.DataFrame({"date": dates, "net_return": strategy}).to_csv(returns_path, index=False)
    out = tmp_path / "attribution.json"
    assert _run_cli(monkeypatch, ["attribute", "--returns-csv", str(returns_path),
                                  "--factors-csv", str(factors_path),
                                  "--output-json", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["nobs"] == 60
    assert payload["coefficients"]["Mkt-RF"]["coef"] == pytest.approx(0.8, abs=0.05)
    assert "alignment_note" in payload
