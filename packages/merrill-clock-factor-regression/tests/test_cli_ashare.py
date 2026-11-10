"""Tests that the cross-section CLI reports the correct universe type and that
an A-share panel (real adjusted prices) flows through the same factor pipeline.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from factorlab import cli as factor_cli


def _synthetic_ashare_prices(path, n_codes: int = 20,
                             n_months: int = 36) -> str:
    months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(7)
    cols = {f"{i:06d}": 100 * np.exp(np.cumsum(rng.normal(0.005, 0.04, n_months)))
            for i in range(n_codes)}
    frame = pd.DataFrame(cols, index=months)
    frame.index.name = "date"
    target = str(path)
    frame.to_csv(target, index_label="date")
    return target


def test_check_reports_ashare_universe_type(tmp_path):
    csv = _synthetic_ashare_prices(tmp_path / "ashare.csv")
    out_json = tmp_path / "report.json"
    args = argparse.Namespace(
        prices_csv=csv, db=None, dataset="ashare_prices", date_column="date",
        factors="mom_1m,mom_3m,rev_1m", n_groups=5, min_assets=5,
        no_preprocess=False, output_json=out_json, ic_csv=None,
    )
    rc = factor_cli.cmd_check(args)
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "A-share" in payload["universe_type"]
    assert "A股" in payload["caveats"][0]
    assert set(payload["factors"]) == {"mom_1m", "mom_3m", "rev_1m"}


def test_check_reports_etf_universe_type_for_market_prices(tmp_path):
    # The default dataset keeps the ETF caveat; ensure the dynamic branch works.
    csv = _synthetic_ashare_prices(tmp_path / "etf.csv")
    out_json = tmp_path / "report_etf.json"
    args = argparse.Namespace(
        prices_csv=csv, db=None, dataset="market_prices", date_column="date",
        factors="mom_1m", n_groups=5, min_assets=5, no_preprocess=False,
        output_json=out_json, ic_csv=None,
    )
    rc = factor_cli.cmd_check(args)
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "ETF" in payload["universe_type"]


def test_check_writes_ic_csv_for_synthesize(tmp_path):
    csv = _synthetic_ashare_prices(tmp_path / "ashare2.csv")
    out_json = tmp_path / "report2.json"
    ic_csv = tmp_path / "ic.csv"
    args = argparse.Namespace(
        prices_csv=csv, db=None, dataset="ashare_prices", date_column="date",
        factors="mom_1m,vol_3m", n_groups=5, min_assets=5, no_preprocess=False,
        output_json=out_json, ic_csv=ic_csv,
    )
    factor_cli.cmd_check(args)
    ic = pd.read_csv(ic_csv, parse_dates=["date"]).set_index("date")
    assert {"mom_1m", "vol_3m"}.issubset(ic.columns)
    # rank_ic returns one row per panel month (months that fail min_assets get NaN),
    # so the IC history spans all 36 months rather than 35.
    assert len(ic) == 36
