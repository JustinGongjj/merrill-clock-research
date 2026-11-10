"""Pipeline manifest tests plus integration checks against the real local database."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from clockdata.cli import _write_pipeline_manifest
from clockdata.sources import SOURCES
from clockdata.store import Store

REAL_DB = Path(os.environ.get("CLOCK_DATA_DB")
               or Path.home() / ".local" / "share" / "clockdata" / "finance.db")


def test_pipeline_manifest_records_datasets_and_artifacts(tmp_path: Path):
    db = tmp_path / "finance.db"
    frame = pd.DataFrame({"Mkt-RF": [0.01, -0.02]},
                         index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    store = Store(db).init_schema(sources=SOURCES).upsert_frame(
        "ff5_factors", "fama_french", "factors", frame, frequency="月")
    _write_pipeline_manifest(Store(db), tmp_path, cross_section_ok=True)
    payload = json.loads((tmp_path / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert payload["workstation"] == "clockdata-pipeline"
    assert "ff5_factors" in payload["dataset_ids"]
    assert "cross_section_report.json" in payload["artifacts"]
    assert payload["tested_variants"] == 1
    # Defaults are recorded as unconfirmed rather than silently accepted.
    assert any("maxlags" in i for i in payload["validation_issues"])


def test_pipeline_manifest_marks_skipped_cross_section(tmp_path: Path):
    db = tmp_path / "finance.db"
    frame = pd.DataFrame({"x": [1.0]}, index=pd.to_datetime(["2020-01-31"]))
    Store(db).init_schema(sources=SOURCES).upsert_frame(
        "custom_set", "local_csv", "custom", frame)
    _write_pipeline_manifest(Store(db), tmp_path, cross_section_ok=False)
    payload = json.loads((tmp_path / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert payload["cross_section_check_completed"] is False
    assert "synthesis_report.json" not in payload["artifacts"]


@pytest.mark.skipif(not REAL_DB.exists(), reason="local financial database not initialized")
def test_real_database_has_expected_datasets_and_coverage():
    store = Store(REAL_DB)
    frame = store.datasets()
    ids = set(frame["dataset_id"])
    assert {"fred_macro", "market_prices", "ff5_factors"}.issubset(ids)
    assert (frame["n_obs"] > 0).all()
    ff5 = store.load_frame("ff5_factors")
    assert {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"}.issubset(ff5.columns)
    assert ff5.index.min() < pd.Timestamp("1970-01-01")
    assert store.load_frame("market_prices").shape[1] >= 5


@pytest.mark.skipif(not REAL_DB.exists(), reason="local financial database not initialized")
def test_real_database_drives_cross_section_check(tmp_path: Path, monkeypatch):
    """End-to-end functional test: local DB -> factor panel -> RankIC report."""
    import sys
    from factorlab import cli as factor_cli

    out = tmp_path / "cross_section_report.json"
    monkeypatch.setattr(sys, "argv", ["merrill-clock-factor", "check",
                                      "--db", str(REAL_DB),
                                      "--factors", "mom_3m,vol_3m",
                                      "--output-json", str(out)])
    assert factor_cli.main() == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["data_source"] == "clockdata:market_prices"
    assert payload["n_months"] > 100
    assert len(payload["universe"]) >= 5
    for name, report in payload["factors"].items():
        assert report["ic_summary"]["n_periods"] > 50
        assert -1.0 <= report["ic_summary"]["ic_mean"] <= 1.0
        assert report["layer_monotonicity"]["shape"] != "insufficient_data"
