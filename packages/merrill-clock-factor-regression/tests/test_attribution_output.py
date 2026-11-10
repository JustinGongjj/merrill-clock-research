"""Tests for the factor-attribution JSON output added to the demo script."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import factorlab


def _example_script() -> Path:
    return Path(factorlab.__file__).resolve().parent.parent / "examples_factor_regression.py"


def test_demo_writes_machine_readable_attribution(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "attribution.json"
    proc = subprocess.run(
        [sys.executable, str(_example_script()), "--output-json", str(target)],
        capture_output=True, text=True, check=True,
    )
    assert "Wrote factor attribution JSON" in proc.stdout
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["factors_source"] == "synthetic_demo"
    assert payload["nobs"] == 120
    assert payload["risk_free"] == "supplied"
    assert set(payload["factor_names"]) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA"}
    assert "const" in payload["coefficients"]
    for item in payload["coefficients"].values():
        assert set(item) == {"coef", "t", "p"}
    assert 0.0 <= payload["rsquared"] <= 1.0
    assert "forecasts" in payload["interpretation"]


def test_no_output_json_flag_keeps_original_behaviour(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(_example_script())],
        capture_output=True, text=True, check=True,
    )
    assert "R-squared:" in proc.stdout
    assert "Wrote factor attribution JSON" not in proc.stdout
    assert list(tmp_path.iterdir()) == []
