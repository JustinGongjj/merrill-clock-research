"""Tests for the machine-readable research run contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from clockdata.manifest import (
    SCHEMA_VERSION, build_manifest, parameter_confirmations, read_manifest,
    unconfirmed_parameters, validate_manifest, write_manifest,
)


def _complete(**overrides):
    payload = dict(
        workstation="unit-test", run_id="run-1", dataset_ids=["ff5_factors"],
        sample_start="2000-01-31", sample_end="2020-12-31",
        as_of_rule="month-end + 20d lag", rebalance="monthly",
        execution="T -> T+1", execution_price="close", universe="SPY,IEF",
        cost_bps=10.0, benchmark="equal weight",
        parameter_confirmations={"cost_bps": "explicit"},
    )
    payload.update(overrides)
    return build_manifest(**payload)


def test_complete_manifest_has_no_validation_issues():
    assert validate_manifest(_complete()) == []


def test_required_fields_are_all_present():
    manifest = _complete()
    from clockdata.manifest import REQUIRED_FIELDS
    assert set(REQUIRED_FIELDS).issubset(manifest)
    assert manifest["schema_version"] == SCHEMA_VERSION


def test_empty_critical_fields_are_reported():
    issues = validate_manifest(_complete(sample_start=None, benchmark=""))
    assert any("sample_start" in i for i in issues)
    assert any("benchmark" in i for i in issues)


def test_missing_required_field_is_reported():
    manifest = _complete()
    del manifest["universe"]
    assert any("missing field: universe" in i for i in validate_manifest(manifest))


def test_unconfirmed_parameters_are_listed_as_issues():
    issues = validate_manifest(_complete(parameter_confirmations={"z_window": "default"}))
    assert any("z_window" in i and "without explicit confirmation" in i for i in issues)
    assert unconfirmed_parameters(_complete(parameter_confirmations={"z_window": "default"})) == ["z_window"]


def test_parameter_confirmations_distinguish_explicit_from_default():
    args = argparse.Namespace(cost_bps=10.0, z_window=36, smooth=3)
    result = parameter_confirmations(args, {"cost_bps": 10.0, "z_window": 60, "smooth": 3})
    assert result == {"cost_bps": "default", "z_window": "explicit", "smooth": "default"}


def test_parameter_confirmations_handles_missing_attribute():
    args = argparse.Namespace()
    assert parameter_confirmations(args, {"temperature": 1.0}) == {"temperature": "default"}


def test_tested_variants_must_be_positive():
    with pytest.raises(ValueError, match="tested_variants"):
        _complete(tested_variants=0)
    assert any("tested_variants" in i for i in validate_manifest(
        {**_complete(), "tested_variants": 0}))


def test_write_manifest_attaches_validation_issues(tmp_path: Path):
    path = write_manifest(_complete(parameter_confirmations={"smooth": "default"}),
                          tmp_path / "reports" / "run_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["validation_issues"] and "smooth" in " ".join(payload["validation_issues"])
    assert read_manifest(path)["run_id"] == "run-1"


def test_read_manifest_rejects_unknown_schema(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": "9.9"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported manifest schema"):
        read_manifest(path)


def test_dataset_ids_must_be_a_list():
    manifest = _complete()
    manifest["dataset_ids"] = "ff5_factors"
    assert any("dataset_ids must be a list" in i for i in validate_manifest(manifest))


def test_extra_fields_survive_roundtrip(tmp_path: Path):
    manifest = _complete(extra={"data_source": "local", "smooth_span": 3})
    payload = read_manifest(write_manifest(manifest, tmp_path / "m.json"))
    assert payload["data_source"] == "local" and payload["smooth_span"] == 3
