"""Machine-readable research run contract shared by all three workstations.

Why this exists: broker research desks publish the assumptions next to the
numbers (portfolio constraints, cost, rebalance, sample window, tested
variants). Keeping those only in README prose makes results impossible to
audit. A run manifest makes the assumptions travel with the outputs.

The schema is deliberately dependency-free so the Node-side workstation can
emit the same JSON without importing this package.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

REQUIRED_FIELDS = (
    "schema_version", "run_id", "workstation", "dataset_ids",
    "sample_start", "sample_end", "as_of_rule", "rebalance",
    "execution", "execution_price", "universe", "cost_bps",
    "constraints", "tested_variants", "selection_rule", "benchmark",
    "parameter_confirmations", "data_warnings", "produced_at",
)

# Fields that, when left blank, mean the reader cannot interpret the result.
CRITICAL_FIELDS = (
    "sample_start", "sample_end", "as_of_rule", "execution",
    "universe", "benchmark", "cost_bps",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_manifest(
    *,
    workstation: str,
    run_id: str,
    dataset_ids: list[str] | tuple[str, ...] = (),
    sample_start: str | None = None,
    sample_end: str | None = None,
    as_of_rule: str = "",
    rebalance: str = "",
    execution: str = "",
    execution_price: str = "",
    universe: str = "",
    cost_bps: float | None = None,
    slippage_bps: float | None = None,
    constraints: dict[str, Any] | None = None,
    tested_variants: int = 1,
    selection_rule: str = "",
    benchmark: str = "",
    parameter_confirmations: dict[str, str] | None = None,
    data_warnings: list[str] | tuple[str, ...] = (),
    git_commit: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one manifest. Nothing is inferred from the environment."""
    if tested_variants < 1:
        raise ValueError("tested_variants must be >= 1")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workstation": workstation,
        "dataset_ids": list(dataset_ids),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "as_of_rule": as_of_rule,
        "rebalance": rebalance,
        "execution": execution,
        "execution_price": execution_price,
        "universe": universe,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "constraints": dict(constraints or {}),
        "tested_variants": int(tested_variants),
        "selection_rule": selection_rule,
        "benchmark": benchmark,
        "parameter_confirmations": dict(parameter_confirmations or {}),
        "data_warnings": list(data_warnings),
        "git_commit": git_commit,
        "produced_at": utcnow(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def parameter_confirmations(args: Any, defaults: dict[str, Any]) -> dict[str, str]:
    """Mark which research parameters were explicitly set by the caller.

    ``defaults`` maps argument names to the parser default. A parameter whose
    value equals its default is recorded as ``default`` (unconfirmed), not as an
    explicit research decision. This is the "confirmation gate": silent
    defaults are allowed to run, but they can no longer hide in the output.
    """
    result: dict[str, str] = {}
    for name, default in defaults.items():
        value = getattr(args, name, default)
        result[name] = "default" if value == default else "explicit"
    return result


def unconfirmed_parameters(manifest: dict[str, Any]) -> list[str]:
    return sorted(k for k, v in manifest.get("parameter_confirmations", {}).items()
                  if v == "default")


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return human-readable issues; an empty list means the manifest is complete."""
    issues: list[str] = []
    for key in REQUIRED_FIELDS:
        if key not in manifest:
            issues.append(f"missing field: {key}")
    for key in CRITICAL_FIELDS:
        value = manifest.get(key)
        if value is None or value == "":
            issues.append(f"empty critical field: {key}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        issues.append(f"unsupported schema_version: {manifest.get('schema_version')}")
    if not isinstance(manifest.get("dataset_ids"), list):
        issues.append("dataset_ids must be a list")
    if not isinstance(manifest.get("tested_variants"), int) or manifest.get("tested_variants", 0) < 1:
        issues.append("tested_variants must be a positive integer")
    for name in unconfirmed_parameters(manifest):
        issues.append(f"parameter used without explicit confirmation: {name}")
    return issues


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write the manifest with its own validation report attached."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload["validation_issues"] = validate_manifest(manifest)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def read_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema: {payload.get('schema_version')}")
    return payload
