"""Standalone HAC factor-regression example.

No arguments runs an offline deterministic synthetic demo. To analyze your own
strategy return CSV, pass --returns-csv; supply --factors-csv to avoid the
optional live Fama-French download.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.data import fetch_fama_french_5_factors, read_factor_csv
from factorlab.regression import factor_regression


def default_db_path() -> Path:
    return Path(os.environ.get("CLOCK_DATA_DB")
                or Path.home() / ".local" / "share" / "clockdata" / "finance.db")


def load_factors_from_db(db: str | None) -> pd.DataFrame:
    """Read FF5 factors from the local financial database (optional dependency)."""
    try:
        from clockdata.store import Store
    except ImportError as exc:
        raise RuntimeError("Local database mode needs clockdata: pip install -e <clockdata>") from exc
    path = Path(db) if db else default_db_path()
    frame = Store(path).load_frame("ff5_factors")
    if frame.empty:
        raise RuntimeError(f"Local database {path} has no ff5_factors. Run: clock-data sync --dataset all")
    print(f"Reading FF5 factors from local financial database {path} (no network calls)")
    return frame


def synthetic_demo(seed: int = 2026, periods: int = 120):
    """Return simulated monthly factor and strategy data for an offline demo."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=periods, freq="ME")
    factors = pd.DataFrame({
        "Mkt-RF": rng.normal(0.006, 0.04, periods),
        "SMB": rng.normal(0.001, 0.025, periods),
        "HML": rng.normal(0.001, 0.025, periods),
        "RMW": rng.normal(0.001, 0.02, periods),
        "CMA": rng.normal(0.001, 0.02, periods),
    }, index=idx)
    rf = pd.Series(0.001, index=idx, name="RF")
    beta = np.array([0.7, 0.15, -0.1, 0.05, 0.1])
    strategy = pd.Series(0.001 + factors.to_numpy() @ beta + rf.to_numpy()
                          + rng.normal(0.0, 0.02, periods), index=idx, name="strategy_return")
    return strategy, factors, rf


def describe_inputs(strategy: pd.Series, factors: pd.DataFrame, rf: pd.Series | None) -> None:
    months = lambda idx: pd.DatetimeIndex(pd.to_datetime(idx)).to_period("M")
    shared = len(set(months(strategy.index)) & set(months(factors.index)))
    print(f"Strategy rows: {len(strategy)}  {strategy.index.min().date()} .. {strategy.index.max().date()}")
    print(f"Factor rows:   {len(factors)}  {factors.index.min().date()} .. {factors.index.max().date()}")
    print(f"Overlapping calendar months: {shared}")
    print(f"Risk-free series: {'supplied' if rf is not None else 'none (raw returns regressed on factors)'}")


def write_result_json(path: Path, result, args: argparse.Namespace,
                      strategy: pd.Series, factors: pd.DataFrame,
                      rf: pd.Series | None) -> None:
    """Persist a machine-readable summary so terminal output is not the only record."""
    coefficients = {}
    for name in result.params.index:
        coefficients[str(name)] = {
            "coef": float(result.params[name]),
            "t": float(result.tvalues[name]),
            "p": float(result.pvalues[name]),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "returns_csv": str(args.returns_csv) if args.returns_csv else None,
        "return_column": args.return_column,
        "factors_source": ("local_csv" if args.factors_csv else
                           "local_database" if args.factors_from_db else
                           "fama_french_download" if args.returns_csv else "synthetic_demo"),
        "maxlags": args.maxlags,
        "use_correction": args.use_correction,
        "risk_free": "supplied" if rf is not None else "none",
        "nobs": int(result.nobs),
        "sample_start": getattr(result, "sample_start", None),
        "sample_end": getattr(result, "sample_end", None),
        "alignment_note": getattr(result, "alignment_note", None),
        "dropped_incomplete_rows": getattr(result, "dropped_incomplete_rows", None),
        "rsquared": float(result.rsquared),
        "factor_names": [str(c) for c in factors.columns],
        "coefficients": coefficients,
        "interpretation": ("The intercept and factor loadings describe this historical sample; "
                           "they are not forecasts or causal effects."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote factor attribution JSON -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="HAC factor regression demo")
    parser.add_argument("--returns-csv", type=Path, help="CSV with date and strategy return column (decimal returns)")
    parser.add_argument("--return-column", default="net_return")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--factors-csv", type=Path, help="optional CSV with date and decimal factor returns; may include RF")
    parser.add_argument("--factors-from-db", action="store_true",
                        help="read FF5 factors from the local financial database instead of downloading")
    parser.add_argument("--db", default=None, help="SQLite path for --factors-from-db (env: CLOCK_DATA_DB)")
    parser.add_argument("--maxlags", type=int, default=3, help="Newey-West/HAC maximum lag")
    parser.add_argument("--use-correction", action="store_true", help="apply the HAC small-sample degrees-of-freedom correction")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="also write a machine-readable summary to this JSON path")
    args = parser.parse_args()

    if args.factors_csv is not None and args.factors_from_db:
        parser.error("--factors-csv and --factors-from-db are mutually exclusive")
    if args.returns_csv is None and (args.factors_csv is not None or args.factors_from_db):
        parser.error("factor inputs require --returns-csv")
    if args.returns_csv is None:
        strategy, factors, rf = synthetic_demo()
        print("Demo data: deterministic synthetic monthly returns (not real market data).")
    else:
        raw = pd.read_csv(args.returns_csv)
        if args.date_column not in raw or args.return_column not in raw:
            parser.error(f"Returns CSV needs {args.date_column!r} and {args.return_column!r} columns")
        raw[args.date_column] = pd.to_datetime(raw[args.date_column], errors="coerce")
        raw[args.return_column] = pd.to_numeric(raw[args.return_column], errors="coerce")
        strategy = raw.dropna(subset=[args.date_column, args.return_column]).set_index(args.date_column)[args.return_column].sort_index()
        if args.factors_csv is not None:
            factor_frame = read_factor_csv(str(args.factors_csv), date_column=args.date_column)
        elif args.factors_from_db:
            factor_frame = load_factors_from_db(args.db)
        else:
            print("Fetching the public Fama-French five-factor file; check its current terms before use.")
            factor_frame = fetch_fama_french_5_factors()
        rf = factor_frame["RF"] if "RF" in factor_frame.columns else None
        factors = factor_frame.drop(columns=["RF"], errors="ignore")

    describe_inputs(strategy, factors, rf)
    result = factor_regression(strategy, factors, maxlags=args.maxlags,
                               risk_free_returns=rf, use_correction=args.use_correction)
    print(f"HAC maxlags: {args.maxlags}  small-sample correction: {args.use_correction}")
    print(f"Observations: {int(result.nobs)}")
    print(f"Sample window: {getattr(result, 'sample_start', 'n/a')} .. {getattr(result, 'sample_end', 'n/a')}")
    print(f"Date alignment: {getattr(result, 'alignment_note', 'n/a')}")
    print("Coefficients:\n", result.params.to_string())
    print("HAC t-statistics:\n", result.tvalues.to_string())
    print("HAC p-values:\n", result.pvalues.to_string())
    print(f"R-squared: {float(result.rsquared):.6f}")
    print("Interpretation: the intercept and factor loadings describe this historical sample; they are not forecasts or causal effects.")
    if args.output_json is not None:
        write_result_json(args.output_json, result, args, strategy, factors, rf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
