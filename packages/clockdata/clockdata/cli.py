"""Command-line interface for the local financial database."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .sources import DATASETS, SOURCES, check_connectivity, fetch_dataset
from .store import Store, default_db_path, utcnow


def _store(args: argparse.Namespace) -> Store:
    return Store(args.db).init_schema(sources=SOURCES)


def cmd_init(args: argparse.Namespace) -> int:
    store = _store(args)
    print(f"Database: {store.path}")
    print(f"Registered sources: {len(store.sources())}")
    for row in store.sources().itertuples(index=False):
        print(f"  - {row.source_id:<14} {row.display_name}  [{row.transport}]")
    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    store = _store(args)
    print(store.sources().to_string(index=False))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    for probe in check_connectivity(timeout=args.timeout):
        print(f"[{probe['status']:<12}] {probe['source']:<14} {probe['detail']}")
    print("\nDatasets available for sync: " + ", ".join(sorted(DATASETS)))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    store = _store(args)
    names = sorted(DATASETS) if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]
    for name in names:
        if name not in DATASETS:
            raise SystemExit(f"Unknown dataset {name!r}; available: {sorted(DATASETS)}")
        spec = DATASETS[name]
        started = utcnow()
        print(f"Syncing {name} from {spec['source_id']} ...", flush=True)
        try:
            frame = fetch_dataset(name, start=args.start)
            rows = store.upsert_frame(
                name, spec["source_id"], spec["kind"], frame,
                description=spec["description"], unit=spec["unit"],
                frequency=spec["frequency"],
                params={"requested_start": args.start or spec.get("default_start")},
            )
            store.log_sync(name, spec["source_id"], started, "ok", rows,
                           f"{frame.index.min().date()} .. {frame.index.max().date()}, {len(frame.columns)} series")
            print(f"  wrote {rows:,} observations, {len(frame.columns)} series, "
                  f"{frame.index.min().date()} .. {frame.index.max().date()}")
        except Exception as exc:
            store.log_sync(name, spec["source_id"], started, "failed", 0, f"{type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            if not args.continue_on_error:
                return 1
    print("\nDataset status:")
    print(store.datasets()[["dataset_id", "source_id", "first_date", "last_date", "n_obs", "fetched_at"]]
          .to_string(index=False))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = _store(args)
    frame = store.datasets()
    if frame.empty:
        print("No datasets yet. Run: clock-data sync --dataset all")
        return 0
    print(f"Database: {store.path}  ({store.db_size_bytes():,} bytes)")
    print(frame.to_string(index=False))
    print("\nRecent syncs:")
    print(store.sync_history(limit=args.limit).to_string(index=False))
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    store = _store(args)
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    raw = pd.read_csv(path)
    if args.date_column not in raw.columns:
        raise SystemExit(f"Missing date column {args.date_column!r}; got {raw.columns.tolist()}")
    raw[args.date_column] = pd.to_datetime(raw[args.date_column], errors="coerce")
    frame = (raw.dropna(subset=[args.date_column])
             .set_index(args.date_column).sort_index()
             .apply(pd.to_numeric, errors="coerce").dropna(how="all"))
    if frame.empty:
        raise SystemExit("No numeric rows to import")
    rows = store.upsert_frame(
        args.dataset, args.source, args.kind, frame,
        description=args.description or f"Imported from {path.name}",
        unit=args.unit, frequency=args.frequency,
        params={"file": str(path), "date_column": args.date_column},
    )
    store.log_sync(args.dataset, args.source, utcnow(), "ok", rows, f"import {path.name}")
    print(f"Imported {rows:,} observations into {args.dataset!r} "
          f"({frame.index.min().date()} .. {frame.index.max().date()}, {len(frame.columns)} series)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = _store(args)
    frame = store.load_frame(args.dataset, start=args.start, end=args.end)
    if frame.empty:
        raise SystemExit(f"Dataset {args.dataset!r} has no rows")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index_label="date")
    print(f"Exported {frame.shape[0]} rows x {frame.shape[1]} series -> {target}")
    return 0


def cmd_sql(args: argparse.Namespace) -> int:
    store = _store(args)
    print(store.query(args.query).to_string(index=False))
    return 0


def _write_pipeline_manifest(store: Store, out: Path, *, cross_section_ok: bool) -> None:
    """Record which datasets and outputs this pipeline run actually used."""
    from .manifest import build_manifest, write_manifest

    coverage = store.datasets()
    dataset_ids = sorted(coverage["dataset_id"].tolist()) if not coverage.empty else []
    artifacts = ["metrics.json", "clock_backtest.csv", "factor_attribution.json",
                 "phase_diagnostics.json", "run_manifest.json"]
    if cross_section_ok:
        artifacts += ["cross_section_report.json", "factor_ic_history.csv",
                      "synthesis_report.json"]
    manifest = build_manifest(
        workstation="clockdata-pipeline",
        run_id=f"pipeline-{utcnow()[:10]}",
        dataset_ids=dataset_ids,
        sample_start=str(coverage["first_date"].min()) if not coverage.empty else None,
        sample_end=str(coverage["last_date"].max()) if not coverage.empty else None,
        as_of_rule="local database snapshot; FRED latest-vintage; see run_manifest.json for the strategy's release-lag rule",
        rebalance="monthly",
        execution="T month-end signal -> T+1 month return",
        execution_price="month-end adjusted close",
        universe="9 ETF proxies (SPY/IEF/TLT/GLD/DBC/USO/CPER/DBA/BIL)",
        cost_bps=10.0,
        constraints={"long_only": True, "industry": None, "turnover_cap": None},
        tested_variants=1,
        selection_rule="single configured run; no parameter search performed",
        benchmark="equal-weight monthly rebalance of four core ETF proxies, no costs",
        parameter_confirmations={"maxlags": "default", "start": "default"},
        data_warnings=[
            "本地库为快照，FRED 最新修订值、Yahoo 第三方复权价，均非点时数据",
            "横截面检验的标的池是 ETF 代理，不是 A 股股票池",
        ],
        extra={"artifacts": artifacts, "cross_section_check_completed": cross_section_ok},
    )
    write_manifest(manifest, out / "pipeline_manifest.json")


def cmd_run(args: argparse.Namespace) -> int:
    """Fully offline pipeline: local DB -> backtest -> attribution -> factor check."""
    store = _store(args)
    missing = [name for name in ("fred_macro", "market_prices", "ff5_factors")
               if not store.load_frame(name).shape[0]]
    if missing:
        raise SystemExit(f"Local database is missing: {missing}. Run: clock-data sync --dataset all")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CLOCK_DATA_DB"] = str(store.path)

    print("=== Step 1/3: cross-asset backtest from local database ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "quantclock.cli", "run", "--data-source", "local",
         "--db", str(store.path), "--start", args.start, "--output", str(out)],
        check=True, env=env,
    )

    import factorlab
    example = Path(factorlab.__file__).resolve().parent.parent / "examples_factor_regression.py"
    print("\n=== Step 2/3: FF5 factor attribution from local database ===", flush=True)
    subprocess.run(
        [sys.executable, str(example), "--returns-csv", str(out / "clock_backtest.csv"),
         "--return-column", "net_return", "--factors-from-db", "--db", str(store.path),
         "--maxlags", str(args.maxlags),
         "--output-json", str(out / "factor_attribution.json")],
        check=True, env=env,
    )

    # Step 3 answers a different question from Step 2: does a cross-sectional
    # signal rank next-period returns? Both reports stay separate on purpose.
    cross_section_ok = False
    if not args.no_crosssection:
        print("\n=== Step 3/3: cross-sectional single-factor check from local database ===", flush=True)
        try:
            import factorlab  # noqa: F401  (availability probe)
            subprocess.run(
                [sys.executable, "-m", "factorlab.cli", "check", "--db", str(store.path),
                 "--output-json", str(out / "cross_section_report.json"),
                 "--ic-csv", str(out / "factor_ic_history.csv")],
                check=True, env=env,
            )
            subprocess.run(
                [sys.executable, "-m", "factorlab.cli", "synthesize",
                 "--ic-csv", str(out / "factor_ic_history.csv"),
                 "--method", "ic_half_life", "--window", "12",
                 "--output-json", str(out / "synthesis_report.json")],
                check=False, env=env,
            )
            cross_section_ok = True
        except (ImportError, subprocess.CalledProcessError) as exc:
            print(f"Cross-sectional check skipped: {type(exc).__name__}: {exc}")

    _write_pipeline_manifest(store, out, cross_section_ok=cross_section_ok)
    print(f"\nOutputs: {out.resolve()}")
    if not args.no_dashboard:
        try:
            from .dashboard import build_dashboard
            report = build_dashboard(out, store.path)
            print(f"Dashboard: {report}")
        except Exception as exc:  # a reporting failure must not invalidate the research output
            print(f"Dashboard generation skipped: {type(exc).__name__}: {exc}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import build_dashboard
    out = Path(args.output)
    target = Path(args.out) if args.out else out / "dashboard.html"
    path = build_dashboard(out, Path(args.db), target=target)
    print(f"Dashboard: {path}")
    if args.open:
        subprocess.run(["open", str(path)], check=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="clock-data",
        description="Local financial database for the Merrill-clock research stack",
    )
    parser.add_argument("--db", default=os.environ.get("CLOCK_DATA_DB", str(default_db_path())),
                        help="SQLite database path (env: CLOCK_DATA_DB)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create schema and register data sources").set_defaults(func=cmd_init)
    sub.add_parser("sources", help="list registered data sources").set_defaults(func=cmd_sources)

    check = sub.add_parser("check", help="probe upstream data-source connectivity")
    check.add_argument("--timeout", type=int, default=15)
    check.set_defaults(func=cmd_check)

    sync = sub.add_parser("sync", help="download datasets into the local database")
    sync.add_argument("--dataset", default="all", help="'all' or comma-separated dataset ids")
    sync.add_argument("--start", default=None, help="first date to request (YYYY-MM-DD)")
    sync.add_argument("--continue-on-error", action="store_true")
    sync.set_defaults(func=cmd_sync)

    status = sub.add_parser("status", help="show dataset coverage and sync history")
    status.add_argument("--limit", type=int, default=10)
    status.set_defaults(func=cmd_status)

    imp = sub.add_parser("import", help="import a wide CSV (e.g. a Wind EDB export)")
    imp.add_argument("--file", required=True)
    imp.add_argument("--dataset", required=True)
    imp.add_argument("--source", default="local_csv", choices=[s["source_id"] for s in SOURCES])
    imp.add_argument("--kind", default="macro", choices=("macro", "prices", "factors", "custom"))
    imp.add_argument("--date-column", default="date")
    imp.add_argument("--description", default=None)
    imp.add_argument("--unit", default="")
    imp.add_argument("--frequency", default="")
    imp.set_defaults(func=cmd_import)

    exp = sub.add_parser("export", help="export a dataset back to CSV")
    exp.add_argument("--dataset", required=True)
    exp.add_argument("--out", required=True)
    exp.add_argument("--start", default=None)
    exp.add_argument("--end", default=None)
    exp.set_defaults(func=cmd_export)

    sql = sub.add_parser("sql", help="run a read-only SQL query")
    sql.add_argument("query")
    sql.set_defaults(func=cmd_sql)

    run = sub.add_parser("run", help="offline end-to-end: local DB -> backtest -> factor attribution")
    run.add_argument("--start", default="2000-01-01")
    run.add_argument("--output", default="./outputs/local-run")
    run.add_argument("--maxlags", type=int, default=3)
    run.add_argument("--no-dashboard", action="store_true",
                     help="skip generating the self-contained HTML result dashboard")
    run.add_argument("--no-crosssection", action="store_true",
                     help="skip the cross-sectional single-factor check (step 3)")
    run.set_defaults(func=cmd_run)

    dash = sub.add_parser("dashboard", help="rebuild the self-contained HTML dashboard from saved results")
    dash.add_argument("--output", default="./outputs/local-run", help="directory holding clock_backtest.csv / metrics.json")
    dash.add_argument("--out", default=None, help="HTML path (default: <output>/dashboard.html)")
    dash.add_argument("--open", action="store_true", help="open the report in the default browser")
    dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
