"""Command-line entry point for the cross-asset Merrill-clock baseline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .backtest import performance_metrics, prices_to_monthly_returns, run_backtest
from .calibration import phase_diagnostics, temperature_sensitivity
from .data import fetch_fred_panel, fetch_market_prices, yahoo_healthy
from .macro import make_clock_features
from .signals import default_research_tilts, map_regimes_to_assets, regime_probabilities

CORE_SLEEVES = {
    "equity": "SPY",
    "treasury_intermediate": "IEF",
    "gold": "GLD",
    "broad_commodities": "DBC",
}
CASH_ZERO_COLUMN = "__cash_zero__"


def default_db_path() -> Path:
    return Path(os.environ.get("CLOCK_DATA_DB")
                or Path.home() / ".local" / "share" / "clockdata" / "finance.db")


def _fetch_live(args: argparse.Namespace):
    print("Fetching FRED macro observations and Yahoo Finance adjusted-price proxies...")
    return fetch_fred_panel(start=args.start), fetch_market_prices(start=args.start)


def _load_local(args: argparse.Namespace):
    """Fully offline read from the local financial database (lazy import)."""
    try:
        from clockdata.store import Store
    except ImportError as exc:
        raise RuntimeError(
            "Local database mode needs the clockdata package: pip install -e <clockdata>"
        ) from exc
    db = Path(args.db) if args.db else default_db_path()
    store = Store(db)
    macro = store.load_frame("fred_macro", start=args.start)
    prices = store.load_frame("market_prices", start=args.start)
    if macro.empty or prices.empty:
        raise RuntimeError(
            f"Local database {db} is missing fred_macro / market_prices. "
            "Run: clock-data sync --dataset all"
        )
    missing = {"industrial_production", "cpi"} - set(macro.columns)
    if missing:
        raise RuntimeError(f"Local fred_macro dataset is missing columns: {sorted(missing)}")
    print(f"Reading from local financial database {db} (no network calls)")
    return macro, prices


def load_market_inputs(args: argparse.Namespace) -> tuple[object, object]:
    """Live fetch (default), offline local read, or auto fallback.

    The clockdata package is imported lazily so this repository stays usable on
    its own; only local/auto modes require it. A fallback is always recorded in
    metrics.json so the data provenance is never ambiguous.
    """
    if args.data_source in ("live", "auto"):
        return _fetch_live(args)
    return _load_local(args)


def run_with_fallback(args: argparse.Namespace) -> int:
    """Run the baseline; ``auto`` retries the whole pipeline from the local DB.

    Catching at pipeline level (rather than only at fetch time) also covers the
    case where the upstream returns partial columns and no complete month has
    all four core sleeves. The fallback is recorded in metrics.json.
    """
    if args.data_source != "auto":
        return run_research(args)
    if not yahoo_healthy():
        print("Upstream price probe failed; using the local financial database directly.")
        args.data_source = "local-fallback"
        return run_research(args)
    try:
        return run_research(args)
    except (RuntimeError, ValueError) as exc:
        print(f"Live path failed ({type(exc).__name__}: {exc}); falling back to the local database.")
        args.data_source = "local-fallback"
        return run_research(args)


RESEARCH_DEFAULTS = {
    "start": "2000-01-01", "cost_bps": 10.0, "ip_lag_days": 20, "cpi_lag_days": 20,
    "z_window": 60, "smooth": 3, "cash_policy": "auto", "temperature": 1.0,
}


def write_run_manifest(args: argparse.Namespace, out: Path, *, sample_start: str,
                       sample_end: str, universe: list[str], dataset_ids: list[str]) -> dict:
    """Persist assumptions next to the results, with a confirmation gate.

    Parameters left at their parser defaults are recorded as ``default`` and
    surfaced as warnings: silent defaults may run, but cannot hide in the
    output. Requires the clockdata package; a standalone install skips it.
    """
    try:
        from clockdata.manifest import (
            build_manifest, parameter_confirmations, validate_manifest, write_manifest,
        )
    except ImportError:
        print("clockdata not installed; skipping run_manifest.json (install the local database package to enable it).")
        return {}
    confirmations = parameter_confirmations(args, RESEARCH_DEFAULTS)
    warnings = [
        "FRED 返回最新修订值，不是点时 vintage 数据",
        "Yahoo 为第三方复权价代理，非交易所认证行情",
        "宏观发布滞后为固定天数近似，不是真实发布日历",
    ]
    unconfirmed = [k for k, v in confirmations.items() if v == "default"]
    if unconfirmed:
        warnings.append("以下研究参数未显式确认，使用默认值：" + ", ".join(unconfirmed))
    manifest = build_manifest(
        workstation="merrill-clock-quant",
        run_id=f"clock-{sample_start}-{sample_end}",
        dataset_ids=dataset_ids,
        sample_start=sample_start,
        sample_end=sample_end,
        as_of_rule=(f"month-end observation + {getattr(args, 'ip_lag_days', 20)}d / "
                    f"{getattr(args, 'cpi_lag_days', 20)}d release lag; FRED latest-vintage"),
        rebalance="monthly",
        execution="T month-end signal -> T+1 month return (weights shifted one period)",
        execution_price="month-end adjusted close",
        universe=", ".join(universe),
        cost_bps=float(getattr(args, "cost_bps", 10.0)),
        slippage_bps=None,
        constraints={"long_only": True, "leverage": False, "industry": None,
                     "turnover_cap": None, "cash_sleeve": getattr(args, "cash_policy", "auto")},
        tested_variants=1,
        selection_rule="single configured run; no parameter search performed",
        benchmark="equal-weight monthly rebalance of the four core ETF proxies, no costs",
        parameter_confirmations=confirmations,
        data_warnings=warnings,
        extra={"data_source": getattr(args, "data_source", "unknown"),
               "smooth_span": getattr(args, "smooth", 3),
               "z_window": getattr(args, "z_window", 60),
               "temperature": getattr(args, "temperature", 1.0)},
    )
    write_manifest(manifest, out / "run_manifest.json")
    issues = validate_manifest(manifest)
    for issue in issues:
        print(f"[manifest] {issue}")
    return manifest


def run_research(args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    macro, prices = load_market_inputs(args)
    returns = prices_to_monthly_returns(prices, drop_incomplete_period=not args.keep_incomplete_month)
    macro.to_csv(out / "macro_latest_vintage.csv", index_label="date")
    prices.to_csv(out / "daily_adjusted_prices_proxy.csv", index_label="date")
    returns.to_csv(out / "monthly_asset_returns.csv", index_label="date")

    core_map = {sleeve: ticker for sleeve, ticker in CORE_SLEEVES.items() if ticker in returns.columns}
    if not set(CORE_SLEEVES).issubset(core_map):
        raise RuntimeError(f"Downloaded market data lacks core sleeves. Got {returns.columns.tolist()}")
    core_columns = list(core_map.values())
    usable = returns.dropna(subset=core_columns).copy()
    dropped_months = len(returns) - len(usable)
    if usable.empty:
        raise RuntimeError(
            f"No complete month has all four core sleeves {core_columns}. "
            "Check upstream price coverage, or run offline from the local database: "
            "merrill-clock-quant run --data-source local"
        )

    # The cash sleeve should not truncate the whole sample just because BIL has
    # a shorter listing history. Default: explicit zero-return cash accrual
    # before BIL exists, with the assumption recorded in metrics.json.
    asset_map = dict(core_map)
    bil_complete = "BIL" in usable.columns and bool(usable["BIL"].notna().all())
    if args.cash_policy == "require_bil" and not bil_complete:
        raise RuntimeError("BIL history is incomplete; use --cash-policy auto or zero-cash")
    if args.cash_policy in ("zero",) or not bil_complete:
        usable[CASH_ZERO_COLUMN] = 0.0
        asset_map["cash"] = CASH_ZERO_COLUMN
        cash_note = "zero-return cash accrual; understates T-bill income before BIL history starts"
    else:
        asset_map["cash"] = "BIL"
        cash_note = "BIL adjusted-price proxy"

    features = make_clock_features(
        macro["industrial_production"], macro["cpi"], usable.index,
        ip_release_lag_days=args.ip_lag_days,
        cpi_release_lag_days=args.cpi_lag_days,
    )
    temperature = float(getattr(args, "temperature", 1.0))
    probabilities = regime_probabilities(features, window=args.z_window,
                                         temperature=temperature, smooth=args.smooth)
    weights = map_regimes_to_assets(probabilities, asset_map, default_research_tilts())
    result = run_backtest(usable, weights, cost_bps=args.cost_bps)

    # Phase probabilities are a logistic decomposition, not calibrated forecasts.
    # Record that limitation numerically next to the backtest, and show how
    # decisiveness changes with the temperature parameter.
    diagnostics: dict = {}
    try:
        diagnostics = {
            "calibration": phase_diagnostics(probabilities, features),
            "temperature_sensitivity": temperature_sensitivity(
                features, temperatures=(0.5, 1.0, 2.0),
                window=args.z_window, smooth=args.smooth),
        }
    except (ValueError, KeyError) as exc:
        diagnostics = {"error": f"{type(exc).__name__}: {exc}"}
    (out / "phase_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # A z-score window plus 6/12-month differencing leaves an initial warm-up
    # with all-zero weights. Excluding those rows prevents a flat cash period
    # from diluting volatility and inflating the Sharpe ratio.
    invested = weights.sum(axis=1).shift(1).fillna(0.0) > 0
    if not invested.any():
        raise RuntimeError("No usable weights: macro history is too short for the configured z-window")
    evaluated = result.loc[invested, "net_return"]
    metrics = performance_metrics(evaluated)
    benchmark_returns = usable[core_columns].loc[invested].mean(axis=1)
    benchmark = performance_metrics(benchmark_returns)

    combined = (features.join(probabilities.add_prefix("p_"), how="outer")
                .join(weights.add_prefix("w_"), how="outer").join(result, how="outer"))
    evaluated_flag = invested.reindex(combined.index).fillna(False)
    combined["evaluated"] = evaluated_flag.astype(int)
    # Warm-up months were not invested; exporting their zero net return as a
    # strategy observation would bias any downstream factor attribution.
    combined.loc[~evaluated_flag, "net_return"] = float("nan")
    combined.to_csv(out / "clock_backtest.csv", index_label="date")
    metrics.update({
        "observations": int(evaluated.notna().sum()),
        "sample_start": str(evaluated.index.min().date()),
        "sample_end": str(evaluated.index.max().date()),
        "warmup_months_excluded": int((~invested).sum()),
        "output_flag": "clock_backtest.csv: evaluated=1 rows enter metrics; warm-up net_return is NaN",
        "months_dropped_for_missing_core_prices": int(dropped_months),
        "cash_sleeve": cash_note,
        "cost_bps_per_one_way_turnover": args.cost_bps,
        "benchmark": "equal-weight monthly rebalance of the four core ETF proxies, no costs",
        "benchmark_cagr": benchmark["cagr"],
        "benchmark_sharpe_zero_rf": benchmark["sharpe_zero_rf"],
        "benchmark_max_drawdown": benchmark["max_drawdown"],
        "weights": "illustrative research tilts; not an investment recommendation",
        "data_source": args.data_source,
        "data_warning": "FRED latest revisions + Yahoo adjusted-price proxies; not point-in-time data",
        "temperature": temperature,
        "phase_diagnostics": "phase_diagnostics.json (Brier / quadrant hit rate / entropy / temperature sensitivity)",
    })
    calibration = diagnostics.get("calibration", {})
    metrics["phase_quadrant_hit_rate"] = calibration.get("quadrant_hit_rate")
    metrics["phase_brier_growth_up"] = calibration.get("brier_growth_up")
    metrics["phase_hit_rate_minus_baseline"] = calibration.get("hit_rate_minus_baseline")

    if not getattr(args, "no_manifest", False):
        write_run_manifest(
            args, out,
            sample_start=str(evaluated.index.min().date()),
            sample_end=str(evaluated.index.max().date()),
            universe=sorted(set(asset_map.values())),
            dataset_ids=(["fred_macro", "market_prices"]
                         if str(args.data_source).startswith("local") else
                         ["fred_upstream", "yahoo_upstream"]),
        )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved generated outputs to {out.resolve()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="merrill-clock-quant",
        description="Public-data Merrill-clock cross-asset research baseline (not trading advice)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="fetch public proxies and run the illustrative baseline")
    run.add_argument("--start", default="2000-01-01", help="first date to request (YYYY-MM-DD)")
    run.add_argument("--output", default="./outputs", help="directory for generated data and metrics")
    run.add_argument("--cost-bps", type=float, default=10.0)
    run.add_argument("--ip-lag-days", type=int, default=20)
    run.add_argument("--cpi-lag-days", type=int, default=20)
    run.add_argument("--z-window", type=int, default=60)
    run.add_argument("--smooth", type=int, default=3)
    run.add_argument("--temperature", type=float, default=1.0,
                     help="logistic temperature for phase probabilities (lower = more decisive)")
    run.add_argument("--no-manifest", action="store_true",
                     help="skip run_manifest.json (enabled by default when clockdata is installed)")
    run.add_argument("--cash-policy", choices=("auto", "zero", "require_bil"), default="auto",
                     help="cash sleeve: use BIL when its history is complete, else explicit zero-return cash")
    run.add_argument("--keep-incomplete-month", action="store_true",
                     help="keep a partially elapsed month labelled with a future month-end date")
    run.add_argument("--data-source", choices=("live", "local", "auto"), default="live",
                     help="live = fetch upstream; local = offline database; auto = live with automatic local fallback")
    run.add_argument("--db", default=None,
                     help="SQLite path for --data-source local (env: CLOCK_DATA_DB)")
    run.set_defaults(func=run_with_fallback)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
