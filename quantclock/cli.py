"""Command line entry point for a reproducible baseline run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .backtest import performance_metrics, prices_to_monthly_returns, run_backtest
from .data import DEFAULT_MARKET_TICKERS, fetch_fred_panel, fetch_market_prices, write_csv
from .macro import make_clock_features
from .signals import default_research_tilts, map_regimes_to_assets, regime_probabilities
from .company_analysis import build_company_snapshot
from .company_report import write_snapshot


def run_research(args: argparse.Namespace) -> int:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print("Downloading public macro proxies from FRED and adjusted-price proxies from Yahoo Finance...")
    macro = fetch_fred_panel(start=args.start)
    prices = fetch_market_prices(start=args.start)
    returns = prices_to_monthly_returns(prices)
    macro.to_csv(out / "macro_latest_vintage.csv", index_label="date")
    prices.to_csv(out / "daily_adjusted_prices_proxy.csv", index_label="date")
    returns.to_csv(out / "monthly_asset_returns.csv", index_label="date")

    features = make_clock_features(
        macro["industrial_production"], macro["cpi"], returns.index,
        ip_release_lag_days=args.ip_lag_days,
        cpi_release_lag_days=args.cpi_lag_days,
    )
    probabilities = regime_probabilities(features, window=args.z_window, smooth=args.smooth)
    asset_map = {sleeve: ticker for sleeve, ticker in {
        "equity": "SPY", "treasury_intermediate": "IEF",
        "gold": "GLD", "broad_commodities": "DBC", "cash": "BIL",
    }.items() if ticker in returns.columns}
    if not {"equity", "treasury_intermediate", "gold", "broad_commodities"}.issubset(asset_map):
        raise RuntimeError(f"Downloaded market data lacks core sleeves. Got {returns.columns.tolist()}")
    # Do not silently turn an unlaunched ETF's missing history into a zero-return sleeve.
    returns = returns.dropna(subset=list(asset_map.values()))
    weights = map_regimes_to_assets(probabilities, asset_map, default_research_tilts())
    result = run_backtest(returns, weights, cost_bps=args.cost_bps)
    combined = features.join(probabilities.add_prefix("p_"), how="outer").join(weights.add_prefix("w_"), how="outer").join(result, how="outer")
    combined.to_csv(out / "clock_backtest.csv", index_label="date")
    metrics = performance_metrics(result["net_return"])
    metrics.update({"observations": int(result["net_return"].notna().sum()), "cost_bps_per_one_way_turnover": args.cost_bps,
                    "weights": "illustrative research tilts; not an investment recommendation", "data_warning": "FRED latest revisions + Yahoo adjusted-price proxies"})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {out.resolve()}")
    return 0


def run_company_snapshot(args: argparse.Namespace) -> int:
    """Build a public-data, five-dimension company research snapshot."""
    print(f"Fetching public A-share data for {args.code}; no orders will be placed...")
    summary = build_company_snapshot(
        code=args.code,
        start_date=args.start,
        benchmark_code=args.benchmark,
        flow_limit=args.flow_limit,
        announcement_limit=args.announcement_limit,
        policy_file=args.policy_file,
        include_us_clock=not args.skip_us_clock,
        flow_csv=args.flow_csv,
    )
    paths = write_snapshot(args.output, summary)
    print(json.dumps({
        "security": f"{summary['metadata'].get('name')} ({args.code})",
        "price_range": [summary["metadata"].get("price_start"), summary["metadata"].get("price_end")],
        "dimensions": ["technical", "vendor flow", "public disclosures", "trading sentiment proxy", "policy keyword watch"],
        "merrill_clock": summary.get("merrill_clock", {}).get("phase_cn", summary.get("merrill_clock", {}).get("note")),
        "output": paths,
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="quantclock", description="Public-data Merrill-clock research scaffold")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="download public proxies and run the baseline backtest")
    run.add_argument("--start", default="2000-01-01")
    run.add_argument("--output", default="./outputs")
    run.add_argument("--cost-bps", type=float, default=10.0)
    run.add_argument("--ip-lag-days", type=int, default=20)
    run.add_argument("--cpi-lag-days", type=int, default=20)
    run.add_argument("--z-window", type=int, default=60)
    run.add_argument("--smooth", type=int, default=3)
    run.set_defaults(func=run_research)

    company = sub.add_parser("company", help="build an A-share company five-dimension research snapshot")
    company.add_argument("--code", default="300136", help="6-digit A-share code; default 300136")
    company.add_argument("--start", default="2023-01-01", help="daily history start date (YYYY-MM-DD)")
    company.add_argument("--benchmark", default="399006", help="benchmark index code; default 399006 ChiNext Index")
    company.add_argument("--flow-limit", type=int, default=100, help="provider fund-flow history limit (max 100)")
    company.add_argument("--announcement-limit", type=int, default=20, help="number of recent disclosure titles")
    company.add_argument("--flow-csv", default=None, help="optional local fund-flow CSV; must contain the documented schema")
    company.add_argument("--policy-file", default=None, help="policy watchlist CSV; defaults to data/policy_watchlist.csv")
    company.add_argument("--skip-us-clock", action="store_true", help="skip the legacy US FRED Merrill-clock context")
    company.add_argument("--output", default="./outputs/300136", help="snapshot output directory")
    company.set_defaults(func=run_company_snapshot)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
