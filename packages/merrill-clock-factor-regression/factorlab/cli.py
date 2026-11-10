"""Command line entry point for the two factor-research tracks.

  attribute  -- time-series attribution: strategy excess return on FF5 (OLS + HAC)
  check      -- cross-sectional single-factor testing (RankIC / layers / Fama-MacBeth)
  synthesize -- composite-factor weights from an IC history

The two tracks answer different questions and write separate reports. Neither
is a trading signal generator.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .crosssection import evaluate_factor, rank_ic
from .panels import build_panel, load_prices_from_db
from .synthesis import METHODS, synthesize_weights, weight_stability


def default_db_path() -> Path:
    return Path(os.environ.get("CLOCK_DATA_DB")
                or Path.home() / ".local" / "share" / "clockdata" / "finance.db")


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return path


def cmd_check(args: argparse.Namespace) -> int:
    if args.prices_csv:
        prices = pd.read_csv(args.prices_csv, parse_dates=[args.date_column]).set_index(args.date_column).sort_index()
    else:
        prices = load_prices_from_db(args.db or default_db_path(), args.dataset)
    panel = build_panel(prices)
    wanted = [name.strip() for name in args.factors.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in panel["factors"]]
    if unknown:
        raise SystemExit(f"Unknown factors {unknown}; available: {sorted(panel['factors'])}")

    reports: dict[str, Any] = {}
    ic_history: dict[str, pd.Series] = {}
    for name in wanted:
        factor = panel["factors"][name]
        reports[name] = evaluate_factor(
            factor, panel["forward_returns"],
            n_groups=args.n_groups, min_assets=args.min_assets,
            preprocess=not args.no_preprocess,
        )
        ic_history[name] = rank_ic(factor, panel["forward_returns"], min_assets=args.min_assets)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "track": "cross_sectional_single_factor_test",
        "data_source": ("local_csv" if args.prices_csv else f"clockdata:{args.dataset}"),
        "universe": panel["universe"],
        "universe_type": "ETF price proxies, NOT an A-share stock universe",
        "sample_start": panel["sample_start"],
        "sample_end": panel["sample_end"],
        "n_months": panel["n_months"],
        "n_groups": args.n_groups,
        "min_assets": args.min_assets,
        "preprocess": not args.no_preprocess,
        "factors": reports,
        "caveats": [
            "横截面为 9 只 ETF 代理，不是股票池；IC 数值不可与券商 A 股因子 IC 横向比较",
            "因子仅由月频复权价构造，未使用财务、行业、市值、Barra 数据",
            "IC 与 Fama-MacBeth t 值未做序列相关校正，只作描述性统计",
        ],
    }
    _write_json(args.output_json, payload)
    if args.ic_csv:
        ic_frame = pd.DataFrame(ic_history)
        ic_frame.index.name = "date"
        ic_frame.to_csv(args.ic_csv, index_label="date")
        print(f"Wrote IC history {args.ic_csv}")
    for name, report in reports.items():
        s = report["ic_summary"]
        print(f"{name:10s} RankIC 均值 {s['ic_mean']:+.4f} · ICIR {s['icir']:+.3f} · "
              f"胜率 {s['positive_ratio']:.1%} · 分层形态 {report['layer_monotonicity']['shape']}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.ic_csv, parse_dates=["date"]).set_index("date").sort_index()
    if frame.empty:
        raise SystemExit("IC history CSV is empty")
    result = synthesize_weights(frame, method=args.method, window=args.window,
                                half_life=args.half_life, allow_short=args.allow_short)
    sequence = []
    for end in range(args.window, len(frame) + 1):
        window_frame = frame.iloc[:end].tail(args.window)
        if len(window_frame) >= 2:
            sequence.append(synthesize_weights(window_frame, method=args.method,
                                               window=args.window, half_life=args.half_life,
                                               allow_short=args.allow_short)["weights"])
    result["rolling_weight_stability"] = weight_stability(sequence)
    result["source_rows"] = int(len(frame))
    result["caveats"] = [
        "权重只描述该 IC 历史窗口，不是最优配置",
        "非做空模式下用「负权重截断后重新归一」近似正权重约束，不是求解带约束优化问题",
    ]
    _write_json(args.output_json, result)
    print("Weights: " + ", ".join(f"{k}={v:.3f}" for k, v in result["weights"].items()))
    return 0


def cmd_attribute(args: argparse.Namespace) -> int:
    """Delegate to the existing, tested standalone attribution script."""
    example = Path(__file__).resolve().parent.parent / "examples_factor_regression.py"
    if not example.exists():
        raise SystemExit(f"Attribution script not found: {example}")
    command = [sys.executable, str(example), "--returns-csv", str(args.returns_csv),
               "--return-column", args.return_column, "--maxlags", str(args.maxlags)]
    if args.factors_csv:
        command += ["--factors-csv", str(args.factors_csv)]
    else:
        command += ["--factors-from-db"]
        if args.db:
            command += ["--db", str(args.db)]
    if args.use_correction:
        command.append("--use-correction")
    if args.output_json:
        command += ["--output-json", str(args.output_json)]
    env = os.environ.copy()
    if args.db:
        env["CLOCK_DATA_DB"] = str(args.db)
    return subprocess.run(command, check=False, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="merrill-clock-factor",
        description="Two-track factor research: time-series attribution and cross-sectional testing",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="cross-sectional single-factor test on a local price panel")
    check.add_argument("--prices-csv", type=Path, default=None,
                       help="wide daily price CSV; omit to read the local financial database")
    check.add_argument("--db", type=Path, default=None, help="clockdata SQLite path")
    check.add_argument("--dataset", default="market_prices")
    check.add_argument("--date-column", default="date")
    check.add_argument("--factors", default="mom_1m,mom_3m,mom_6m,mom_12m,vol_3m,vol_12m,rev_1m")
    check.add_argument("--n-groups", type=int, default=5)
    check.add_argument("--min-assets", type=int, default=5)
    check.add_argument("--no-preprocess", action="store_true")
    check.add_argument("--output-json", type=Path, default=Path("cross_section_report.json"))
    check.add_argument("--ic-csv", type=Path, default=None,
                       help="also write a date x factor RankIC history for the synthesize command")
    check.set_defaults(func=cmd_check)

    synth = sub.add_parser("synthesize", help="composite-factor weights from an IC history")
    synth.add_argument("--ic-csv", type=Path, required=True)
    synth.add_argument("--method", choices=METHODS, default="equal")
    synth.add_argument("--window", type=int, default=12)
    synth.add_argument("--half-life", type=float, default=12.0)
    synth.add_argument("--allow-short", action="store_true")
    synth.add_argument("--output-json", type=Path, default=Path("synthesis_report.json"))
    synth.set_defaults(func=cmd_synthesize)

    attr = sub.add_parser("attribute", help="FF5 time-series attribution (OLS + HAC)")
    attr.add_argument("--returns-csv", type=Path, required=True)
    attr.add_argument("--return-column", default="net_return")
    attr.add_argument("--factors-csv", type=Path, default=None)
    attr.add_argument("--db", type=Path, default=None)
    attr.add_argument("--maxlags", type=int, default=3)
    attr.add_argument("--use-correction", action="store_true")
    attr.add_argument("--output-json", type=Path, default=Path("factor_attribution.json"))
    attr.set_defaults(func=cmd_attribute)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
