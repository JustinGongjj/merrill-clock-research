"""Command-line interface for the constellation (starmap) tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assemble import build_payload
from .viz import write_html


def _safe_stem(symbol: str) -> str:
    """Turn a symbol into a filesystem-safe file stem.

    Crypto pairs such as ``BTC/USDT`` contain a slash, which would otherwise
    create a non-existent nested path.
    """
    stem = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in symbol)
    return stem.strip("_") or "symbol"


def cmd_build(args: argparse.Namespace) -> int:
    """Build a constellation payload and render it to a standalone HTML file."""
    payload = build_payload(
        args.symbol,
        db_path=args.db,
        output_dir=args.output,
        include_instrument=not args.no_instrument,
        include_sources=not args.no_sources,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(args.symbol)

    json_path = out_dir / f"constellation_{stem}.json"
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"payload 无法序列化为 JSON（{exc}）。这通常意味着某个数值是 NaN 或 "
            f"存在未被转换的对象——请检查数据源。"
        ) from exc
    json_path.write_text(text, encoding="utf-8")

    html_path = write_html(payload, out_dir / f"constellation_{stem}.html")

    graph = payload["graph"]
    print(f"✓ payload : {json_path}")
    print(f"✓ html    : {html_path}")
    print(f"  节点 {len(graph['nodes'])} · 连线 {len(graph['links'])}")
    for section, blob in payload["forecasts"].items():
        state = "可用" if blob.get("available") else f"不可用（{blob.get('note', '')}）"
        print(f"  预测 · {section}: {state}")
    for section, blob in payload["clouds"].items():
        state = "可用" if blob.get("available") else f"不可用（{blob.get('note', '')}）"
        print(f"  云图 · {section}: {state}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Probe every registered data source and print the health table."""
    from clockdata.registry import probe_all

    rows = probe_all(timeout=args.timeout)
    width = max(len(r["source"]) for r in rows) + 2
    print(f"{'源'.ljust(width)}{'状态'.ljust(12)}说明")
    for row in rows:
        status = row["status"]
        print(f"{row['source'].ljust(width)}{status.ljust(12)}{row['detail']}")
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\n{ok}/{len(rows)} 个源当前可用")
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Print the capability -> source routing matrix."""
    from clockdata.registry import capability_matrix

    matrix = capability_matrix()
    width = max(len(k) for k in matrix) + 2
    for cap, chain in matrix.items():
        print(f"{cap.ljust(width)}{' → '.join(chain)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the local backend data service."""
    from clockdata.server import serve

    serve(host=args.host, port=args.port, db_path=args.db)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="starmap",
        description="美林时钟研究台 · 科幻星图可视化（任意标的可复用）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="构建星图并输出自包含 HTML")
    build.add_argument("symbol", help="标的代码（A股 6 位 / 美股 ticker / 加密对）")
    build.add_argument("--out", default="./outputs/starmap", help="输出目录")
    build.add_argument("--db", default=None, help="本地数据库路径")
    build.add_argument("--output", default=None,
                       help="已有研究结果目录（含 metrics.json / cross_section_report.json）")
    build.add_argument("--no-instrument", action="store_true", help="跳过个股快照层")
    build.add_argument("--no-sources", action="store_true", help="跳过数据源层")
    build.set_defaults(func=cmd_build)

    probe = sub.add_parser("probe", help="实测各数据源连通性")
    probe.add_argument("--timeout", type=int, default=12)
    probe.set_defaults(func=cmd_probe)

    matrix = sub.add_parser("matrix", help="打印能力→数据源路由矩阵")
    matrix.set_defaults(func=cmd_matrix)

    serve = sub.add_parser("serve", help="启动本地后台数据服务")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--db", default=None, help="本地数据库路径")
    serve.add_argument("--output", default=None,
                       help="研究结果目录（含 clock_backtest.csv / cross_section_report.json）")
    serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
