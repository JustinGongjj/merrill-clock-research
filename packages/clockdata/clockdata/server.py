"""Local backend data service for the research stack.

A standard-library HTTP server (no extra dependencies) that exposes the
research data as a small REST API so the workstation can act as a *backend*
for other tools -- the HTML constellation page, a chat assistant, a
spreadsheet, or any local script.

Endpoints
---------
``GET /api/health``              service + database status
``GET /api/sources``             registered data sources and their capability
``GET /api/sources/probe``       live connectivity probe (slow, ~seconds)
``GET /api/capabilities``        capability -> source routing matrix
``GET /api/datasets``            datasets stored in the local database
``GET /api/dataset/<id>``        one dataset as a wide table (JSON)
``GET /api/instruments``         symbol universe understood by the stack
``GET /api/instrument/<symbol>`` resolved instrument metadata
``GET /api/constellation``       full constellation payload (JSON)
``GET /api/constellation.html``  rendered standalone HTML page
``GET /api/factors``             cross-sectional factor table
``GET /api/forecast``            phase / factor / price projections

Only the standard library is used so the service starts with no extra install
and no network dependency. Bind to 127.0.0.1 by default.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_DB = Path.home() / ".local" / "share" / "clockdata" / "finance.db"


class _Handler(BaseHTTPRequestHandler):
    """Request handler. Holds a reference to the service state on the server."""

    server_version = "ClockDataAPI/1.0"

    # -- helpers --------------------------------------------------------- #
    @property
    def state(self) -> "DataService":
        return self.server.state  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str,
                          allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "verbose", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    # -- routing --------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        try:
            if route in ("/", "/api"):
                self._send_json(self.state.index())
                return
            if route == "/api/health":
                self._send_json(self.state.health())
                return
            if route == "/api/sources":
                self._send_json(self.state.sources())
                return
            if route == "/api/sources/probe":
                self._send_json(self.state.probe(timeout=int(query.get("timeout", 12))))
                return
            if route == "/api/capabilities":
                self._send_json(self.state.capabilities())
                return
            if route == "/api/datasets":
                self._send_json(self.state.datasets())
                return
            if route.startswith("/api/dataset/"):
                self._send_json(self.state.dataset(route.split("/", 3)[-1]))
                return
            if route == "/api/instruments":
                self._send_json(self.state.instruments())
                return
            if route.startswith("/api/instrument/"):
                self._send_json(self.state.instrument(route.split("/", 3)[-1]))
                return
            if route == "/api/constellation":
                self._send_json(self.state.constellation(
                    query.get("symbol", "300136"), query.get("output"),
                    offline=query.get("live", "0") not in ("1", "true")))
                return
            if route == "/api/constellation.html":
                html = self.state.constellation_html(
                    query.get("symbol", "300136"), query.get("output"),
                    offline=query.get("live", "0") not in ("1", "true"))
                self._send_html(html)
                return
            if route == "/api/factors":
                self._send_json(self.state.factors(query.get("output")))
                return
            if route == "/api/forecast":
                self._send_json(self.state.forecast(
                    query.get("symbol", "300136"), query.get("output")))
                return
            self._error(f"未知端点 {route}", status=404)
        except Exception as exc:  # noqa: BLE001 - never kill the server
            self._error(f"{type(exc).__name__}: {exc}", status=500)


class DataService:
    """Holds the shared state and implements the endpoint logic."""

    def __init__(self, db_path: str | Path | None = None,
                 output_dir: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB
        self.output_dir = Path(output_dir) if output_dir else None
        self._lock = threading.Lock()

    # -- endpoints ------------------------------------------------------- #
    def index(self) -> dict[str, Any]:
        return {
            "service": "clockdata backend",
            "version": "1.0",
            "endpoints": [
                "/api/health", "/api/sources", "/api/sources/probe",
                "/api/capabilities", "/api/datasets", "/api/dataset/<id>",
                "/api/instruments", "/api/instrument/<symbol>",
                "/api/constellation", "/api/constellation.html",
                "/api/factors", "/api/forecast",
            ],
        }

    def health(self) -> dict[str, Any]:
        db_ok = self.db_path.exists()
        size = self.db_path.stat().st_size if db_ok else 0
        datasets = 0
        observations = 0
        if db_ok:
            try:
                from clockdata.store import Store
                store = Store(self.db_path)
                datasets = int(len(store.datasets()))
                observations = int(store.query(
                    "SELECT COUNT(*) AS n FROM observations")["n"].iloc[0])
            except Exception as exc:  # noqa: BLE001
                return {"status": "degraded", "error": f"{type(exc).__name__}: {exc}",
                        "database": str(self.db_path), "generated_at": _now()}
        return {
            "status": "ok" if db_ok else "degraded",
            "database": str(self.db_path),
            "database_exists": db_ok,
            "database_bytes": size,
            "datasets": datasets,
            "observations": observations,
            "generated_at": _now(),
        }

    def sources(self) -> dict[str, Any]:
        from clockdata.registry import registry_rows
        return {"sources": registry_rows(), "generated_at": _now()}

    def probe(self, timeout: int = 12) -> dict[str, Any]:
        from clockdata.registry import probe_all
        rows = probe_all(timeout=timeout)
        return {
            "probes": rows,
            "ok": sum(1 for r in rows if r["status"] == "ok"),
            "total": len(rows),
            "generated_at": _now(),
        }

    def capabilities(self) -> dict[str, Any]:
        from clockdata.registry import capability_matrix
        return {"routing": capability_matrix(), "generated_at": _now()}

    def datasets(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"datasets": [], "note": "本地数据库不存在", "generated_at": _now()}
        from clockdata.store import Store
        frame = Store(self.db_path).datasets()
        records = frame.to_dict("records") if not frame.empty else []
        return {"datasets": records, "count": len(records), "generated_at": _now()}

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"error": "本地数据库不存在", "dataset_id": dataset_id}
        from clockdata.store import Store
        frame = Store(self.db_path).load_frame(dataset_id)
        if frame.empty:
            return {"error": f"数据集 {dataset_id!r} 不存在或为空", "dataset_id": dataset_id}
        return {
            "dataset_id": dataset_id,
            "shape": list(frame.shape),
            "columns": [str(c) for c in frame.columns],
            "first_date": str(frame.index[0])[:10],
            "last_date": str(frame.index[-1])[:10],
            "records": [
                {"date": str(idx)[:10], **{str(k): (None if pd_isna(v) else v)
                                           for k, v in row.items()}}
                for idx, row in frame.tail(2000).to_dict("index").items()
            ],
            "generated_at": _now(),
        }

    def instruments(self) -> dict[str, Any]:
        from clockdata.universe import BUILTIN_PROXIES
        return {
            "builtins": [
                {"symbol": k, "market": v["market"], "kind": v["kind"], "label": v["label"]}
                for k, v in BUILTIN_PROXIES.items()
            ],
            "accepted_formats": [
                "A股 6 位数字（600519 / 000001 / 300136）",
                "美股 ticker（AAPL / MSFT / SPY）",
                "加密对（BTC/USDT / ETH/USDT）",
            ],
            "generated_at": _now(),
        }

    def instrument(self, symbol: str) -> dict[str, Any]:
        from clockdata.universe import describe_sources_for, resolve
        inst = resolve(symbol)
        return {
            "instrument": inst.to_dict(),
            "sources": describe_sources_for(inst),
            "generated_at": _now(),
        }

    def constellation(self, symbol: str, output: str | None = None,
                      offline: bool = True) -> dict[str, Any]:
        """Assemble the constellation payload.

        ``offline=True`` (the default for HTTP requests) never touches the
        network: it reads the local database and the saved research output.
        That keeps the API responsive even when an upstream is slow or down.
        """
        from starmap.assemble import build_payload
        with self._lock:
            return build_payload(symbol, db_path=self.db_path,
                                 output_dir=output or self.output_dir,
                                 offline=offline)

    def constellation_html(self, symbol: str, output: str | None = None,
                           offline: bool = True) -> str:
        from starmap.viz import render_html
        payload = self.constellation(symbol, output, offline=offline)
        return render_html(payload)

    def factors(self, output: str | None = None) -> dict[str, Any]:
        from starmap.assemble import _load_factor_stats
        stats = _load_factor_stats(output or self.output_dir)
        if not stats:
            return {"available": False,
                    "note": "未找到 cross_section_report.json；先运行 clock-data run",
                    "factors": [], "generated_at": _now()}
        return {"available": True, "factors": stats["table"],
                "raw": stats["raw"], "generated_at": _now()}

    def forecast(self, symbol: str, output: str | None = None) -> dict[str, Any]:
        payload = self.constellation(symbol, output)
        return {
            "symbol": symbol,
            "forecasts": payload.get("forecasts", {}),
            "clouds": payload.get("clouds", {}),
            "caveats": payload.get("caveats", []),
            "generated_at": _now(),
        }


def pd_isna(value: Any) -> bool:
    try:
        import pandas as pd
        result = pd.isna(value)
        return bool(result) if isinstance(result, bool) else False
    except Exception:  # noqa: BLE001
        return False


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def serve(host: str = "127.0.0.1", port: int = 8765,
          db_path: str | Path | None = None,
          output_dir: str | Path | None = None,
          verbose: bool = False,
          state: "DataService | None" = None) -> None:
    """Start the backend service (blocking).

    ``state`` lets a caller inject a pre-built :class:`DataService` (the CLI
    does this so it can reuse an already-resolved database path).
    """
    if state is None:
        state = DataService(db_path=db_path, output_dir=output_dir)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.state = state          # type: ignore[attr-defined]
    httpd.verbose = verbose      # type: ignore[attr-defined]
    print(f"◆ clockdata 后台数据服务已启动：http://{host}:{port}")
    print(f"  数据库：{state.db_path}")
    print(f"  星图页：http://{host}:{port}/api/constellation.html?symbol=300136")
    print("  按 Ctrl+C 停止")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n■ 服务已停止")
    finally:
        httpd.server_close()
