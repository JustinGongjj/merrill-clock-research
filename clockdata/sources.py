"""Source catalogue and fetch adapters for the local financial database.

Three adapters run directly from Python (FRED, Yahoo, Kenneth French). Wind
Alice is reachable through the WorkBuddy session; its series are imported as
CSV via ``clock-data import`` so every row still lands in the same audited
store. No credentials are embedded here.
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import requests

SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "fred",
        "display_name": "FRED 宏观经济序列",
        "endpoint": "https://fred.stlouisfed.org/graph/fredgraph.csv",
        "transport": "python",
        "requires_key": 0,
        "note": "公开 CSV 端点，无需密钥；返回最新修订值，非点时(vintage)数据",
    },
    {
        "source_id": "yahoo",
        "display_name": "Yahoo Finance 调整收盘价",
        "endpoint": "https://query1.finance.yahoo.com/v8/finance/chart",
        "transport": "python",
        "requires_key": 0,
        "note": "第三方复权价代理；研究用途，非交易所认证行情",
    },
    {
        "source_id": "fama_french",
        "display_name": "Kenneth French FF5 月频因子",
        "endpoint": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "transport": "python",
        "requires_key": 0,
        "note": "原始值为百分点，入库前统一换算为小数；使用前核对数据使用条款",
    },
    {
        "source_id": "wind_mcp",
        "display_name": "Wind Alice 万得金融数据",
        "endpoint": "WorkBuddy MCP: wind-finance",
        "transport": "workbuddy-mcp",
        "requires_key": 0,
        "note": "在 WorkBuddy 会话内取数(EDB/行情)，导出 CSV 后用 clock-data import 入库",
    },
    {
        "source_id": "akshare",
        "display_name": "AkShare A股行情（免费·跨平台）",
        "endpoint": "https://akshare.akfamily.xyz",
        "transport": "python",
        "requires_key": 0,
        "note": "免费、macOS/Linux/Windows 可用；经东方财富取 A股前复权日线，无需 token",
    },
    {
        "source_id": "windpy",
        "display_name": "WindPy 万得量化接口（Windows 专用）",
        "endpoint": "Wind 终端 COM/API",
        "transport": "python",
        "requires_key": 0,
        "note": "仅 Windows + 授权 Wind 终端可运行；macOS 不可导入，调用会显式报错并降级",
    },
    {
        "source_id": "local_csv",
        "display_name": "本地 CSV / 人工导入",
        "endpoint": "file://",
        "transport": "file",
        "requires_key": 0,
        "note": "通用宽表导入：一列日期 + 若干数值列",
    },
]

DATASETS: dict[str, dict[str, Any]] = {
    "fred_macro": {
        "source_id": "fred", "kind": "macro", "frequency": "月",
        "description": "美林时钟宏观输入：工业生产 INDPRO、CPI CPIAUCSL（最新修订值）",
        "unit": "指数点", "default_start": "1990-01-01",
    },
    "market_prices": {
        "source_id": "yahoo", "kind": "prices", "frequency": "日",
        "description": "跨资产 ETF 复权收盘价代理（SPY/IEF/TLT/GLD/DBC/USO/CPER/DBA/BIL）",
        "unit": "价格", "default_start": "2000-01-01",
    },
    "ff5_factors": {
        "source_id": "fama_french", "kind": "factors", "frequency": "月",
        "description": "Fama-French 五因子 + 无风险利率（小数收益）",
        "unit": "小数收益", "default_start": None,
    },
    "ashare_prices": {
        "source_id": "akshare", "kind": "prices", "frequency": "日",
        "description": "A股真实前复权日线（沪深300样本，max_stocks=120 默认取样）",
        "unit": "价格", "default_start": "2010-01-01",
    },
    "tb3ms": {
        "source_id": "fred", "kind": "rates", "frequency": "日",
        "description": "3个月美国国债收益率 TB3MS（年利率%，FRED 免密端点）；用于现金 sleeve 替代零收益假设",
        "unit": "年利率%", "default_start": "1990-01-01",
    },
}


def _fetch_fred(start: str) -> pd.DataFrame:
    from quantclock.data import DEFAULT_MACRO_SERIES, fetch_fred_panel
    return fetch_fred_panel(series=DEFAULT_MACRO_SERIES, start=start)


def _fetch_yahoo(start: str) -> pd.DataFrame:
    from quantclock.data import fetch_market_prices, yahoo_healthy
    if not yahoo_healthy():
        raise RuntimeError(
            "Yahoo upstream health probe failed; the local dataset was left unchanged. "
            "Retry the sync later, or use clock-data run for the offline pipeline."
        )
    return fetch_market_prices(start=start)


def _fetch_ff5(start: str | None = None) -> pd.DataFrame:
    from factorlab.data import fetch_fama_french_5_factors
    return fetch_fama_french_5_factors()


def _fetch_ashare(start: str) -> pd.DataFrame:
    from .providers import fetch_ashare_daily
    panel, failed = fetch_ashare_daily(start=start, max_stocks=120)
    if failed:
        print(f"[ashare] {len(failed)}/{panel.shape[1] + len(failed)} codes failed; "
              f"using {panel.shape[1]} for the panel.")
    return panel


def _fetch_tb3ms(start: str) -> pd.DataFrame:
    from .providers import fetch_tb3ms
    return fetch_tb3ms(start=start)


FETCHERS: dict[str, Callable[..., pd.DataFrame]] = {
    "fred_macro": _fetch_fred,
    "market_prices": _fetch_yahoo,
    "ff5_factors": _fetch_ff5,
    "ashare_prices": _fetch_ashare,
    "tb3ms": _fetch_tb3ms,
}


def fetch_dataset(dataset_id: str, start: str | None = None) -> pd.DataFrame:
    """Fetch one registered dataset from its upstream source."""
    if dataset_id not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset_id!r}; available: {sorted(DATASETS)}")
    spec = DATASETS[dataset_id]
    effective_start = start or spec.get("default_start") or "1990-01-01"
    return FETCHERS[dataset_id](effective_start)


def check_connectivity(timeout: int = 15) -> list[dict[str, Any]]:
    """Probe each Python-reachable source with a minimal request."""
    probes: list[dict[str, Any]] = []

    def record(source_id: str, ok: bool, detail: str, status: str = "ok") -> None:
        probes.append({"source": source_id, "status": status if ok else "failed", "detail": detail})

    try:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": "INDPRO", "cosd": "2025-01-01"}, timeout=timeout)
        record("fred", r.ok and "INDPRO" in r.text[:300],
               f"HTTP {r.status_code}, {len(r.content)} bytes" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:  # network probe: report, never abort
        record("fred", False, f"{type(exc).__name__}: {str(exc)[:120]}")

    try:
        r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/SPY",
                         params={"range": "5d", "interval": "1d"}, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        record("yahoo", r.ok, f"HTTP {r.status_code}, {len(r.content)} bytes" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:
        record("yahoo", False, f"{type(exc).__name__}: {str(exc)[:120]}")

    try:
        r = requests.head(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
            timeout=timeout, allow_redirects=True)
        record("fama_french", r.ok, f"HTTP {r.status_code}")
    except Exception as exc:
        record("fama_french", False, f"{type(exc).__name__}: {str(exc)[:120]}")

    record("wind_mcp", True,
           "WorkBuddy 会话内连接器 wind-finance；脚本侧经 CSV 导入，连通性由会话工具验证",
           status="session-only")
    try:
        from .providers import windpy_available
        ok = windpy_available()
        record("windpy", ok,
               "已安装并可用" if ok else "本机不可导入（仅 Windows + 授权 Wind 终端）",
               status="ok" if ok else "unavailable")
    except Exception as exc:  # noqa: BLE001
        record("windpy", False, f"{type(exc).__name__}: {str(exc)[:120]}", status="unavailable")
    try:
        from .providers import csi300_universe
        _ = csi300_universe  # import probe only; live pull is rate-limited
        record("akshare", True, "AkShare 已安装；A股行情经东方财富，无需 token",
               status="ok")
    except Exception as exc:  # noqa: BLE001
        record("akshare", False, f"{type(exc).__name__}: {str(exc)[:120]}", status="unavailable")
    record("local_csv", True, "本地文件，无需网络", status="local")
    return probes
