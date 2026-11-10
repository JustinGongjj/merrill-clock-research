"""Public-data adapters for a single A-share research snapshot.

Eastmoney endpoints are public webpage data interfaces, not a stable or
exchange-certified API. Preserve their raw semantics and source timestamps.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

EASTMONEY_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_FLOW = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
EASTMONEY_ANNOUNCEMENTS = "https://np-anotice-stock.eastmoney.com/api/security/ann"
EASTMONEY_STOCK_PAGE = "https://data.eastmoney.com/stockdata/{code}.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; QuantClockResearch/0.2; public-data research)",
    "Referer": "https://quote.eastmoney.com/",
}
KLINE_COLUMNS = [
    "date", "open", "close", "high", "low", "volume_raw",
    "amount_cny", "amplitude_pct", "change_pct", "change_cny", "turnover_pct",
]
FLOW_COLUMNS = [
    "date", "main_net_inflow_cny", "small_net_inflow_cny", "medium_net_inflow_cny",
    "large_net_inflow_cny", "super_large_net_inflow_cny", "main_net_inflow_ratio_pct",
    "small_net_inflow_ratio_pct", "medium_net_inflow_ratio_pct", "large_net_inflow_ratio_pct",
    "super_large_net_inflow_ratio_pct", "close_qfq", "change_pct_vendor", "unused_1", "unused_2",
]


def validate_a_share_code(code: str) -> str:
    code = str(code).strip()
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("A股证券代码应为6位数字，例如 300136")
    return code


def _secid(code: str) -> str:
    code = validate_a_share_code(code)
    # Eastmoney market ids: 1=Shanghai, 0=Shenzhen/Beijing. The initial
    # product explicitly targets mainstream Shanghai/Shenzhen A shares.
    market = "1" if code.startswith(("5", "6", "9")) else "0"
    return f"{market}.{code}"


def _get_json(url: str, params: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("data provider returned a non-object JSON response")
            return payload
        except Exception as exc:  # upstream public endpoints can fail transiently
            last_error = exc
            if attempt == 0:
                time.sleep(0.4)
    raise RuntimeError(f"公开数据源请求失败: {url}: {last_error}") from last_error


def _parse_kline_strings(values: list[str], columns: list[str], value_column: str = "close") -> pd.DataFrame:
    rows = [str(item).split(",") for item in values]
    rows = [row for row in rows if len(row) >= len(columns)]
    if not rows:
        raise RuntimeError("公开行情源未返回可解析的K线记录")
    frame = pd.DataFrame([row[:len(columns)] for row in rows], columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in columns[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date", value_column]).drop_duplicates("date", keep="last").sort_values("date").set_index("date")


def fetch_daily_ohlcv(
    code: str = "300136",
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Fetch daily OHLCV and amount; qfq is the default for chart indicators.

    The returned volume field is kept in the provider's raw unit to avoid
    presenting an unverified scale. Amount is reported in CNY by the source.
    """
    code = validate_a_share_code(code)
    if adjust not in {"", "qfq", "hfq"}:
        raise ValueError("adjust must be '', 'qfq', or 'hfq'")
    begin = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d") if end_date else (date.today() + timedelta(days=2)).strftime("%Y%m%d")
    adjust_code = {"": 0, "qfq": 1, "hfq": 2}[adjust]
    params = {
        "secid": _secid(code),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": str(adjust_code), "beg": begin, "end": end,
    }
    payload = _get_json(EASTMONEY_KLINE, params)
    data = payload.get("data") or {}
    if str(data.get("code", "")) != code:
        raise RuntimeError(f"行情源证券代码不匹配：请求{code}，返回{data.get('code')}")
    frame = _parse_kline_strings(data.get("klines") or [], KLINE_COLUMNS)
    frame.attrs.update({
        "code": code, "name": data.get("name", code), "adjust": adjust,
        "provider": "Eastmoney public quote endpoint",
        "source_url": "https://quote.eastmoney.com/" + ("sh" if _secid(code).startswith("1.") else "sz") + code + ".html",
        "retrieved_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "raw_unit_warning": "volume_raw 保留服务商原始单位；amount_cny 为服务商返回的成交额字段；复权口径为前复权。",
    })
    return frame


def fetch_benchmark_daily(code: str = "399006", start_date: str = "2023-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Fetch an index daily price series as a contextual benchmark (not total return)."""
    code = validate_a_share_code(code)
    begin = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d") if end_date else (date.today() + timedelta(days=2)).strftime("%Y%m%d")
    params = {
        "secid": _secid(code), "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": "0", "beg": begin, "end": end,
    }
    payload = _get_json(EASTMONEY_KLINE, params)
    data = payload.get("data") or {}
    frame = _parse_kline_strings(data.get("klines") or [], KLINE_COLUMNS)
    frame.attrs.update({"code": code, "name": data.get("name", code), "provider": "Eastmoney public quote endpoint", "adjust": "index price series", "retrieved_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")})
    return frame


def fetch_money_flow(code: str = "300136", limit: int = 100) -> pd.DataFrame:
    """Fetch vendor-classified order-size fund-flow data (usually <=100 sessions)."""
    code = validate_a_share_code(code)
    limit = max(1, min(int(limit), 100))
    params = {
        "secid": _secid(code), "lmt": str(limit), "klt": "101",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
    }
    payload = _get_json(EASTMONEY_FLOW, params)
    data = payload.get("data") or {}
    if data and str(data.get("code", "")) != code:
        raise RuntimeError(f"资金流源证券代码不匹配：请求{code}，返回{data.get('code')}")
    frame = _parse_kline_strings(data.get("klines") or [], FLOW_COLUMNS, value_column="close_qfq")
    for col in FLOW_COLUMNS[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.attrs.update({
        "code": code, "name": data.get("name", code),
        "provider": "Eastmoney public fund-flow endpoint",
        "source_url": f"https://data.eastmoney.com/zjlx/detail.html?fc={code}",
        "retrieved_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "caveat": "大/中/小单等为平台分类口径的估算值，不代表交易所确认的机构资金流或投资者身份。",
    })
    return frame


def read_money_flow_csv(path: str, code: str = "300136") -> pd.DataFrame:
    """Load a user-supplied/cached flow extract without inventing missing rows."""
    frame = pd.read_csv(path, parse_dates=["date"])
    required = set(FLOW_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"资金流CSV缺少字段：{', '.join(sorted(missing))}")
    frame = frame[FLOW_COLUMNS].copy().set_index("date").sort_index()
    for col in FLOW_COLUMNS[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    metadata = {}
    sidecar = Path(path).with_suffix(".json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    frame.attrs.update({
        "code": validate_a_share_code(code), "provider": "User-supplied public-data CSV",
        "source_url": metadata.get("source_url", f"https://data.eastmoney.com/zjlx/detail.html?fc={code}"),
        "retrieved_at": metadata.get("captured_at", "未提供采集时间"),
        "caveat": metadata.get("caveat", "输入CSV的采集时间和许可需由使用者核验；不对缺失日期插值。"),
    })
    return frame


def _parse_eastmoney_display_time(value: Any) -> pd.Timestamp:
    """Parse display_time values such as '2026-10-29 16:30:36:518'."""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    text = re.sub(r":(\d{3})$", r".\1", text)
    return pd.to_datetime(text, errors="coerce")


def fetch_announcements(code: str = "300136", limit: int = 20) -> pd.DataFrame:
    """Fetch recent public disclosure titles and timestamps; no sentiment label is inferred."""
    code = validate_a_share_code(code)
    params = {
        "sr": "-1", "page_size": str(max(1, min(int(limit), 100))), "page_index": "1",
        "ann_type": "A", "client_source": "web", "stock_list": code,
    }
    payload = _get_json(EASTMONEY_ANNOUNCEMENTS, params)
    rows = ((payload.get("data") or {}).get("list") or [])
    normalized = []
    for item in rows:
        display_time = item.get("display_time") or item.get("notice_date") or item.get("sort_date")
        normalized.append({
            "published_at": _parse_eastmoney_display_time(display_time),
            "notice_date": pd.to_datetime(item.get("notice_date"), errors="coerce"),
            "title": item.get("title_ch") or item.get("title") or "",
            "category": "; ".join(c.get("column_name", "") for c in item.get("columns", []) if c.get("column_name")),
            "article_code": item.get("art_code", ""),
            "source_url": EASTMONEY_STOCK_PAGE.format(code=code),
        })
    frame = pd.DataFrame(normalized)
    if frame.empty:
        return pd.DataFrame(columns=["published_at", "notice_date", "title", "category", "article_code", "source_url"])
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    frame["notice_date"] = pd.to_datetime(frame["notice_date"], errors="coerce")
    return frame.sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)
