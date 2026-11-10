"""Unified registry of financial data sources with capability-based routing.

This module is the spine of the local database: instead of hard-coding one
provider per dataset, every upstream is declared once with the *capabilities* it
can serve (A-share / US equities / crypto / macro / factors / fundamentals),
whether it needs a key, and how it behaves under failure. Callers then ask for
a capability and get an ordered chain of providers to try.

Design rules that matter for research integrity:

* **No silent substitution.** A fallback provider is recorded in the result so
  the caller can state which upstream actually served the data.
* **No credential guessing.** Providers needing a key are marked unavailable
  unless the key environment variable is present; they never prompt.
* **Failures are reported, not swallowed.** :func:`probe_all` returns per
  source status with the concrete error, so a broken upstream is visible in
  the dashboard instead of masquerading as "no data".

The catalogue below was assembled from the providers' own documented APIs and
verified by live probes on the host machine; the ``verified`` field records
whether a real request succeeded during development, not a marketing claim.
"""
from __future__ import annotations

import importlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import pandas as pd

# Capability tags. A provider may serve several.
CAP_A_SHARE = "a_share"
CAP_US_EQUITY = "us_equity"
CAP_HK_EQUITY = "hk_equity"
CAP_CRYPTO = "crypto"
CAP_ETF = "etf"
CAP_MACRO = "macro"
CAP_FACTORS = "factors"
CAP_FUNDAMENTALS = "fundamentals"
CAP_FUND_FLOW = "fund_flow"
CAP_DISCLOSURE = "disclosure"

ALL_CAPABILITIES = (
    CAP_A_SHARE, CAP_US_EQUITY, CAP_HK_EQUITY, CAP_CRYPTO, CAP_ETF,
    CAP_MACRO, CAP_FACTORS, CAP_FUNDAMENTALS, CAP_FUND_FLOW, CAP_DISCLOSURE,
)


# --------------------------------------------------------------------------- #
# Import cache
# --------------------------------------------------------------------------- #
# Availability checks call ``import_module`` for every optional SDK. Doing that
# on every request is wasteful, and doing it inside a probe thread that later
# times out is actively dangerous: CPython serialises imports of the same
# module behind a per-module lock, so an abandoned thread caught mid-import
# keeps that lock and every later ``import_module`` call in the process blocks
# until the abandoned thread finishes. Caching the verdict per module means
# each SDK is imported at most once and never re-imported afterwards.
_IMPORT_CACHE: dict[str, tuple[bool, str]] = {}
_IMPORT_LOCK = threading.Lock()


def _import_status(module: str) -> tuple[bool, str]:
    """Import ``module`` once and remember whether it is usable.

    Returns ``(available, detail)``. The second call for the same module never
    touches the import system again.
    """
    with _IMPORT_LOCK:
        cached = _IMPORT_CACHE.get(module)
    if cached is not None:
        return cached
    try:
        importlib.import_module(module)
        result = (True, f"已安装 {module}")
    except Exception as exc:  # noqa: BLE001 - optional dependency
        result = (False, f"未安装 {module}（{type(exc).__name__}）")
    with _IMPORT_LOCK:
        _IMPORT_CACHE[module] = result
    return result


def _mark_import_pending(module: str) -> None:
    """Record that an import was attempted but did not finish in time.

    This prevents a later call from re-entering ``import_module`` for a module
    whose import lock may still be held by the abandoned probe thread.
    """
    with _IMPORT_LOCK:
        _IMPORT_CACHE.setdefault(module, (False, f"导入超时（{module}）"))


@dataclass(frozen=True)
class SourceSpec:
    """Static description of one upstream data source."""

    source_id: str
    display_name: str
    capabilities: tuple[str, ...]
    transport: str                       # python | http | mcp | file
    requires_key: bool = False
    key_env: str | None = None
    tier: int = 3                       # 1 = most authoritative for its coverage
    latency: str = "unknown"            # realtime | delayed | eod | daily | monthly
    note: str = ""
    homepage: str = ""
    module: str | None = None           # python module to import for availability
    fee: str = "free"
    verified: bool = False              # live request succeeded during development

    def key_present(self) -> bool:
        if not self.requires_key:
            return True
        return bool(self.key_env and os.environ.get(self.key_env))

    def available(self) -> bool:
        """True when the source can actually be used on this host right now."""
        if not self.key_present():
            return False
        if self.module:
            return _import_status(self.module)[0]
        return True

    def status_detail(self) -> str:
        if not self.key_present():
            return f"缺少环境变量 {self.key_env}"
        if self.module:
            ok, detail = _import_status(self.module)
            if ok:
                return "可用"
            return detail
        return "可用"


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #
CATALOGUE: tuple[SourceSpec, ...] = (
    # ---- A-share ---------------------------------------------------------- #
    SourceSpec(
        "baostock", "Baostock A股历史日线（自有服务器直推·免注册）",
        (CAP_A_SHARE, CAP_ETF, CAP_HK_EQUITY), "python", module="baostock",
        tier=1, latency="eod", verified=True,
        note="每日固定时点入库（17:30 日K / 18:00 复权因子 / 次日 1:30 财报），非爬虫，稳定性最强；"
             "无实时行情，返回字符串需自行转类型。",
        homepage="http://baostock.com",
    ),
    SourceSpec(
        "akshare", "AkShare A股/港美股/宏观/另类（免费·覆盖最广·1100+ 接口）",
        (CAP_A_SHARE, CAP_US_EQUITY, CAP_HK_EQUITY, CAP_ETF, CAP_MACRO,
         CAP_FUNDAMENTALS, CAP_FUND_FLOW, CAP_DISCLOSURE),
        "python", module="akshare", tier=2, latency="delayed", verified=True,
        note="爬虫聚合，上游改版即失效，高频调用易封 IP；指数成分股只有当前快照（无历史），"
             "财务数据无 point-in-time，回测需自行处理幸存者偏差与前视偏差。",
        homepage="https://akshare.akfamily.xyz",
    ),
    SourceSpec(
        "efinance", "efinance 东方财富行情/基金/可转债（免 key）",
        (CAP_A_SHARE, CAP_ETF), "python", module="efinance",
        tier=2, latency="delayed",
        note="直连东方财富公开接口，与 AkShare 部分重叠，可作为交叉校验源。",
        homepage="https://github.com/Micro-sheep/efinance",
    ),
    SourceSpec(
        "mootdx", "mootdx 通达信行情（A股/ETF/可转债/板块·免 key）",
        (CAP_A_SHARE, CAP_ETF), "python", module="mootdx",
        tier=2, latency="delayed",
        note="读取通达信本地/在线行情服务器，字段为国内交易终端口径，适合做量价与板块维度。",
        homepage="https://github.com/mootdx/mootdx",
    ),
    SourceSpec(
        "adata", "adata 多源聚合行情（东财/新浪/腾讯）",
        (CAP_A_SHARE, CAP_ETF), "python", module="adata",
        tier=2, latency="delayed",
        note="一个接口聚合多家东财/新浪/腾讯源，单源失败可自动切换。",
        homepage="https://github.com/1nchaos/adata",
    ),
    SourceSpec(
        "tushare", "Tushare Pro（A股/港美股·积分制）",
        (CAP_A_SHARE, CAP_HK_EQUITY, CAP_US_EQUITY, CAP_ETF, CAP_MACRO,
         CAP_FACTORS, CAP_FUNDAMENTALS),
        "python", requires_key=True, key_env="TUSHARE_TOKEN", module="tushare",
        tier=1, latency="eod", fee="积分制（2000 积分档约 ¥200/年）",
        note="统一 ts_code 规范与 DataFrame 结构，是 A 股免费源中最接近专业 API 的；"
             "积分档与分钟线/港美股/新闻研报为独立权限，需分开计费。",
        homepage="https://tushare.pro",
    ),
    # ---- Global equity / crypto ------------------------------------------ #
    SourceSpec(
        "yahoo", "Yahoo Finance 复权价代理（全球·免 key）",
        (CAP_US_EQUITY, CAP_ETF, CAP_CRYPTO, CAP_HK_EQUITY),
        "python", module="yfinance", tier=2, latency="delayed", verified=True,
        note="非官方第三方代理，2025-09 曾因 Cookie/Crumb 校验升级全球停摆；"
             "服务条款限个人研究、禁商用与再分发，不可进生产链路。",
        homepage="https://finance.yahoo.com",
    ),
    SourceSpec(
        "alpha_vantage", "Alpha Vantage（全球股票/外汇/加密/技术指标）",
        (CAP_US_EQUITY, CAP_ETF, CAP_CRYPTO, CAP_FUNDAMENTALS),
        "http", requires_key=True, key_env="ALPHA_VANTAGE_API_KEY",
        tier=2, latency="delayed", fee="免费 25 次/日",
        note="免费额度是全市场最紧的一档；基本面历史仅约 5 年。",
        homepage="https://www.alphavantage.co",
    ),
    SourceSpec(
        "eodhd", "EODHD 全球 EOD（60+ 交易所·15 万+ 标的）",
        (CAP_US_EQUITY, CAP_ETF, CAP_CRYPTO, CAP_HK_EQUITY, CAP_FUNDAMENTALS),
        "http", requires_key=True, key_env="EODHD_API_KEY",
        tier=2, latency="eod", fee="免费 20 次/日；全市场 $19.99/月",
        note="不使用交易所原始行情，为 OTC/平台聚合 VWAP，报价具指示性；标准套餐限个人使用。",
        homepage="https://eodhd.com",
    ),
    SourceSpec(
        "polygon", "Polygon / Massive 美股行情",
        (CAP_US_EQUITY), "http", requires_key=True, key_env="POLYGON_API_KEY",
        tier=2, latency="eod", fee="免费 5 次/分（仅 EOD）",
        note="免费档仅日线；实时自 $29/月起。",
        homepage="https://polygon.io",
    ),
    SourceSpec(
        "tiingo", "Tiingo 美股/外汇/加密 + 新闻",
        (CAP_US_EQUITY, CAP_CRYPTO, CAP_FUNDAMENTALS),
        "http", requires_key=True, key_env="TIINGO_API_KEY",
        tier=2, latency="eod", fee="免费 1000 次/日",
        note="个人套餐限 internal use，不可对外展示或分享。",
        homepage="https://www.tiingo.com",
    ),
    SourceSpec(
        "finnhub", "Finnhub 实时美股 + 另类数据",
        (CAP_US_EQUITY, CAP_FUNDAMENTALS, CAP_FUND_FLOW),
        "http", requires_key=True, key_env="FINNHUB_API_KEY",
        tier=2, latency="realtime", fee="免费 60 次/分",
        note="免费档即可拿到实时美股报价与 WebSocket；个人非商用。",
        homepage="https://finnhub.io",
    ),
    SourceSpec(
        "ccxt", "CCXT 加密货币（100+ 交易所统一接口）",
        (CAP_CRYPTO,), "python", module="ccxt",
        tier=2, latency="realtime", verified=True,
        note="统一 REST/WS 接口覆盖现货与衍生品；各交易所速率与地域限制不同。",
        homepage="https://github.com/ccxt/ccxt",
    ),
    # ---- Macro / factors -------------------------------------------------- #
    SourceSpec(
        "fred", "FRED 圣路易斯联储宏观序列（80 万+ · 免密 CSV 端点）",
        (CAP_MACRO,), "python", module="requests", tier=1, latency="daily",
        verified=True,
        note="公开 CSV 端点无需密钥；返回最新修订值，做点时研究须改用 ALFRED vintage（需 key）。",
        homepage="https://fred.stlouisfed.org",
    ),
    SourceSpec(
        "alfred", "ALFRED 点时(vintage)宏观数据",
        (CAP_MACRO,), "http", requires_key=True, key_env="FRED_API_KEY",
        tier=1, latency="daily",
        note="回测点时数据的正确来源；需 FRED API Key。",
        homepage="https://alfred.stlouisfed.org",
    ),
    SourceSpec(
        "world_bank", "World Bank Open Data（约 16000 指标·免 key）",
        (CAP_MACRO,), "http", tier=2, latency="monthly",
        note="CC BY 4.0，可商用（需遵守附加条款）；多为年度频率，有发布时滞。",
        homepage="https://data.worldbank.org",
    ),
    SourceSpec(
        "dbnomics", "DBnomics 元聚合器（7 亿+ 序列·免 key·无限速）",
        (CAP_MACRO,), "http", tier=1, latency="daily",
        note="镜像 IMF/世行/欧统/ECB/BIS 等 80+ 机构并保留每次修订，作为宏观检索层。",
        homepage="https://db.nomics.world",
    ),
    SourceSpec(
        "fama_french", "Kenneth French FF5/动量因子（免 key）",
        (CAP_FACTORS,), "python", tier=1, latency="monthly", verified=True,
        note="原始值为百分点，入库前统一换算为小数；使用前核对数据使用条款。",
        homepage="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
    ),
    # ---- Terminals / session --------------------------------------------- #
    SourceSpec(
        "wind_mcp", "Wind Alice 万得金融数据（WorkBuddy 会话内 MCP）",
        (CAP_A_SHARE, CAP_US_EQUITY, CAP_HK_EQUITY, CAP_MACRO, CAP_FUNDAMENTALS),
        "mcp", tier=1, latency="eod",
        note="在 WorkBuddy 会话内经 wind-finance 连接器取数，导出 CSV 后由 clock-data import 入库。",
    ),
    SourceSpec(
        "windpy", "WindPy 万得量化接口（Windows 专用·需授权终端）",
        (CAP_A_SHARE, CAP_US_EQUITY, CAP_HK_EQUITY, CAP_MACRO, CAP_FUNDAMENTALS),
        "python", module="WindPy", tier=1, latency="eod",
        note="仅 Windows + 授权 Wind 终端可运行；macOS 不可导入，调用会显式报错并降级。",
    ),
    SourceSpec(
        "local_db", "本地 SQLite 金融数据库（离线兜底）",
        (CAP_A_SHARE, CAP_US_EQUITY, CAP_HK_EQUITY, CAP_ETF, CAP_MACRO,
         CAP_FACTORS, CAP_CRYPTO),
        "file", tier=5, latency="n/a", verified=True,
        note="已落库的数据集，离线可用、结果可复现；是所有在线源的兜底路径，"
             "但覆盖范围取决于此前同步过哪些数据集。",
    ),
    SourceSpec(
        "local_csv", "本地 CSV / 人工导入",
        (CAP_A_SHARE, CAP_US_EQUITY, CAP_MACRO, CAP_FACTORS),
        "file", tier=5, latency="n/a",
        note="通用宽表导入：一列日期 + 若干数值列；也是所有外部源的最终落地方式。",
    ),
)

BY_ID: dict[str, SourceSpec] = {s.source_id: s for s in CATALOGUE}


def sources_for(capability: str, *, require_available: bool = True) -> list[SourceSpec]:
    """Return sources serving ``capability``, best tier first.

    Ordering is by (tier, latency) so authoritative-but-slow sources are tried
    before convenient scrapers, and within a tier the faster feed wins.
    """
    if capability not in ALL_CAPABILITIES:
        raise KeyError(f"Unknown capability {capability!r}; known: {ALL_CAPABILITIES}")
    hits = [s for s in CATALOGUE if capability in s.capabilities]
    if require_available:
        hits = [s for s in hits if s.available()]
    latency_rank = {"realtime": 0, "delayed": 1, "eod": 2, "daily": 3, "monthly": 4, "n/a": 9, "unknown": 8}
    return sorted(hits, key=lambda s: (s.tier, latency_rank.get(s.latency, 8)))


def register_sources(store: Any) -> int:
    """Register the whole catalogue into a ``Store`` so foreign keys resolve.

    ``Store.upsert_frame`` references ``sources(source_id)``; any source that
    has not been registered yet would fail the foreign-key check. Calling this
    keeps the database consistent with the catalogue.
    """
    rows = []
    for spec in CATALOGUE:
        rows.append({
            "source_id": spec.source_id,
            "display_name": spec.display_name,
            "endpoint": spec.homepage or None,
            "transport": spec.transport,
            "requires_key": int(spec.requires_key),
            "note": spec.note,
        })
    store.init_schema(rows)
    return len(rows)


def registry_rows() -> list[dict[str, Any]]:
    """Flatten the catalogue for display in the dashboard / API."""
    rows = []
    for spec in CATALOGUE:
        rows.append({
            "source_id": spec.source_id,
            "display_name": spec.display_name,
            "capabilities": list(spec.capabilities),
            "transport": spec.transport,
            "requires_key": spec.requires_key,
            "key_env": spec.key_env,
            "tier": spec.tier,
            "latency": spec.latency,
            "fee": spec.fee,
            "homepage": spec.homepage,
            "note": spec.note,
            "available": spec.available(),
            "status_detail": spec.status_detail(),
            "verified_live": spec.verified,
        })
    return rows


def _guarded(fn: Callable[["_Deadline"], tuple[bool, str]], timeout: float) -> tuple[bool, str]:
    """Run ``fn`` with a hard wall-clock deadline.

    ``fn`` receives a :class:`_Deadline` it should check periodically; that
    lets a cooperative worker stop cleanly instead of being orphaned.

    Returns ``(ok, detail)``. On timeout the worker thread is abandoned (it is
    a daemon thread, so it cannot keep the process alive) and the source is
    reported as failed with a timeout message rather than hanging the caller.
    """
    deadline = _Deadline(timeout)
    box: dict[str, Any] = {}

    def _runner() -> None:
        try:
            ok, detail = fn(deadline)
            box["result"] = (ok, detail)
        except Exception as exc:  # noqa: BLE001
            box["result"] = (False, f"{type(exc).__name__}: {str(exc)[:140]}")

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout)
    deadline.expire()
    if "result" in box:
        return box["result"]
    return False, f"探测超时（>{int(timeout)}s），上游无响应"


class _Deadline:
    """Cooperative cancellation token handed to a guarded probe."""

    __slots__ = ("_deadline", "_expired")

    def __init__(self, timeout: float):
        self._deadline = time.monotonic() + timeout
        self._expired = False

    def expire(self) -> None:
        self._expired = True

    @property
    def expired(self) -> bool:
        return self._expired or time.monotonic() >= self._deadline

    def remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())


def probe_all(timeout: int = 12) -> list[dict[str, Any]]:
    """Run a live, minimal request against each Python/HTTP source.

    The probe is deliberately cheap (one small request each) and never raises:
    it returns a status row per source so callers can render the health of the
    whole backend at a glance.

    Every network-touching step runs inside :func:`_guarded`, which enforces a
    hard wall-clock deadline on a worker thread. Upstream SDKs such as Baostock
    do not expose a socket timeout and can block indefinitely on an unreachable
    server; without the guard a single dead host would hang the whole probe
    (and the REST endpoint that calls it).
    """
    results: list[dict[str, Any]] = []

    def _add(spec: SourceSpec, ok: bool, detail: str) -> None:
        results.append({
            "source": spec.source_id,
            "display_name": spec.display_name,
            "status": "ok" if ok else ("unavailable" if not spec.key_present() else "failed"),
            "detail": detail,
            "latency_ms": None,
        })

    # -- FRED (no key needed)
    spec = BY_ID["fred"]
    started = time.perf_counter()
    try:
        import requests
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": "INDPRO", "cosd": "2025-01-01"}, timeout=timeout)
        ok = r.ok and "INDPRO" in r.text[:300]
        _add(spec, ok, f"HTTP {r.status_code}, {len(r.content)} bytes"
             if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        _add(spec, False, f"{type(exc).__name__}: {str(exc)[:120]}")
    results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)

    # -- Yahoo
    spec = BY_ID["yahoo"]
    started = time.perf_counter()
    try:
        import requests
        r = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/SPY",
                         params={"range": "5d", "interval": "1d"}, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        _add(spec, r.ok, f"HTTP {r.status_code}, {len(r.content)} bytes" if r.ok else f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        _add(spec, False, f"{type(exc).__name__}: {str(exc)[:120]}")
    results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)

    # -- Kenneth French
    spec = BY_ID["fama_french"]
    started = time.perf_counter()
    try:
        import requests
        r = requests.head(
            "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
            timeout=timeout, allow_redirects=True)
        _add(spec, r.ok, f"HTTP {r.status_code}")
    except Exception as exc:  # noqa: BLE001
        _add(spec, False, f"{type(exc).__name__}: {str(exc)[:120]}")
    results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)

    # -- Baostock (login + one quote). Its SDK has no socket timeout, so the
    #    whole call is wrapped in the deadline guard. The import happens on the
    #    calling thread (and is cached) so an aborted probe cannot leave the
    #    module's import lock held.
    spec = BY_ID["baostock"]
    started = time.perf_counter()
    baostock_ok, baostock_detail = _import_status("baostock")
    if not baostock_ok:
        _add(spec, False, baostock_detail)
    else:
        import baostock as bs

        def _probe_baostock(deadline: _Deadline) -> tuple[bool, str]:
            lg = bs.login()
            if lg.error_code != "0":
                return False, f"login error_code={lg.error_code}: {lg.error_msg}"
            try:
                rs = bs.query_history_k_data_plus(
                    "sh.600000", "date,close", start_date="2024-01-01", end_date="2024-01-10",
                    frequency="d", adjustflag="3")
                # Baostock's cursor only advances when ``get_row_data()`` is
                # called; ``next()`` keeps returning True. Looping on ``next()``
                # alone is an infinite spin that holds the GIL and starves every
                # other thread in the process, so the row read is mandatory here.
                rows = 0
                while rs.error_code == "0" and rs.next():
                    rs.get_row_data()
                    rows += 1
                    if rows > 5000 or deadline.expired:
                        break
                return True, f"登录成功，样例查询返回 {rows} 行"
            finally:
                try:
                    bs.logout()
                except Exception:  # noqa: BLE001 - logout failure must not mask the result
                    pass

        ok, detail = _guarded(_probe_baostock, timeout)
        _add(spec, ok, detail)
    results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)

    # -- AkShare (one daily bar)
    spec = BY_ID["akshare"]
    started = time.perf_counter()
    akshare_ok, akshare_detail = _import_status("akshare")
    if not akshare_ok:
        _add(spec, False, akshare_detail)
    else:
        import akshare as ak

        def _probe_akshare(deadline: _Deadline) -> tuple[bool, str]:
            df = ak.stock_zh_a_hist(symbol="600000", period="daily",
                                    start_date="20240102", end_date="20240110", adjust="qfq")
            return bool(df is not None and not df.empty), \
                f"东财前复权日线返回 {0 if df is None else len(df)} 行"

        ok, detail = _guarded(_probe_akshare, timeout)
        _add(spec, ok, detail)
    results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)

    # -- CCXT (exchange list only, no network call needed)
    spec = BY_ID["ccxt"]
    ccxt_ok, ccxt_detail = _import_status("ccxt")
    if ccxt_ok:
        import ccxt
        _add(spec, True, f"已安装，可访问 {len(ccxt.exchanges)} 个交易所")
    else:
        _add(spec, False, ccxt_detail)

    # -- Optional SDKs: import-only availability. The import itself is run
    #    through the cache so it happens at most once per process.
    for sid in ("efinance", "mootdx", "adata", "tushare"):
        spec = BY_ID[sid]
        started = time.perf_counter()
        ok, detail = _import_status(spec.module or "")
        if ok:
            detail = "已安装（按需联网取数）"
        _add(spec, ok, detail)
        results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)

    # -- Key-gated HTTP providers: report only, never call without a key
    for sid in ("alpha_vantage", "eodhd", "polygon", "tiingo", "finnhub", "alfred",
                "world_bank", "dbnomics"):
        spec = BY_ID[sid]
        if not spec.key_present():
            _add(spec, False, f"需要 {spec.key_env}，未配置时跳过实测")
        elif spec.source_id == "world_bank":
            started = time.perf_counter()
            try:
                import requests
                r = requests.get("https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.MKTP.CD",
                                 params={"format": "json", "per_page": 1}, timeout=timeout)
                _add(spec, r.ok, f"HTTP {r.status_code}")
            except Exception as exc:  # noqa: BLE001
                _add(spec, False, f"{type(exc).__name__}: {str(exc)[:120]}")
            results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)
        elif spec.source_id == "dbnomics":
            started = time.perf_counter()
            try:
                import requests
                r = requests.get("https://api.db.nomics.world/v22/series/FRED/INDPRO",
                                 params={"observations": 1}, timeout=timeout)
                _add(spec, r.ok, f"HTTP {r.status_code}")
            except Exception as exc:  # noqa: BLE001
                _add(spec, False, f"{type(exc).__name__}: {str(exc)[:120]}")
            results[-1]["latency_ms"] = int((time.perf_counter() - started) * 1000)
        else:
            _add(spec, True, f"已配置 {spec.key_env}，可调用")

    # -- Session-only / platform / file
    _add(BY_ID["wind_mcp"], True,
         "WorkBuddy 会话内连接器 wind-finance；脚本侧经 CSV 导入，连通性由会话工具验证")
    windpy = BY_ID["windpy"]
    _add(windpy, windpy.available(),
         "已安装并可用" if windpy.available() else "本机不可导入（仅 Windows + 授权 Wind 终端）")
    _add(BY_ID["local_db"], True, "本地 SQLite 数据库（离线兜底）")
    _add(BY_ID["local_csv"], True, "本地文件，无需网络")
    return results


# --------------------------------------------------------------------------- #
# Capability-routed fetch with automatic fallback
# --------------------------------------------------------------------------- #
@dataclass
class FetchOutcome:
    """Result of a routed fetch: which source served it, and what failed."""

    frame: pd.DataFrame
    source_id: str | None
    attempts: list[dict[str, str]] = field(default_factory=list)
    degraded: bool = False

    def manifest(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "degraded": self.degraded,
            "attempts": self.attempts,
        }


def routed_fetch(
    capability: str,
    fetchers: dict[str, Callable[[], pd.DataFrame]],
    *,
    require_non_empty: bool = True,
) -> FetchOutcome:
    """Try each source that serves ``capability`` until one returns usable data.

    ``fetchers`` maps a ``source_id`` to a zero-argument callable. Sources
    without a matching entry are skipped. Every attempt is recorded so the
    caller can report exactly which upstream answered and which ones failed.
    """
    attempts: list[dict[str, str]] = []
    for spec in sources_for(capability, require_available=True):
        fn = fetchers.get(spec.source_id)
        if fn is None:
            continue
        try:
            frame = fn()
        except Exception as exc:  # noqa: BLE001 - try the next source
            attempts.append({"source": spec.source_id, "outcome": "error",
                             "detail": f"{type(exc).__name__}: {str(exc)[:160]}"})
            continue
        if frame is None or (require_non_empty and frame.empty):
            attempts.append({"source": spec.source_id, "outcome": "empty",
                             "detail": "返回空数据"})
            continue
        attempts.append({"source": spec.source_id, "outcome": "ok",
                         "detail": f"{frame.shape[0]} 行 × {frame.shape[1]} 列"})
        # "Degraded" means a fallback actually happened: an earlier, higher
        # priority source failed or returned nothing. It does not mean "the
        # caller only offered one source".
        degraded = any(a["outcome"] in ("error", "empty") for a in attempts[:-1])
        return FetchOutcome(frame=frame, source_id=spec.source_id,
                            attempts=attempts, degraded=degraded)
    raise RuntimeError(
        f"No source could serve capability {capability!r}. Attempts: {attempts}"
    )


def capability_matrix() -> dict[str, list[str]]:
    """Map every capability to the ordered source chain that serves it."""
    return {
        cap: [s.source_id for s in sources_for(cap, require_available=False)]
        for cap in ALL_CAPABILITIES
    }
