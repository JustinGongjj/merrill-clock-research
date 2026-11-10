"""Assemble a constellation payload from real, measured data.

This is the layer that makes the visualisation reusable: it takes a *symbol*
plus the local database and produces the whole payload, without any
instrument-specific code. Point it at 300136, AAPL or BTC/USDT and it produces
the same structure with that instrument's own numbers.

Data sources consumed (all optional -- a missing source degrades that section
and says so, rather than silently rendering an empty panel):

* ``clockdata`` local SQLite database: macro panel, ETF price panel, FF5
* ``factorlab`` cross-sectional report: factor IC / ICIR / hit rate
* ``quantclock``: phase probabilities, asset weights, phase history
* ``quantclock.company_analysis``: per-instrument public-data snapshot
* ``clockdata.registry``: live source health
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cloud as cloud_mod
from . import forecast as fc_mod
from . import graph as graph_mod


def build_payload(
    symbol: str,
    *,
    db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    include_instrument: bool = True,
    include_sources: bool = True,
    instrument: dict[str, Any] | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    """Build the full constellation payload for ``symbol``.

    Returns a JSON-serialisable dict. Sections that could not be built carry an
    ``available: False`` plus a ``note`` explaining why, so the renderer can
    state the limitation instead of showing a decorative blank.

    ``offline=True`` skips every network call (no live price fetch), which is
    what the HTTP service uses to stay responsive.
    """
    from clockdata import registry as reg
    from clockdata import universe as uni

    inst = instrument or (uni.classify(symbol) if offline else uni.resolve(symbol))
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    payload: dict[str, Any] = {
        "meta": {
            "title": f"美林时钟星图 · {inst.symbol}",
            "subtitle": (f"{inst.label} · 市场 {inst.market} · 数据源 "
                         f"{inst.provider_hint or '多源路由'}"),
            "generated_at": generated,
            "symbol": inst.symbol,
            "market": inst.market,
        },
        "badges": [],
        "graph": {"nodes": [], "links": []},
        "phases": [],
        "factors": [],
        "instrument": {"symbol": inst.symbol, "label": inst.label, "metrics": []},
        "sources": [],
        "clouds": {},
        "forecasts": {},
        "caveats": [],
        "legend": [
            {"label": "相位", "color": "#5ad7ff"},
            {"label": "资产", "color": "#9d7bff"},
            {"label": "因子", "color": "#ff6bd6"},
            {"label": "标的", "color": "#ffd166"},
            {"label": "数据源", "color": "#5b8c5a"},
            {"label": "预测", "color": "#ff8a5c"},
        ],
    }

    # ------------------------------------------------------------------ #
    # 1. Phase probabilities + asset weights + phase history
    # ------------------------------------------------------------------ #
    phase_info = _load_phase_info(output_dir)
    if phase_info:
        payload["phases"] = phase_info["phases"]
        payload["allocations"] = phase_info.get("allocations_list", [])
        payload["graph"]["nodes"].extend(graph_mod.build_phase_nodes(
            phase_info["probabilities"], phase_info.get("phase_cn")))
        payload["graph"]["links"].extend(graph_mod.build_phase_links(
            phase_info["allocations"], phase_info["current_phase"]))
        transition = phase_info.get("transition_matrix") or {}
        if transition:
            payload["graph"]["links"].extend(
                graph_mod.build_transition_links(transition))
        payload["forecasts"]["phase_path"] = fc_mod.project_phase_path(
            phase_info["current_phase"], transition, horizon=6)
        payload["clouds"]["regime"] = phase_info.get("regime_cloud") or {}
    else:
        payload["caveats"].append(
            "未找到相位概率输出（缺少 metrics.json 或 regime_probabilities），相位层为空。")

    # ------------------------------------------------------------------ #
    # 2. Factor effectiveness from the cross-sectional report
    # ------------------------------------------------------------------ #
    factor_stats = _load_factor_stats(output_dir)
    if factor_stats:
        payload["factors"] = factor_stats["table"]
        payload["graph"]["nodes"].extend(
            graph_mod.build_factor_nodes(factor_stats["raw"]))
        payload["forecasts"]["factor_paths"] = fc_mod.project_factor_expectations(
            factor_stats["raw"], horizon=6)

    # ------------------------------------------------------------------ #
    # 3. The instrument under study
    # ------------------------------------------------------------------ #
    if include_instrument:
        snap = _load_instrument_snapshot(inst.symbol, output_dir, offline=offline)
        if snap:
            node = graph_mod.build_instrument_node(
                inst.symbol, snap.get("name") or inst.label, snap)
            payload["graph"]["nodes"].append(node)
            payload["instrument"] = {
                "symbol": inst.symbol,
                "label": snap.get("name") or inst.label,
                "metrics": snap.get("metric_rows", []),
            }
            if snap.get("factor_values"):
                payload["graph"]["links"].extend(
                    graph_mod.build_factor_links(snap["factor_values"], inst.symbol))
            payload["forecasts"]["price_path"] = fc_mod.project_price_paths(
                snap["monthly_returns"], horizon=12)
            payload["clouds"]["correlation"] = snap.get("cloud") or {}
        else:
            payload["instrument"]["note"] = (
                "该标的无可用快照；可运行 merrill-clock-quant company --code "
                f"{inst.symbol} 生成。")

    # ------------------------------------------------------------------ #
    # 4. Source health (the backend's own diagnostics)
    # ------------------------------------------------------------------ #
    used_sources = {l.kind for l in payload["graph"]["links"]
                    if getattr(l, "kind", "") == "data_lineage"}
    used_source_ids = {l.source.split(":", 1)[-1]
                       for l in payload["graph"]["links"]
                       if getattr(l, "kind", "") == "data_lineage"}
    if include_sources:
        rows = reg.registry_rows()
        payload["sources"] = [
            {
                "source_id": r["source_id"],
                "capabilities": " / ".join(r["capabilities"][:4]),
                "available": r["available"],
                "status_detail": r["status_detail"],
                "note": r["note"],
                "tier": r["tier"],
            }
            for r in rows
        ]
        payload["graph"]["nodes"].extend(
            graph_mod.build_source_nodes(rows, used=used_source_ids))
        for sid in sorted(used_source_ids):
            payload["graph"]["links"].extend(graph_mod.build_lineage_links(
                [sid], [n.id for n in payload["graph"]["nodes"]
                        if n.kind in ("macro_phase", "instrument", "factor")][:3]))

    # ------------------------------------------------------------------ #
    # 5. Caveats and badges
    # ------------------------------------------------------------------ #
    payload["caveats"].extend(_default_caveats(inst))
    payload["badges"] = _badges(inst, payload)

    # Structural self-check: never ship a broken graph.
    issues = graph_mod.Constellation(
        nodes=list(payload["graph"]["nodes"]),
        links=list(payload["graph"]["links"]),
    ).validate()
    if issues:
        payload["caveats"].append("图结构自检发现问题：" + "；".join(issues[:5]))

    # The graph is held as Node/Link objects during assembly; convert to plain
    # dicts so the payload is JSON-serialisable without lossy fallbacks.
    payload["graph"] = {
        "nodes": [n.to_dict() for n in payload["graph"]["nodes"]],
        "links": [l.to_dict() for l in payload["graph"]["links"]],
    }

    return payload


# --------------------------------------------------------------------------- #
# Loaders -- each returns None when the data is unavailable
# --------------------------------------------------------------------------- #
def _node_fields(node: dict[str, Any]) -> dict[str, Any]:
    return {k: node.get(k) for k in
            ("id", "kind", "label", "value", "detail", "metrics", "x", "y", "z")}


def _output_dir(output_dir: str | Path | None) -> Path | None:
    """Locate a directory holding research results (metrics / backtest CSVs).

    Search order: explicit argument, then the monorepo ``outputs`` tree, then
    the shared data directory. The first directory that actually contains
    ``clock_backtest.csv`` wins, so the loader never silently reads an empty
    folder.
    """
    if output_dir:
        candidate = Path(output_dir)
        if (candidate / "clock_backtest.csv").exists():
            return candidate
        # An explicit path may point at a parent holding several run folders.
        for sub in sorted(candidate.glob("*/clock_backtest.csv")):
            return sub.parent
        return candidate

    roots = [
        Path.home() / ".local" / "share" / "clockdata",
        Path.cwd() / "outputs",
        Path.cwd(),
    ]
    for root in roots:
        if not root.exists():
            continue
        if (root / "clock_backtest.csv").exists():
            return root
        for sub in sorted(root.glob("*/clock_backtest.csv")):
            return sub.parent
    return None


def _load_phase_info(output_dir: str | Path | None) -> dict[str, Any] | None:
    """Phase probabilities, allocations and transition matrix.

    The authoritative source is ``clock_backtest.csv``: it carries the monthly
    ``p_<phase>`` probabilities, the ``w_<asset>`` target weights, the
    growth/inflation features and the evaluated flag. ``metrics.json`` only
    carries scalars, so it cannot supply the probability path on its own.
    """
    base = _output_dir(output_dir)
    if base is None:
        return None
    backtest_path = base / "clock_backtest.csv"
    if not backtest_path.exists():
        return None
    try:
        frame = pd.read_csv(backtest_path, parse_dates=["date"]).set_index("date")
    except Exception:  # noqa: BLE001
        return None

    prob_cols = [c for c in frame.columns if c.startswith("p_")]
    weight_cols = [c for c in frame.columns if c.startswith("w_")]
    if not prob_cols:
        return None

    evaluated = frame["evaluated"] if "evaluated" in frame.columns else pd.Series(
        1, index=frame.index)
    usable = frame[evaluated.astype(str).isin(("1", "1.0", "True", "true"))]
    if usable.empty:
        usable = frame

    latest = usable.iloc[-1]
    probabilities = {c[2:]: float(latest[c]) for c in prob_cols
                     if pd.notna(latest[c])}
    if not probabilities:
        return None
    current = max(probabilities, key=probabilities.get)

    allocations = {}
    for col in weight_cols:
        value = latest[col]
        if pd.notna(value):
            allocations[col[2:]] = float(value)

    # Phase history for the transition matrix and the regime density cloud.
    history = usable.dropna(subset=prob_cols)
    phase_series = history[prob_cols].idxmax(axis=1).str.replace("p_", "", regex=False)
    transition = cloud_mod.phase_transition_matrix(phase_series)

    regime_cloud: dict[str, Any] = {}
    if {"growth_accel", "inflation_accel"}.issubset(history.columns):
        regime_cloud = cloud_mod.regime_density_cloud(
            history[["growth_accel", "inflation_accel"]],
            phase_col=None)

    phase_cn = {"reflation": "再通胀", "recovery": "复苏", "overheat": "过热",
                "stagflation": "滞胀"}

    # Probabilities are keyed by phase, weights by asset. The panel shows both
    # as separate lists: a phase probability is not an allocation weight, and
    # conflating them would misrepresent the model.
    return {
        "probabilities": probabilities,
        "current_phase": current,
        "allocations": allocations,
        "phase_cn": phase_cn,
        "transition_matrix": transition,
        "regime_cloud": regime_cloud,
        "history": history,
        "phases": [
            {"label": phase_cn.get(p, p), "probability": float(v), "weight": 0.0}
            for p, v in sorted(probabilities.items(), key=lambda kv: -float(kv[1]))
        ],
        "allocations_list": [
            {"label": asset, "weight": float(w)}
            for asset, w in sorted(allocations.items(), key=lambda kv: -float(kv[1]))
        ],
    }


def _load_factor_stats(output_dir: str | Path | None) -> dict[str, Any] | None:
    """Cross-sectional factor IC / ICIR / hit rate."""
    base = _output_dir(output_dir)
    if base is None:
        return None
    path = base / "cross_section_report.json"
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    factors = report.get("factors") or {}
    if not factors:
        return None
    raw: dict[str, dict[str, float]] = {}
    table = []
    for name, rep in factors.items():
        s = rep.get("ic_summary") or {}
        raw[name] = {
            "ic_mean": float(s.get("ic_mean", 0.0) or 0.0),
            "icir": float(s.get("icir", 0.0) or 0.0),
            "positive_ratio": float(s.get("positive_ratio", 0.0) or 0.0),
            "n_periods": int(s.get("n_periods", 0) or 0),
        }
        table.append({
            "name": name,
            "ic_mean": raw[name]["ic_mean"],
            "icir": raw[name]["icir"],
            "positive_ratio": raw[name]["positive_ratio"],
            "n_periods": raw[name]["n_periods"],
        })
    table.sort(key=lambda r: abs(r["ic_mean"]), reverse=True)
    return {"raw": raw, "table": table}


def _load_instrument_snapshot(symbol: str,
                              output_dir: str | Path | None,
                              offline: bool = False) -> dict[str, Any] | None:
    """Per-instrument snapshot + factor exposures + price history.

    Reads the artefacts produced by ``merrill-clock-quant company``:
    ``metrics.json`` (snapshot payload) and ``daily_ohlcv_qfq.csv`` (prices).

    When no company snapshot exists, fall back to fetching the price panel
    directly from the multi-source router -- that path works for any market
    (A-share, US, crypto), which is what keeps the constellation reusable
    beyond a single ticker. In ``offline`` mode only the local database is
    consulted, so the HTTP service never blocks on an upstream.
    """
    base = _output_dir(output_dir)
    candidates = []
    if base:
        candidates.append(base / f"company_{symbol}")
        candidates.append(base / "company")
    for cand in candidates:
        metrics_path = cand / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            snap = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if str(snap.get("metadata", {}).get("code", "")) != str(symbol):
            continue
        return _snapshot_to_payload(symbol, snap, cand)

    if offline:
        return _local_db_snapshot(symbol)
    return _fetch_instrument_from_sources(symbol, base)


def _local_db_snapshot(symbol: str) -> dict[str, Any] | None:
    """Offline variant: read the price panel straight from the local database."""
    from clockdata.prices import _local_db_prices

    try:
        frame = _local_db_prices([symbol], start="2015-01-01", end=None)
    except Exception:  # noqa: BLE001 - symbol not stored locally
        return None
    if frame.empty or symbol not in frame.columns:
        return None
    series = frame[symbol].dropna()
    if series.empty:
        return None
    monthly = series.resample("ME").last().pct_change().dropna()
    rows = [
        ["最新收盘", f"{float(series.iloc[-1]):.3f}"],
        ["数据源", "本地金融数据库（离线）"],
        ["样本起止", f"{str(series.index[0])[:10]} → {str(series.index[-1])[:10]}"],
        ["观测数", str(int(len(series)))],
    ]
    cloud_payload = {}
    if len(monthly) >= 10:
        combined = pd.DataFrame({
            "动量": monthly.rolling(6).mean(),
            "波动": monthly.rolling(6).std(),
        }).dropna()
        if len(combined) >= 8:
            cloud_payload = cloud_mod.correlation_cloud(
                combined, "动量", "波动", width=52, height=32)
    return {
        "name": "",
        "metric_rows": rows,
        "factor_values": _derive_exposures(monthly.to_frame("close")),
        "monthly_returns": monthly,
        "cloud": cloud_payload,
    }


def _fetch_instrument_from_sources(symbol: str,
                                   base: Path | None) -> dict[str, Any] | None:
    """Fetch a price panel for ``symbol`` through the multi-source router."""
    from clockdata import universe as uni
    from clockdata.prices import fetch_prices

    inst = uni.classify(symbol)
    if inst.market == uni.MARKET_UNKNOWN:
        return None
    try:
        outcome = fetch_prices([symbol], start="2018-01-01")
    except Exception:  # noqa: BLE001 - upstream unavailable
        return None

    frame = outcome.frame
    if frame.empty:
        return None
    series = frame.iloc[:, 0].dropna()
    if series.empty:
        return None

    monthly = series.resample("ME").last().pct_change().dropna()
    last = float(series.iloc[-1])
    returns_60 = float(series.pct_change(60).dropna().iloc[-1]) if len(series) > 60 else None
    vol_60 = float(series.pct_change().tail(60).std() * np.sqrt(252)) if len(series) > 60 else None
    max_dd = float((series / series.cummax() - 1).tail(60).min()) if len(series) > 20 else None

    rows = [
        ["最新收盘", f"{last:.3f}"],
        ["数据源", outcome.source_id or "unknown"],
        ["降级取数", "是" if outcome.degraded else "否"],
        ["样本起止", f"{str(series.index[0])[:10]} → {str(series.index[-1])[:10]}"],
        ["观测数", str(int(len(series)))],
    ]
    if returns_60 is not None:
        rows.append(["60日收益", f"{returns_60:.3f}"])
    if vol_60 is not None:
        rows.append(["年化波动(60日)", f"{vol_60:.3f}"])
    if max_dd is not None:
        rows.append(["最大回撤(60日)", f"{max_dd:.3f}"])

    cloud_payload = {}
    if len(monthly) >= 10:
        combined = pd.DataFrame({
            "动量": monthly.rolling(6).mean(),
            "波动": monthly.rolling(6).std(),
        }).dropna()
        if len(combined) >= 8:
            cloud_payload = cloud_mod.correlation_cloud(
                combined, "动量", "波动", width=52, height=32)

    return {
        "name": inst.label or "",
        "metric_rows": rows,
        "factor_values": _derive_exposures(monthly.to_frame("close")),
        "monthly_returns": monthly,
        "cloud": cloud_payload,
    }


def _snapshot_to_payload(symbol: str, snap: dict[str, Any],
                         folder: Path) -> dict[str, Any]:
    """Flatten a company snapshot into display rows + price-derived features.

    Field names follow the real ``metrics.json`` written by
    ``quantclock.company_analysis.build_company_snapshot``: ``technical``,
    ``flow``, ``announcements``, ``sentiment``, ``market_model``.
    """
    technical = snap.get("technical") or {}
    flow = snap.get("flow") or {}
    sentiment = snap.get("sentiment") or {}
    market_model = snap.get("market_model") or {}
    announcements = snap.get("announcements") or {}

    rows: list[list[str]] = []

    def _add(label: str, value: Any, digits: int = 3) -> None:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append([label, f"{float(value):.{digits}f}"])
        else:
            rows.append([label, str(value)])

    _add("最新收盘", technical.get("close_qfq"), 2)
    _add("5日收益", technical.get("return_5d"))
    _add("20日收益", technical.get("return_20d"))
    _add("60日收益", technical.get("return_60d"))
    _add("120日收益", technical.get("return_120d"))
    _add("年化波动(20日)", technical.get("annualized_volatility_20d"))
    _add("最大回撤(60日)", technical.get("max_drawdown_60d"))
    _add("RSI14", technical.get("rsi14"), 1)
    _add("Beta", market_model.get("beta"))
    _add("R²", market_model.get("r_squared"))
    _add("相对基准20日", sentiment.get("relative_return_20d_vs_benchmark"))
    _add("主力净流入(5日,亿)",
         (flow.get("main_net_5d_cny") or 0) / 1e8 if flow.get("available") else None, 2)
    _add("主力净流入(20日,亿)",
         (flow.get("main_net_20d_cny") or 0) / 1e8 if flow.get("available") else None, 2)
    _add("公告条数", announcements.get("count"), 0)
    _add("最新公告", (announcements.get("latest_title") or "")[:26])

    # Price history -> monthly returns for the bootstrap fan + density cloud.
    monthly_returns = pd.Series(dtype=float)
    factor_values: dict[str, float] = {}
    prices_path = folder / "daily_ohlcv_qfq.csv"
    if not prices_path.exists():
        prices_path = folder / "prices.csv"
    if prices_path.exists():
        try:
            prices = pd.read_csv(prices_path, parse_dates=[0], index_col=0)
            numeric = prices.select_dtypes("number")
            if "close" in numeric.columns and not numeric.empty:
                monthly = numeric["close"].resample("ME").last().pct_change().dropna()
                monthly_returns = monthly
                factor_values = _derive_exposures(monthly.to_frame("close"))
        except Exception:  # noqa: BLE001
            pass

    cloud_payload = {}
    if len(monthly_returns) >= 10:
        combined = pd.DataFrame({
            "动量": monthly_returns.rolling(6).mean(),
            "波动": monthly_returns.rolling(6).std(),
        }).dropna()
        if len(combined) >= 8:
            cloud_payload = cloud_mod.correlation_cloud(
                combined, "动量", "波动", width=52, height=32)

    return {
        "name": snap.get("metadata", {}).get("name", ""),
        "metric_rows": rows,
        "factor_values": factor_values,
        "monthly_returns": monthly_returns,
        "cloud": cloud_payload,
    }


def _derive_exposures(monthly_frame: pd.DataFrame) -> dict[str, float]:
    """Derive simple factor exposures from a monthly return series."""
    out: dict[str, float] = {}
    try:
        if monthly_frame.empty:
            return out
        series = monthly_frame.iloc[:, 0].dropna()
        if len(series) < 6:
            return out
        if len(series) >= 6:
            out["mom_6m"] = float(series.tail(6).sum())
        if len(series) >= 12:
            out["mom_12m"] = float(series.tail(12).sum())
        out["vol_6m"] = float(series.tail(6).std() * np.sqrt(12))
    except Exception:  # noqa: BLE001
        return {}
    return out


def _default_caveats(inst: Any) -> list[str]:
    return [
        "本图为研究工具的可视化输出，所有数值来自本地数据库与实测计算，不含任何前瞻信息。",
        "相位概率来自 FRED 宏观序列的最新修订值（INDPRO / CPIAUCSL），非点时(vintage)数据；"
        "若做严肃回测需改用 ALFRED vintage。",
        "相位路径为历史转移频率的外推，是概率分布而非预测；熵随步数上升代表不确定性累积。",
        "路径扇形图为历史月度收益的自助抽样分位带，假设收益分布在未来不变，"
        "分位带是分布区间而非置信区间。",
        "因子 IC 为横截面统计口径，与单标的暴露不完全等同；IC 与 t 值未做序列相关校正。",
        "本图不构成投资建议。任何预测性输出都不代表未来必然发生。",
    ]


def _badges(inst: Any, payload: dict[str, Any]) -> list[tuple[str, str]]:
    badges: list[tuple[str, str]] = []
    badges.append((f"市场 {inst.market}", "on"))
    if payload["phases"]:
        top = payload["phases"][0]
        badges.append((f"当前相位 {top['label']} {top['probability']:.0%}", "on"))
    if payload["factors"]:
        badges.append((f"因子 {len(payload['factors'])} 个", "on"))
    available = sum(1 for s in payload["sources"] if s.get("available"))
    badges.append((f"数据源 {available}/{len(payload['sources'])} 可用",
                   "on" if available else "off"))
    if payload["clouds"].get("correlation", {}).get("available"):
        badges.append(("密度云图", "on"))
    if payload["forecasts"].get("price_path", {}).get("available"):
        badges.append(("路径扇形图", "on"))
    return badges
