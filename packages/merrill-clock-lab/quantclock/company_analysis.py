"""Auditable five-dimension A-share research snapshot calculations.

All outputs are descriptive. No composite recommendation score is produced.
"""
from __future__ import annotations

from datetime import time as dt_time
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .company_data import fetch_daily_ohlcv, fetch_money_flow, fetch_benchmark_daily, fetch_announcements

REGIME_CN = {
    "recovery": "复苏", "overheat": "过热", "stagflation": "滞胀", "reflation": "再通胀（增长与通胀动能均弱）",
}
DEFAULT_THEME_KEYWORDS = [
    "电子信息", "通信", "电子元器件", "元器件", "射频", "天线", "连接器",
    "滤波器", "智能终端", "消费电子", "卫星通信", "新型电子元器件",
]


def _last_valid(series: pd.Series) -> float | None:
    value = series.dropna().iloc[-1] if not series.dropna().empty else np.nan
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def split_incomplete_daily_bar(
    daily: pd.DataFrame,
    retrieved_at: str | None = None,
    close_time: dt_time = dt_time(15, 0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate today's still-open daily bar from fully closed sessions.

    The timestamp is interpreted in Asia/Shanghai. A same-date bar fetched
    before the exchange's 15:00 close is provisional and must not enter daily
    returns, indicators, sentiment statistics, or factor regressions.
    """
    if daily is None or daily.empty:
        empty = pd.DataFrame() if daily is None else daily.iloc[0:0].copy()
        return empty, empty.copy()
    try:
        captured = pd.Timestamp(retrieved_at) if retrieved_at else pd.Timestamp.now(tz="Asia/Shanghai")
    except (TypeError, ValueError):
        captured = pd.Timestamp.now(tz="Asia/Shanghai")
    if captured.tzinfo is None:
        captured = captured.tz_localize("Asia/Shanghai")
    else:
        captured = captured.tz_convert("Asia/Shanghai")
    session_date = captured.date()
    latest_date = pd.Timestamp(daily.index[-1]).date()
    if latest_date != session_date or captured.time().replace(tzinfo=None) >= close_time:
        complete = daily.copy()
        complete.attrs.update(daily.attrs)
        return complete, daily.iloc[0:0].copy()

    same_session = pd.Index(daily.index.date) == session_date
    provisional = daily.loc[same_session].tail(1).copy()
    complete = daily.loc[~same_session].copy()
    complete.attrs.update(daily.attrs)
    provisional.attrs.update(daily.attrs)
    return complete, provisional


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative = gain / loss.replace(0, np.nan)
    values = 100 - (100 / (1 + relative))
    values = values.mask((loss == 0) & (gain > 0), 100.0)
    values = values.mask((loss == 0) & (gain == 0), 50.0)
    return values


def technical_analysis(daily: pd.DataFrame) -> dict:
    close = pd.to_numeric(daily["close"], errors="coerce").dropna().sort_index()
    if len(close) < 20:
        raise ValueError("技术面至少需要20个有效交易日")
    returns = close.pct_change(fill_method=None)
    ma20, ma60 = close.rolling(20, min_periods=20).mean(), close.rolling(60, min_periods=60).mean()
    latest = close.index[-1]
    ret = {n: _last_valid(close.pct_change(n, fill_method=None)) for n in (5, 20, 60, 120)}
    rsi14 = _last_valid(_rsi(close, 14))
    ma20v, ma60v = _last_valid(ma20), _last_valid(ma60)
    latest_close = float(close.iloc[-1])
    gap20 = latest_close / ma20v - 1 if ma20v else None
    gap60 = latest_close / ma60v - 1 if ma60v else None
    if ma60v is None:
        trend = "可用历史不足60日，趋势结构暂不判读"
    elif latest_close > ma20v and ma20v > ma60v:
        trend = "价格高于MA20且MA20高于MA60（偏强趋势结构描述）"
    elif latest_close < ma20v and ma20v < ma60v:
        trend = "价格低于MA20且MA20低于MA60（偏弱趋势结构描述）"
    else:
        trend = "均线结构混合/震荡"
    recent = close.tail(60)
    drawdown60 = float((recent / recent.cummax() - 1).min()) if len(recent) else None
    vol20 = _last_valid(returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252))
    rsi_state = "数据不足" if rsi14 is None else ("高位区间代理（>70）" if rsi14 >= 70 else "低位区间代理（<30）" if rsi14 <= 30 else "中性区间代理")
    return {
        "as_of": latest.strftime("%Y-%m-%d"), "close_qfq": latest_close,
        "return_5d": ret[5], "return_20d": ret[20], "return_60d": ret[60], "return_120d": ret[120],
        "ma20": ma20v, "ma60": ma60v, "close_vs_ma20": gap20, "close_vs_ma60": gap60,
        "rsi14": rsi14, "rsi_state": rsi_state, "annualized_volatility_20d": vol20,
        "max_drawdown_60d": drawdown60, "trend_structure": trend,
        "window_note": "收益、均线、RSI与波动均按前复权日线计算；均为历史描述，不构成买卖条件。",
    }


def flow_analysis(flow: pd.DataFrame) -> dict:
    if flow is None or flow.empty:
        return {"available": False, "note": "数据源未返回资金流记录；不以0代替缺失。"}
    f = flow.sort_index()
    amount = f.get("main_net_inflow_cny", pd.Series(dtype=float)).dropna()
    ratio = f.get("main_net_inflow_ratio_pct", pd.Series(dtype=float)).dropna()
    n5, n20 = amount.tail(5), amount.tail(20)
    r20 = ratio.tail(20)
    net5 = float(n5.sum()) if len(amount) >= 5 and len(n5) == 5 else None
    net20 = float(n20.sum()) if len(amount) >= 20 and len(n20) == 20 else None
    return {
        "available": bool(len(amount)), "last_date": f.index[-1].strftime("%Y-%m-%d"),
        "observations": int(len(f)), "main_net_5d_cny": net5, "main_net_20d_cny": net20,
        "positive_days_5d": int((n5 > 0).sum()) if len(amount) >= 5 else None,
        "positive_days_20d": int((n20 > 0).sum()) if len(amount) >= 20 else None,
        "main_net_ratio_mean_20d_pct": float(r20.mean()) if len(ratio) >= 20 and len(r20) == 20 else None,
        "note": "采用数据商订单规模分类资金流估算，不等于交易所核验资金或机构身份。仅当窗口有完整5/20个观测时才计算对应累计值；不足窗口显示缺失，不以较短样本冒充20日数据。",
    }


def market_sentiment_analysis(daily: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> dict:
    d = daily.sort_index().copy()
    close = pd.to_numeric(d["close"], errors="coerce")
    ret = close.pct_change(fill_method=None)
    r5 = _last_valid(close.pct_change(5, fill_method=None))
    amount = pd.to_numeric(d.get("amount_cny", np.nan), errors="coerce")
    amount_ratio = None
    if isinstance(amount, pd.Series) and amount.notna().sum() >= 25:
        avg5 = amount.tail(5).mean()
        baseline = amount.iloc[-25:-5].median()
        amount_ratio = float(avg5 / baseline) if baseline and pd.notna(baseline) else None
    turnover = pd.to_numeric(d.get("turnover_pct", np.nan), errors="coerce")
    turnover_ratio = None
    if isinstance(turnover, pd.Series) and turnover.notna().sum() >= 25:
        base = turnover.iloc[-25:-5].median()
        turnover_ratio = float(turnover.tail(5).mean() / base) if pd.notna(base) and base else None
    close_pos = None
    if all(c in d for c in ("high", "low")):
        hi, lo = float(d["high"].iloc[-1]), float(d["low"].iloc[-1])
        if hi > lo:
            close_pos = float((d["close"].iloc[-1] - lo) / (hi - lo))
    rel20 = None
    if benchmark is not None and not benchmark.empty:
        bclose = pd.to_numeric(benchmark["close"], errors="coerce")
        common = close.index.intersection(bclose.index)
        if len(common) > 20:
            sr = close.loc[common].pct_change(20, fill_method=None).iloc[-1]
            br = bclose.loc[common].pct_change(20, fill_method=None).iloc[-1]
            rel20 = float(sr - br) if pd.notna(sr) and pd.notna(br) else None
    if r5 is None:
        label = "样本不足"
    elif r5 > 0 and (amount_ratio is None or amount_ratio >= 1.2):
        label = "近期价格上行，成交额同步偏活跃（交易情绪代理）"
    elif r5 < 0 and (amount_ratio is not None and amount_ratio >= 1.2):
        label = "近期下行伴随成交额放大（交易分歧/压力代理）"
    else:
        label = "量价信号混合或成交活跃度未显著变化"
    return {
        "classification": label, "return_5d": r5, "amount_5d_vs_prior20d_median": amount_ratio,
        "turnover_5d_vs_prior20d_median": turnover_ratio, "last_day_close_location": close_pos,
        "relative_return_20d_vs_benchmark": rel20,
        "scope_note": "这里的情绪仅是价格、成交额/换手率及相对强弱等交易行为代理；没有抓取社交平台发言，也不代表投资者总体情绪。",
    }


def classify_announcement(title: str, category: str = "") -> str:
    text = f"{title} {category}"
    rules = [
        ("定期报告/业绩", ("年度报告", "半年度报告", "季度报告", "业绩预告", "业绩快报", "业绩说明")),
        ("投资者沟通", ("投资者关系", "调研", "业绩说明会", "路演")),
        ("资本与股权事项", ("回购", "增持", "减持", "股权激励", "定向增发", "股权收购", "再融资", "注册批复", "分红", "关联交易")),
        ("经营与合同事项", ("合同", "订单", "中标", "项目", "合作", "投资", "收购", "增资")),
        ("风险/监管事项", ("风险提示", "问询", "监管", "诉讼", "仲裁", "处罚", "更正", "澄清")),
        ("治理与会议", ("董事会", "监事会", "股东大会", "股东会", "决议公告", "换届", "辞职", "任职变动")),
    ]
    for label, keywords in rules:
        if any(k in text for k in keywords):
            return label
    return "其他披露"


def summarize_announcements(announcements: pd.DataFrame) -> dict:
    if announcements is None or announcements.empty:
        return {"count": 0, "latest_title": None, "latest_published_at": None, "note": "未获取到公告列表，不表示没有公告。"}
    frame = announcements.copy()
    frame["event_type"] = frame.apply(lambda row: classify_announcement(str(row.get("title", "")), str(row.get("category", ""))), axis=1)
    latest = frame.iloc[0]
    return {
        "count": int(len(frame)), "latest_title": str(latest.get("title", "")),
        "latest_published_at": latest.get("published_at").isoformat() if pd.notna(latest.get("published_at")) else None,
        "categories": {str(k): int(v) for k, v in frame["event_type"].value_counts().items()},
        "note": "按标题/栏目进行规则分类，仅作检索导航；不判断利好利空，也不替代公告正文。",
    }


def match_policy_documents(policy_frame: pd.DataFrame, theme_keywords: list[str] | None = None) -> pd.DataFrame:
    if policy_frame is None or policy_frame.empty:
        return pd.DataFrame(columns=["title", "published_at", "publisher", "relevant_terms", "scope_note", "source_url"])
    themes = theme_keywords or DEFAULT_THEME_KEYWORDS
    rows = []
    for _, row in policy_frame.iterrows():
        tags = [x.strip() for x in str(row.get("keywords", "")).split(";") if x.strip()]
        text = " ".join(str(row.get(k, "")) for k in ("title", "summary", "keywords"))
        hits = sorted({key for key in themes if key in text} | {tag for tag in tags if tag in text or tag in themes})
        if hits:
            out = row.to_dict()
            out["relevant_terms"] = "、".join(hits)
            rows.append(out)
    return pd.DataFrame(rows)


def load_policy_watchlist(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"title", "published_at", "publisher", "keywords", "summary", "source_url", "scope_note"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"政策清单缺少字段：{', '.join(sorted(missing))}")
    return frame


def market_model(stock_daily: pd.DataFrame, benchmark_daily: pd.DataFrame | None, max_observations: int = 120) -> dict:
    if benchmark_daily is None or benchmark_daily.empty:
        return {"available": False, "note": "基准指数不可用，暂不计算市场模型。"}
    s = pd.to_numeric(stock_daily["close"], errors="coerce").pct_change(fill_method=None).rename("stock")
    b = pd.to_numeric(benchmark_daily["close"], errors="coerce").pct_change(fill_method=None).rename("benchmark")
    joined = pd.concat([s, b], axis=1).dropna().tail(max_observations)
    if len(joined) < 30:
        return {"available": False, "observations": int(len(joined)), "note": "共同有效日少于30，不报告回归估计。"}
    try:
        from .factors import factor_regression
        fit = factor_regression(joined["stock"], joined[["benchmark"]], maxlags=5)
        return {
            "available": True, "observations": int(fit.nobs), "benchmark_factor": str(benchmark_daily.attrs.get("name", benchmark_daily.attrs.get("code", "index"))),
            "beta": float(fit.params["benchmark"]), "daily_intercept": float(fit.params["const"]),
            "beta_hac_t": float(fit.tvalues["benchmark"]), "intercept_hac_t": float(fit.tvalues["const"]),
            "r_squared": float(fit.rsquared), "correlation": float(joined.corr().iloc[0, 1]),
            "note": "单指数日收益回归；HAC标准误(Newey-West，5阶)；指数为价格指数代理，样本估计不是未来alpha或预测。",
        }
    except Exception as exc:
        return {"available": False, "observations": int(len(joined)), "note": f"市场模型不可用：{type(exc).__name__}: {exc}"}


def fetch_us_merrill_clock_context(start: str = "1990-01-01") -> dict:
    """Use the original project's FRED/US macro setup as a separately labeled context."""
    from .data import fetch_fred_panel
    from .macro import make_clock_features
    from .signals import regime_probabilities
    macro = fetch_fred_panel(start=start)
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    decisions = pd.date_range(start=pd.Timestamp(start).to_period("M").to_timestamp("M"), end=now, freq="ME")
    if len(decisions) == 0:
        raise RuntimeError("没有可用月末日期")
    features = make_clock_features(macro["industrial_production"], macro["cpi"], decisions, ip_release_lag_days=20, cpi_release_lag_days=20)
    probabilities = regime_probabilities(features, window=60, smooth=3)
    valid = probabilities.dropna(how="any")
    if valid.empty:
        raise RuntimeError("宏观动能历史不足以生成时钟概率")
    latest = valid.iloc[-1]
    phase = str(latest.idxmax())
    return {
        "available": True, "region": "美国宏观代理（FRED INDPRO + CPIAUCSL）",
        "as_of": valid.index[-1].strftime("%Y-%m-%d"), "phase": phase, "phase_cn": REGIME_CN[phase],
        "probabilities": {str(k): float(v) for k, v in latest.items()},
        "growth_accel_proxy": float(features.loc[valid.index[-1], "growth_accel"]),
        "inflation_accel_proxy": float(features.loc[valid.index[-1], "inflation_accel"]),
        "source_url": "https://fred.stlouisfed.org/",
        "note": "沿用项目既有美国工业生产与CPI动能软分类；FRED当前修订值、假设20日发布滞后；不是中国宏观周期、不是已校准概率，也不直接映射为个股结论。",
    }


def build_company_snapshot(code: str, start_date: str = "2023-01-01", benchmark_code: str = "399006", flow_limit: int = 100, announcement_limit: int = 20, policy_file: str | Path | None = None, include_us_clock: bool = True, flow_csv: str | Path | None = None) -> dict:
    """Fetch and calculate the end-to-end public-data company snapshot."""
    daily_raw = fetch_daily_ohlcv(code, start_date=start_date)
    captured_at = daily_raw.attrs.get("retrieved_at")
    daily, intraday_frame = split_incomplete_daily_bar(daily_raw, captured_at)
    if daily.empty:
        raise RuntimeError("没有可用于日频收盘分析的完整K线")
    source_errors = {}
    flow_source = "Eastmoney public fund-flow endpoint"
    if flow_csv:
        from .company_data import read_money_flow_csv
        flow = read_money_flow_csv(str(flow_csv), code=code)
        flow_source = f"本地公开数据CSV（{Path(flow_csv).name}）"
    else:
        try:
            flow = fetch_money_flow(code, limit=flow_limit)
        except Exception as exc:
            flow = pd.DataFrame()
            source_errors["fund_flow"] = f"{type(exc).__name__}: {exc}"
    try:
        announcements = fetch_announcements(code, limit=announcement_limit)
    except Exception as exc:
        announcements = pd.DataFrame(columns=["published_at", "notice_date", "title", "category", "article_code", "source_url"])
        source_errors["announcements"] = f"{type(exc).__name__}: {exc}"
    try:
        benchmark_raw = fetch_benchmark_daily(benchmark_code, start_date=start_date)
    except Exception as exc:
        benchmark_raw = pd.DataFrame(columns=["open", "close", "high", "low", "volume_raw", "amount_cny", "amplitude_pct", "change_pct", "change_cny", "turnover_pct"])
        source_errors["benchmark"] = f"{type(exc).__name__}: {exc}"
    if not benchmark_raw.empty and not intraday_frame.empty:
        intraday_date = pd.Timestamp(intraday_frame.index[-1]).date()
        benchmark = benchmark_raw.loc[pd.Index(benchmark_raw.index.date) < intraday_date].copy()
        benchmark.attrs.update(benchmark_raw.attrs)
    else:
        benchmark = benchmark_raw.copy()
        benchmark.attrs.update(benchmark_raw.attrs)
    if policy_file is None:
        policy_file = Path(__file__).resolve().parent.parent / "data" / "policy_watchlist.csv"
    policies = load_policy_watchlist(policy_file)
    matched_policies = match_policy_documents(policies)

    intraday_reference = {"available": False, "note": "抓取时未识别到当日未完成日线。"}
    if not intraday_frame.empty:
        row = intraday_frame.iloc[-1]
        def _bar_value(field: str) -> float | None:
            value = row.get(field)
            return float(value) if pd.notna(value) else None
        intraday_reference = {
            "available": True,
            "trade_date": pd.Timestamp(intraday_frame.index[-1]).strftime("%Y-%m-%d"),
            "captured_at": captured_at,
            "price_qfq": _bar_value("close"),
            "open_qfq": _bar_value("open"),
            "high_qfq": _bar_value("high"),
            "low_qfq": _bar_value("low"),
            "change_pct_vendor": _bar_value("change_pct"),
            "amount_cny": _bar_value("amount_cny"),
            "volume_raw": _bar_value("volume_raw"),
            "note": "东方财富前复权日线接口当日未完成K线字段；价格/高低/成交数据仅为抓取时点参考，不是收盘值，且未纳入任何日频指标。",
        }
    metadata = {
        "code": code, "name": daily.attrs.get("name", ""), "provider": "Eastmoney public quote/flow/disclosure pages",
        "retrieved_at": captured_at, "price_start": daily.index[0].strftime("%Y-%m-%d"),
        "price_end": daily.index[-1].strftime("%Y-%m-%d"), "price_observations": int(len(daily)),
        "raw_price_start": daily_raw.index[0].strftime("%Y-%m-%d"),
        "raw_price_end": daily_raw.index[-1].strftime("%Y-%m-%d"),
        "raw_price_observations": int(len(daily_raw)),
        "intraday_available": bool(intraday_reference.get("available")),
        "intraday_trade_date": intraday_reference.get("trade_date"),
        "intraday_captured_at": intraday_reference.get("captured_at"),
        "daily_window_note": "日频收益、指标、情绪与市场模型均截至最近完整收盘日；n日收益按 close_t / close_{t-n} - 1 计算。抓取时未收盘的当日K线另列盘中参考，不纳入这些统计。",
        "market_hours_source_url": "https://www.szse.cn/English/services/trading/tradOverview/",
        "flow_start": flow.index[0].strftime("%Y-%m-%d") if not flow.empty else None,
        "flow_end": flow.index[-1].strftime("%Y-%m-%d") if not flow.empty else None,
        "flow_observations": int(len(flow)), "benchmark_code": benchmark_code,
        "benchmark_name": benchmark.attrs.get("name", benchmark_code),
        "benchmark_start": benchmark.index[0].strftime("%Y-%m-%d") if not benchmark.empty else None,
        "benchmark_end": benchmark.index[-1].strftime("%Y-%m-%d") if not benchmark.empty else None,
        "announcement_count": int(len(announcements)), "policy_catalog_count": int(len(policies)),
        "flow_source": flow_source, "flow_retrieved_at": flow.attrs.get("retrieved_at"),
        "source_errors": source_errors,
    }
    macro = None
    if include_us_clock:
        try:
            macro = fetch_us_merrill_clock_context()
        except Exception as exc:
            macro = {"available": False, "note": f"美国宏观时钟暂不可用：{type(exc).__name__}: {exc}"}
    flow_result = flow_analysis(flow)
    if "fund_flow" in source_errors:
        flow_result["note"] = "资金流源暂不可用；未将缺失值视为零。错误：" + source_errors["fund_flow"]
    announcement_result = summarize_announcements(announcements)
    if "announcements" in source_errors:
        announcement_result["note"] = "公告源暂不可用；不表示没有公告。错误：" + source_errors["announcements"]
    return {
        "metadata": metadata,
        "technical": technical_analysis(daily),
        "flow": flow_result,
        "announcements": announcement_result,
        "sentiment": market_sentiment_analysis(daily, benchmark),
        "policies": matched_policies.to_dict(orient="records"),
        "market_model": market_model(daily, benchmark),
        "merrill_clock": macro or {"available": False, "note": "已跳过美国宏观时钟数据下载。"},
        "intraday_reference": intraday_reference,
        "frames": {"daily": daily, "daily_raw": daily_raw, "intraday": intraday_frame,
                   "flow": flow, "announcements": announcements,
                   "benchmark": benchmark, "benchmark_raw": benchmark_raw, "policies": matched_policies},
    }
