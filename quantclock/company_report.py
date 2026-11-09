"""Phone-friendly Markdown and self-contained HTML company research reports."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .company_analysis import classify_announcement


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if k != "frames"}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, str)) else False:
        return None
    return value


def _pct(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value) * 100:+.2f}%"


def _pct_unsigned(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value) * 100:.2f}%"


def _money_yi(value: Any) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value) / 100_000_000:+.2f} 亿元"


def _number(value: Any, decimals: int = 2) -> str:
    return "—" if value is None or pd.isna(value) else f"{float(value):,.{decimals}f}"


def _chart_svg(daily: pd.DataFrame) -> str:
    d = daily.tail(180).copy()
    close = pd.to_numeric(d["close"], errors="coerce")
    ma20, ma60 = close.rolling(20).mean(), close.rolling(60).mean()
    series = [("收盘价（前复权）", close, "#175cd3"), ("MA20", ma20, "#f79009"), ("MA60", ma60, "#12b76a")]
    all_values = pd.concat([v for _, v, _ in series]).dropna()
    if all_values.empty:
        return "<p>有效价格不足，暂不绘图。</p>"
    ymin, ymax = float(all_values.min()), float(all_values.max())
    if ymax <= ymin:
        ymax = ymin + 1
    width, height, left, right, top, bottom = 960, 310, 62, 18, 18, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    def xy(i: int, value: float) -> tuple[float, float]:
        return left + i * plot_w / max(len(d) - 1, 1), top + (ymax - value) * plot_h / (ymax - ymin)
    grid = []
    for j in range(5):
        y = top + j * plot_h / 4
        val = ymax - j * (ymax - ymin) / 4
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e4e7ec"/><text x="{left-8}" y="{y+4:.1f}" text-anchor="end" fill="#667085" font-size="12">{val:.2f}</text>')
    paths = []
    for label, s, color in series:
        pts = [xy(i, float(v)) for i, v in enumerate(s) if pd.notna(v)]
        if len(pts) > 1:
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{point_text}"/>')
    dates = [d.index[0].strftime("%Y-%m-%d"), d.index[len(d)//2].strftime("%Y-%m-%d"), d.index[-1].strftime("%Y-%m-%d")]
    labels = [f'<text x="{left+i*plot_w/2:.1f}" y="{height-12}" text-anchor="middle" fill="#667085" font-size="12">{html.escape(day)}</text>' for i, day in enumerate(dates)]
    legend = " ".join(f'<line x1="{left+i*190}" y1="{height-2}" x2="{left+20+i*190}" y2="{height-2}" stroke="{color}" stroke-width="3"/><text x="{left+26+i*190}" y="{height+2}" fill="#475467" font-size="12">{html.escape(label)}</text>' for i, (label, _, color) in enumerate(series))
    return f'<svg viewBox="0 0 {width} {height+12}" role="img" aria-label="前复权价格和均线"><rect width="100%" height="100%" fill="#fff"/>{"".join(grid)}{"".join(paths)}{"".join(labels)}{legend}</svg>'


def _announcements_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return '<p class="muted">没有获取到公告列表；这不表示没有公告。</p>'
    rows = []
    for _, row in frame.head(20).iterrows():
        title = html.escape(str(row.get("title", "")))
        category = html.escape(str(row.get("category", ""))) or "披露"
        published = row.get("published_at")
        date = published.strftime("%Y-%m-%d %H:%M") if pd.notna(published) else "未提供时间"
        kind = html.escape(classify_announcement(str(row.get("title", "")), str(row.get("category", ""))))
        code = html.escape(str(row.get("article_code", "")))
        source = html.escape(str(row.get("source_url", "")), quote=True)
        rows.append(f'<tr><td>{date}</td><td><a href="{source}" target="_blank" rel="noopener">{title}</a><div class="muted">{category} · {kind} · 编号 {code}</div></td></tr>')
    return '<table><thead><tr><th>公开时间</th><th>标题 / 规则分类</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>\n<p class="muted">标题分类是检索辅助，不判断利好或利空。请打开原始披露核对全文。</p>'


def _policies_table(records: list[dict]) -> str:
    if not records:
        return '<p class="muted">政策清单中没有命中关键词的文件；可能是清单缺项或关键字不匹配，不代表政策没有影响。</p>'
    rows = []
    for row in records:
        title = html.escape(str(row.get("title", "")))
        date = html.escape(str(row.get("published_at", "")))
        publisher = html.escape(str(row.get("publisher", "")))
        terms = html.escape(str(row.get("relevant_terms", "")))
        note = html.escape(str(row.get("scope_note", "")))
        source = html.escape(str(row.get("source_url", "")), quote=True)
        title_link = f'<a href="{source}" target="_blank" rel="noopener">{title}</a>' if source else title
        rows.append(f'<tr><td>{date}</td><td>{title_link}<div class="muted">{publisher} · 命中词：{terms}</div><div>{note}</div></td></tr>')
    return '<table><thead><tr><th>发布时间</th><th>公开政策文本（关键词关联）</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'


def _sources(summary: dict) -> list[dict[str, str]]:
    metadata = summary["metadata"]
    flow_source = metadata.get("flow_source") or "东方财富公开个股资金流页面/接口"
    flow_caveat = "按订单规模分类的供应商估算；不是交易所确认的机构资金流。"
    if metadata.get("flow_retrieved_at"):
        flow_caveat += f" 采集/记录时间：{metadata['flow_retrieved_at']}。"
    return [
        {"dimension": "技术面 / 量价", "source": "东方财富公开日线接口（前复权）", "url": "https://quote.eastmoney.com/", "caveat": "第三方公开数据接口；价格口径和公司行动处理依赖数据方。"},
        {"dimension": "资金面", "source": flow_source, "url": f"https://data.eastmoney.com/zjlx/detail.html?fc={metadata['code']}", "caveat": flow_caveat},
        {"dimension": "消息面", "source": "东方财富公司公告列表", "url": f"https://data.eastmoney.com/stockdata/{summary['metadata']['code']}.html", "caveat": "仅拉取标题和公开时间；重要信息应阅读公告原文。"},
        {"dimension": "政策面", "source": "用户维护的政策观察清单（逐条来源见匹配记录）", "url": "", "caveat": "关键词命中是检索线索，不代表公司受益或政策有效性判断。"},
        {"dimension": "美林时钟", "source": "FRED INDPRO/CPIAUCSL（仅美国宏观代理）", "url": "https://fred.stlouisfed.org/", "caveat": "使用最新修订值和假设滞后；不能代替中国宏观状态。"},
        {"dimension": "量化参照", "source": "创业板指 399006（指数价格序列）", "url": "https://quote.eastmoney.com/sz399006.html", "caveat": "价格指数代理，非含分红总回报；市场模型仅描述历史。"},
        {"dimension": "时点对齐", "source": "深交所官方交易时段说明", "url": metadata.get("market_hours_source_url") or "https://www.szse.cn/English/services/trading/tradOverview/", "caveat": "交易日连续竞价9:30–11:30、13:00–14:57，收盘集合竞价14:57–15:00；当日未收盘K线不得作为日收盘统计。"},
    ]


def render_markdown(summary: dict) -> str:
    m, t, f, s, a = summary["metadata"], summary["technical"], summary["flow"], summary["sentiment"], summary["announcements"]
    clock = summary.get("merrill_clock", {})
    quant = summary.get("market_model", {})
    intraday = summary.get("intraday_reference", {})
    lines = [
        f"# {m.get('name') or 'A股公司'}（{m['code']}）研究快照",
        "",
        f"行情抓取时间：{m.get('retrieved_at') or '未记录'}（北京时间）；日频收盘分析覆盖 {m.get('price_start')} 至 {m.get('price_end')}，{m.get('price_observations')} 个完整交易日。",
        f"口径：5/20/60/120日收益均以最近完整收盘日 {t.get('as_of')} 为终点，按 close_t / close_(t-n) − 1 计算；当日未完成K线不进入收益、均线、RSI、情绪统计或市场回归。",
        "",
        "> 研究用途，不构成投资建议、证券买卖建议或收益承诺；不包含自动下单。五个维度分开呈现，不合成为未经校准的总分。公开网页可访问不等于可自由再分发。",
        "",
        "## 一页摘要",
        "",
        f"- 技术面：{t['trend_structure']}；20日/60日收益 {_pct(t.get('return_20d'))} / {_pct(t.get('return_60d'))}；RSI(14) {_number(t.get('rsi14'))}。",
        f"- 资金面：完整5/20日窗口主力净额 {_money_yi(f.get('main_net_5d_cny'))} / {_money_yi(f.get('main_net_20d_cny'))}；本次 {m.get('flow_observations')} 条，覆盖 {m.get('flow_start')} 至 {m.get('flow_end')}。",
        f"- 消息面：已抓取 {a.get('count', 0)} 条披露标题；最新为“{a.get('latest_title') or '无'}”，分类仅作索引。",
        f"- 情绪面：{s.get('classification', '不可用')}；这是量价交易代理，不含社交平台发言。",
        f"- 政策面：关键词匹配 {len(summary.get('policies', []))} 份公开政策文本；匹配不等于公司受益判断。",
        "",
        "## 五个维度",
        "",
        "### 技术面",
        "",
        f"最近完整前复权收盘价 { _number(t.get('close_qfq')) }（{t.get('as_of')}）；5/20/60/120日收益分别为 {_pct(t.get('return_5d'))} / {_pct(t.get('return_20d'))} / {_pct(t.get('return_60d'))} / {_pct(t.get('return_120d'))}。MA20 {_number(t.get('ma20'))}，MA60 {_number(t.get('ma60'))}，RSI(14) {_number(t.get('rsi14'))}（{t.get('rsi_state')}），近20日年化波动 {_pct_unsigned(t.get('annualized_volatility_20d'))}，60日最大回撤 {_pct(t.get('max_drawdown_60d'))}。",
        "",
        f"趋势结构：{t.get('trend_structure')}。指标是基于历史价格的描述，不是入场/退出规则。",
        "",
        "### 资金面",
        "",
        f"完整5日/20日平台主力净额：{_money_yi(f.get('main_net_5d_cny'))} / {_money_yi(f.get('main_net_20d_cny'))}；近5日正值天数 {_number(f.get('positive_days_5d'), 0)}；近20日正值天数 {_number(f.get('positive_days_20d'), 0)}；20日平均净占比 {_number(f.get('main_net_ratio_mean_20d_pct'))}%。",
        "",
        f.get('note', '资金流口径不可用。'),
        "",
        "### 消息面",
        "",
        f"最新公告/披露标题（列表采样）：{a.get('latest_title') or '暂无'}（{a.get('latest_published_at') or '时间未提供'}）。近次返回类别分布：{a.get('categories', {})}。标题规则只用于把公告分组，不推断利好利空；公司公告全文请在下表入口核验。",
        "",
        "|公开时间|公告标题|类别|数据源编号|",
        "|---|---|---|---|",
    ]
    if intraday.get("available"):
        change = intraday.get("change_pct_vendor")
        change_text = "未提供" if change is None else f"{float(change):+.2f}%"
        marker = lines.index("## 五个维度")
        lines[marker:marker] = [
            "## 盘中参考（未完成K线，不纳入日频分析）",
            "",
            f"截至 {intraday.get('captured_at')}（北京时间），东方财富前复权日线接口的 {intraday.get('trade_date')} 当日未完成K线 `close` 字段为 {_number(intraday.get('price_qfq'))}，数据源涨跌幅字段 {change_text}；盘中开/高/低为 {_number(intraday.get('open_qfq'))} / {_number(intraday.get('high_qfq'))} / {_number(intraday.get('low_qfq'))}。",
            "",
            f"{intraday.get('note')}日频技术、情绪和市场回归均截至最近完整收盘日 {t.get('as_of')}。深交所列示的连续竞价为9:30–11:30、13:00–14:57，收盘集合竞价为14:57–15:00；该抓取时点早于收盘。",
            "",
        ]
    for _, row in summary["frames"]["announcements"].head(20).iterrows():
        dt = row.get("published_at")
        dt = dt.strftime("%Y-%m-%d %H:%M") if pd.notna(dt) else "未提供"
        title = str(row.get("title", "")).replace("|", "\\|")
        cat = classify_announcement(str(row.get("title", "")), str(row.get("category", "")))
        lines.append(f"|{dt}|{title}|{cat}|{row.get('article_code', '')}|")
    lines += ["", "### 情绪面（交易情绪代理）", "", f"{s.get('classification', '不可用')}。5日收益 {_pct(s.get('return_5d'))}；5日均成交额/此前20日中位数 {_number(s.get('amount_5d_vs_prior20d_median'))} 倍；5日换手率/此前20日中位数 {_number(s.get('turnover_5d_vs_prior20d_median'))} 倍；相对创业板指20日收益差 {_pct(s.get('relative_return_20d_vs_benchmark'))}。", "", s.get("scope_note", "不含社会化媒体情绪数据。"), "", "### 政策面", ""]
    if summary.get("policies"):
        lines += ["|发布日期|文件|发布机构|匹配关键词|原文|", "|---|---|---|---|---|"]
        for p in summary["policies"]:
            title = str(p.get("title", "")).replace("|", "\\|")
            src = p.get("source_url", "")
            title_cell = f"[{title}]({src})" if src else title
            source_cell = f"[公开原文]({src})" if src else "未提供"
            lines.append(f"|{p.get('published_at','')}|{title_cell}|{p.get('publisher','')}|{p.get('relevant_terms','')}|{source_cell}|")
            lines.append(f"|说明|{p.get('scope_note','')}||||")
    else:
        lines.append("政策清单中未命中关键词；不表示没有相关政策。")
    lines += ["", "当前清单为人工维护的公开政策观察表；历史文件会保留并标注时间，不自动判断其现行有效性。", "", "## 美林时钟与量化参照", ""]
    if clock.get("available"):
        probs = clock.get("probabilities", {})
        prob_text = "、".join(f"{REGIME_CN.get(k,k)} {_number(v*100,1)}%" for k,v in probs.items())
        lines += [f"美国宏观代理时钟截至 {clock.get('as_of')} 的最大概率象限：**{clock.get('phase_cn')}**；概率分布：{prob_text}。增长动能代理 { _number(clock.get('growth_accel_proxy')) }，通胀动能代理 {_number(clock.get('inflation_accel_proxy')) }。", "", clock.get("note", ""), ""]
    else:
        lines += [clock.get("note", "本次未连接美林时钟数据。"), ""]
    if quant.get("available"):
        lines += [f"以{quant.get('benchmark_factor')}为单一市场因子的近{quant.get('observations')}个共同交易日回归：beta {_number(quant.get('beta'))}，HAC t值 {_number(quant.get('beta_hac_t'))}，R² {_number(quant.get('r_squared'))}，相关系数 {_number(quant.get('correlation'))}。", "", quant.get("note", ""), ""]
    else:
        lines += [quant.get("note", "市场模型不可用。"), ""]
    lines += ["## 数据与边界", "", f"- 用于日频分析的行情和创业板指：{m.get('price_start')}—{m.get('price_end')}；前复权口径用于个股技术指标，成交量保留数据方原始单位。原始CSV可另外包含截至抓取时点的未完成当日K线，并以 `bar_status` 标记；该行仅供盘中参考，不参与日频计算。", f"- 资金流：{m.get('flow_observations')} 条，覆盖 {m.get('flow_start')}—{m.get('flow_end')}；仅完整5/20条窗口才显示对应统计，分类值属于供应商估算。", f"- 消息：{m.get('announcement_count')} 条公告/披露标题，来自公开公司披露列表；需阅读原文确认全文和发布时间。", "- 政策：本地 `data/policy_watchlist.csv`；关键词匹配依赖人工维护的主题词，不能推出订单、收入、利润或估值变化。", "- 美林时钟：仅美国FRED工业生产/CPI作为既有宏观研究背景；最新修订值 + 假设滞后，并非中国周期。", "- 量化基准：创业板指数用于市场相对参照，指数非总回报；回归截距/系数只描述所取样本，不是未来alpha或回报预测。", "- 当前版本没有自动财报建模、社交舆情抓取、行业估值比较、股票回测或交易执行。", "", "## 来源", ""]
    for src in _sources(summary):
        source_label = f"[{src['source']}]({src['url']})" if src["url"] else src["source"]
        lines.append(f"- {src['dimension']}：{source_label}。{src['caveat']}")
    lines += ["", "## 研究风险提示", "", "此页面是公开数据研究原型，不构成投资建议、证券买卖建议或收益承诺。数据源可变更、延迟、修订或限流；公开可访问不代表可自由复制或再分发。指标与分类均需结合原始数据核验。", ""]
    return "\n".join(lines)


# Keep Chinese labels available for the renderer without a second implementation.
from .company_analysis import REGIME_CN


def render_html(summary: dict) -> str:
    m, t, f, s, a = summary["metadata"], summary["technical"], summary["flow"], summary["sentiment"], summary["announcements"]
    clock, quant = summary.get("merrill_clock", {}), summary.get("market_model", {})
    intraday = summary.get("intraday_reference", {})
    h = lambda x: html.escape(str(x if x is not None else "—"))
    price_chart = _chart_svg(summary["frames"]["daily"])
    ann_table = _announcements_table(summary["frames"]["announcements"])
    policy_table = _policies_table(summary.get("policies", []))
    clock_html = '<p class="muted">' + h(clock.get("note", "未连接")) + '</p>'
    if clock.get("available"):
        probs = clock.get("probabilities", {})
        clock_html = f'<p><b>{h(clock.get("phase_cn"))}</b> · 美国宏观代理 · {h(clock.get("as_of"))}</p><div class="chips">' + ''.join(f'<span>{h(REGIME_CN.get(k,k))} {float(v)*100:.1f}%</span>' for k,v in probs.items()) + f'</div><p class="muted">{h(clock.get("note", ""))}</p>'
    quant_html = f'<p>beta <b>{_number(quant.get("beta"))}</b> · HAC t <b>{_number(quant.get("beta_hac_t"))}</b> · R² <b>{_number(quant.get("r_squared"))}</b> · n={h(quant.get("observations"))}</p><p class="muted">{h(quant.get("note", ""))}</p>' if quant.get("available") else f'<p class="muted">{h(quant.get("note", "不可用"))}</p>'
    source_list = "".join(f'<li><b>{h(x["dimension"])}</b>：' + (f'<a href="{html.escape(x["url"], quote=True)}" target="_blank" rel="noopener">{h(x["source"])}</a>' if x["url"] else h(x["source"])) + f'<br><span class="muted">{h(x["caveat"])}</span></li>' for x in _sources(summary))
    intraday_html = ""
    if intraday.get("available"):
        change = intraday.get("change_pct_vendor")
        change_text = "—" if change is None else f"{float(change):+.2f}%"
        intraday_html = (
            '<section class="card"><h2>盘中参考（未完成K线）</h2>'
            f'<p><b>抓取时参考价：</b>{_number(intraday.get("price_qfq"))}（前复权接口字段） · '
            f'{h(intraday.get("trade_date"))} · {h(intraday.get("captured_at"))}（北京时间）</p>'
            f'<p>数据源涨跌幅字段：{h(change_text)}；盘中开/高/低：{_number(intraday.get("open_qfq"))} / {_number(intraday.get("high_qfq"))} / {_number(intraday.get("low_qfq"))}</p>'
            f'<p class="warning">{h(intraday.get("note"))} 日频收益、技术指标、量价情绪和市场回归截至 {h(t.get("as_of"))} 的最近完整收盘日，不包含该盘中值。</p></section>'
        )
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{h(m.get('name'))}（{h(m['code'])}）研究快照</title>
<style>
:root{{--ink:#101828;--muted:#667085;--line:#eaecf0;--blue:#175cd3;--bg:#f5f7fa;--card:#fff;--green:#067647;--amber:#b54708}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Noto Sans SC",sans-serif}}main{{max-width:1000px;margin:0 auto;padding:22px 16px 50px}}header,.card{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;margin:0 0 14px;box-shadow:0 2px 5px #10182808}}h1{{font-size:26px;line-height:1.25;margin:0 0 7px}}h2{{font-size:20px;margin:0 0 12px}}h3{{font-size:16px;margin:18px 0 5px}}p{{margin:7px 0}}.muted,.sub{{color:var(--muted);font-size:13px}}.warning{{border-left:4px solid #f79009;background:#fffaeb;padding:11px 13px;border-radius:7px;color:#7a2e0e;margin-top:14px}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}.kpi{{border:1px solid var(--line);border-radius:12px;padding:13px}}.kpi b{{display:block;font-size:21px;margin-top:4px}}.tag{{display:inline-block;background:#eff8ff;border-radius:100px;padding:3px 9px;margin:3px;color:#1849a9;font-size:12px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}a{{color:var(--blue);text-decoration:none}}a:hover{{text-decoration:underline}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px 8px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:var(--muted);font-weight:600}}svg{{width:100%;height:auto}}.chips{{display:flex;gap:8px;flex-wrap:wrap}}.chips span{{padding:5px 10px;border-radius:999px;background:#f2f4f7;font-size:13px}}footer{{color:var(--muted);font-size:12px;padding:6px 3px 16px}}
@media(max-width:680px){{main{{padding:12px 10px 32px}}header,.card{{padding:15px;border-radius:12px}}h1{{font-size:22px}}.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.two{{grid-template-columns:1fr}}table{{font-size:13px}}th,td{{padding:8px 5px}}}}
</style></head><body><main>
<header><h1>{h(m.get('name'))}（{h(m['code'])}）研究快照</h1><div class="sub">行情抓取：{h(m.get('retrieved_at'))}（北京时间） · 完整收盘分析：{h(m.get('price_start'))}—{h(m.get('price_end'))} · {h(m.get('price_observations'))} 个交易日</div><div class="warning"><b>研究用途，不构成投资建议或买卖建议。</b>不承诺收益、不下单；五个维度分开呈现，不合成未经校准的总分。公开页面可访问不等于可以自由复制或再分发。</div></header>
<section class="card"><h2>五维快览</h2><div class="grid">
<div class="kpi"><span class="sub">技术面 · 趋势结构</span><b style="font-size:16px">{h(t.get('trend_structure'))}</b><span class="sub">20日收益 {_pct(t.get('return_20d'))} · RSI {_number(t.get('rsi14'))}</span></div>
<div class="kpi"><span class="sub">资金面 · 供应商估算</span><b>{_money_yi(f.get('main_net_5d_cny'))}</b><span class="sub">近5日主力净额 · 20日 {_money_yi(f.get('main_net_20d_cny'))}</span></div>
<div class="kpi"><span class="sub">消息面 · 最新披露</span><b style="font-size:15px">{h(a.get('latest_title') or '暂无')}</b><span class="sub">{h(a.get('latest_published_at') or '')} · 标题分组不判断利好利空</span></div>
<div class="kpi"><span class="sub">情绪面 · 量价代理</span><b style="font-size:15px">{h(s.get('classification'))}</b><span class="sub">不含社交媒体情绪</span></div>
<div class="kpi"><span class="sub">政策面 · 文本命中</span><b>{len(summary.get('policies', []))} 份</b><span class="sub">关键词关联，不代表公司受益</span></div>
<div class="kpi"><span class="sub">量化参照 · 市场模型</span><b>β {_number(quant.get('beta')) if quant.get('available') else '—'}</b><span class="sub">{h(quant.get('benchmark_factor') or '')} · 近{h(quant.get('observations') or '—')}日</span></div>
</div></section>
{intraday_html}
<section class="card"><h2>技术面：价格与均线</h2>{price_chart}<div class="grid"><div class="kpi"><span class="sub">最近完整前复权收盘</span><b>{_number(t.get('close_qfq'))}</b><span class="sub">{h(t.get('as_of'))}</span></div><div class="kpi"><span class="sub">5 / 20 / 60日收益</span><b style="font-size:17px">{_pct(t.get('return_5d'))} / {_pct(t.get('return_20d'))}</b><span class="sub">60日 {_pct(t.get('return_60d'))}</span></div><div class="kpi"><span class="sub">RSI(14) / 20日年化波动</span><b>{_number(t.get('rsi14'))} / {_pct_unsigned(t.get('annualized_volatility_20d'))}</b><span class="sub">60日最大回撤 {_pct(t.get('max_drawdown_60d'))}</span></div></div><p class="muted">均线、RSI、波动与回撤均为历史描述，不是买卖条件。</p></section>
<section class="two"><div class="card"><h2>资金面</h2><p><b>近5日主力净额：</b>{_money_yi(f.get('main_net_5d_cny'))}</p><p><b>近20日主力净额：</b>{_money_yi(f.get('main_net_20d_cny'))}</p><p><b>5日 / 20日正值天数：</b>{_number(f.get('positive_days_5d'),0)} / {_number(f.get('positive_days_20d'),0)}</p><p><b>数据覆盖：</b>{h(m.get('flow_start'))}—{h(m.get('flow_end'))}，{h(m.get('flow_observations'))}条</p><p class="muted">{h(f.get('note'))}</p></div><div class="card"><h2>情绪面（交易代理）</h2><p><b>{h(s.get('classification'))}</b></p><p>5日均成交额 / 前20日中位数：{_number(s.get('amount_5d_vs_prior20d_median'))}倍</p><p>5日均换手率 / 前20日中位数：{_number(s.get('turnover_5d_vs_prior20d_median'))}倍</p><p>相对创业板指20日收益差：{_pct(s.get('relative_return_20d_vs_benchmark'))}</p><p class="muted">{h(s.get('scope_note'))}</p></div></section>
<section class="card"><h2>消息面：公开公司披露</h2><p class="muted">抓取 {h(a.get('count'))} 条标题；规则分类只用于检索导航，不对内容做利好/利空判断。</p>{ann_table}</section>
<section class="card"><h2>政策面：行业文本关键词观察</h2>{policy_table}<p class="muted">政策文件的行业范围、期限和原文需逐项核实；关键词命中不代表信维通信受益或政策仍然有效。</p></section>
<section class="two"><div class="card"><h2>美林时钟背景</h2>{clock_html}</div><div class="card"><h2>量化：相对基准市场模型</h2>{quant_html}</div></section>
<section class="card"><h2>数据来源与边界</h2><ul>{source_list}</ul><p class="muted">当前原型未自动建模财报、社交舆情或行业估值；未做个股策略回测或交易模拟。日线前复权与创业板价格指数不等同可交易总回报。</p></section>
<footer>量化研究工作台原型 · 生成时间 {h(datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'))} · 公开信息研究，不构成投资建议。</footer>
</main></body></html>'''


def write_snapshot(output_dir: str | Path, summary: dict) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = summary["frames"]
    daily_export = frames.get("daily_raw", frames["daily"]).copy()
    daily_export["bar_status"] = "completed_session_close"
    intraday_frame = frames.get("intraday", pd.DataFrame())
    if intraday_frame is not None and not intraday_frame.empty:
        daily_export.loc[intraday_frame.index, "bar_status"] = "incomplete_intraday_bar"
    daily_export.to_csv(out / "daily_ohlcv_qfq.csv", index_label="date")
    benchmark_export = frames.get("benchmark_raw", frames["benchmark"]).copy()
    benchmark_export["bar_status"] = "completed_session_close"
    if summary.get("intraday_reference", {}).get("available") and not benchmark_export.empty:
        intraday_date = pd.Timestamp(summary["intraday_reference"]["trade_date"]).date()
        benchmark_export.loc[pd.Index(benchmark_export.index.date) == intraday_date, "bar_status"] = "incomplete_intraday_bar"
    benchmark_export.to_csv(out / "benchmark_daily.csv", index_label="date")
    frames["flow"].to_csv(out / "vendor_money_flow.csv", index_label="date")
    pd.DataFrame([summary.get("intraday_reference", {"available": False})]).to_csv(out / "intraday_reference.csv", index=False)
    announcements = frames["announcements"].copy()
    if not announcements.empty:
        announcements["event_type"] = announcements.apply(lambda r: classify_announcement(str(r.get("title", "")), str(r.get("category", ""))), axis=1)
    announcements.to_csv(out / "company_disclosures.csv", index=False)
    frames["policies"].to_csv(out / "matched_policy_watchlist.csv", index=False)
    clean = _clean(summary)
    (out / "metrics.json").write_text(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "snapshot.md").write_text(render_markdown(summary), encoding="utf-8")
    (out / "snapshot.html").write_text(render_html(summary), encoding="utf-8")
    source_manifest = {
        "generated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "metadata": clean["metadata"], "sources": _sources(summary),
        "product_caveats": [
            "Eastmoney webpage endpoints are unofficial and can change or rate-limit.",
            "A same-day daily bar fetched before the Shenzhen close is marked provisional and excluded from daily-close analytics; the captured value is emitted separately as an intraday reference.",
            "Fund-flow numbers use provider order-size classifications and are not exchange-certified investor identity data.",
            "FRED US macro series use latest revisions; the assumed release lag is an approximation.",
            "Policy matches are keyword retrieval aids, not legal or company-impact conclusions.",
            "No recommendation, return guarantee, trading execution, or non-public data is used.",
        ],
    }
    (out / "source_manifest.json").write_text(json.dumps(_clean(source_manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {name: str((out / name).resolve()) for name in ["snapshot.html", "snapshot.md", "metrics.json", "source_manifest.json", "daily_ohlcv_qfq.csv", "intraday_reference.csv", "vendor_money_flow.csv", "company_disclosures.csv", "benchmark_daily.csv", "matched_policy_watchlist.csv"]}
