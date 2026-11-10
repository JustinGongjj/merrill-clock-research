"""Self-contained HTML dashboard for the Merrill-clock research stack.

Reads the saved outputs of ``clock-data run`` plus the local database's dataset
catalogue and writes one HTML file with inline SVG charts. No CDN, no JavaScript
dependencies, no network access: the file works offline and can be archived or
shared as-is.
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

NAVY = "#0b2545"
BLUE = "#1b4f8a"
RED = "#c0392b"       # 涨 / 正向（中国习惯）
GREEN = "#1a7f52"     # 跌 / 负向（中国习惯）
GREY = "#8a97a8"
GRID = "#e6ebf2"
PHASE_COLORS = {
    "p_recovery": "#1b4f8a",
    "p_overheat": "#c0392b",
    "p_stagflation": "#c98a2b",
    "p_reflation": "#2e8b7a",
}
PHASE_LABELS = {
    "p_recovery": "复苏 Recovery",
    "p_overheat": "过热 Overheat",
    "p_stagflation": "滞胀 Stagflation",
    "p_reflation": "再通胀 Reflation",
}
WEIGHT_COLORS = {
    "w_SPY": "#1b4f8a",
    "w_IEF": "#2e8b7a",
    "w_GLD": "#c98a2b",
    "w_DBC": "#c0392b",
    "w___cash_zero__": "#b9c4d2",
}
WEIGHT_LABELS = {
    "w_SPY": "股票 SPY",
    "w_IEF": "债券 IEF",
    "w_GLD": "黄金 GLD",
    "w_DBC": "商品 DBC",
    "w___cash_zero__": "现金（零收益）",
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value * 100:+.{digits}f}%"


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def _tone(value: float | None) -> str:
    """Chinese market convention: positive red, negative green."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return "up" if value >= 0 else "down"


def _clean(values) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = float("nan")
        out.append(f)
    return out


def _x_labels(dates: list[str], left: float, width: float, n: int, y: float) -> str:
    if n < 2:
        return ""
    picks = sorted({round(k * (n - 1) / 5) for k in range(6)})
    parts = []
    for i in picks:
        label = str(dates[i])[:7]
        x = left + width * i / (n - 1)
        anchor = "start" if i == 0 else "end" if i == n - 1 else "middle"
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                     f'font-size="11" fill="#6b7a90">{_esc(label)}</text>')
    return "".join(parts)


def _line_chart(dates: list[str], series: list[tuple[str, list[float], str]],
                *, height: int = 290, y_fmt: str = "{:.1f}", fill: bool = False) -> str:
    width, left, right, top, bottom = 1000, 68, 18, 16, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for _, vs, _ in series for v in vs if not math.isnan(v)]
    if not values or not dates:
        return '<p class="empty">暂无可绘制数据</p>'
    lo, hi = min(values), max(values)
    if fill or lo > 0:
        lo = min(lo, 0.0)
    if hi == lo:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    n = len(dates)
    x = lambda i: left + plot_w * i / max(1, n - 1)
    y = lambda v: top + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'preserveAspectRatio="xMidYMid meet" style="display:block">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        yy = y(v)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 9}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#6b7a90">{_esc(y_fmt.format(v))}</text>')
    if lo < 0 < hi:
        y0 = y(0)
        parts.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{width - right}" y2="{y0:.1f}" '
                     f'stroke="#b9c4d2" stroke-width="1" stroke-dasharray="4 4"/>')
    for _, vs, color in series:
        pts = [(i, v) for i, v in enumerate(vs) if not math.isnan(v)]
        if not pts:
            continue
        if fill:
            y0 = y(0)
            polygon = (f'{x(pts[0][0]):.1f},{y0:.1f} '
                       + " ".join(f'{x(i):.1f},{y(v):.1f}' for i, v in pts)
                       + f' {x(pts[-1][0]):.1f},{y0:.1f}')
            parts.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.16" stroke="none"/>')
        coords = " ".join(f'{x(i):.1f},{y(v):.1f}' for i, v in pts)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                     f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(_x_labels(dates, left, plot_w, n, height - 8))
    parts.append("</svg>")
    return "".join(parts)


def _stacked_chart(dates: list[str], layers: list[tuple[str, list[float], str]],
                   *, height: int = 250, y_fmt: str = "{:.0%}") -> str:
    width, left, right, top, bottom = 1000, 68, 18, 16, 36
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(dates)
    if not n or not layers:
        return '<p class="empty">暂无可绘制数据</p>'
    clean_layers = [(name, _clean(vs), color) for name, vs, color in layers]
    totals = [sum(max(0.0, vs[i]) if not math.isnan(vs[i]) else 0.0
                  for _, vs, _ in clean_layers) for i in range(n)]
    hi = max(totals) if totals else 1.0
    hi = hi * 1.05 if hi > 0 else 1.0
    x = lambda i: left + plot_w * i / max(1, n - 1)
    y = lambda v: top + plot_h * (1 - v / hi)

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'preserveAspectRatio="xMidYMid meet" style="display:block">']
    for k in range(5):
        v = hi * k / 4
        yy = y(v)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{left - 9}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#6b7a90">{_esc(y_fmt.format(v))}</text>')
    cumulative = [0.0] * n
    for _, vs, color in clean_layers:
        upper = [cumulative[i] + (0.0 if math.isnan(vs[i]) else max(0.0, vs[i]))
                 for i in range(n)]
        top_pts = [f'{x(i):.1f},{y(upper[i]):.1f}' for i in range(n)]
        bot_pts = [f'{x(i):.1f},{y(cumulative[i]):.1f}' for i in reversed(range(n))]
        parts.append(f'<polygon points="{" ".join(top_pts + bot_pts)}" '
                     f'fill="{color}" fill-opacity="0.82" stroke="none"/>')
        cumulative = upper
    parts.append(_x_labels(dates, left, plot_w, n, height - 8))
    parts.append("</svg>")
    return "".join(parts)


def _legend(items: list[tuple[str, str]]) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{color}"></i>{_esc(label)}</span>'
        for label, color in items) + "</div>"


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="num">{cell}</td>' if idx >= numeric_from else f"<td>{cell}</td>"
            for idx, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _dataset_rows(db_path: Path) -> list[list[str]]:
    try:
        from .store import Store
        frame = Store(db_path).datasets()
    except Exception:
        return []
    rows = []
    for r in frame.itertuples(index=False):
        rows.append([
            _esc(r.dataset_id), _esc(getattr(r, "source_id", "")),
            _esc(str(getattr(r, "first_date", ""))[:10]),
            _esc(str(getattr(r, "last_date", ""))[:10]),
            f"{int(getattr(r, 'n_obs', 0)):,}",
        ])
    return rows


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _signal_layer_html(out_dir: Path, evaluated: pd.DataFrame, dates: list[str],
                       equity: list[float]) -> str:
    """Split the reported net result into signal layer, cost layer and portfolio layer."""
    if "gross_return" not in evaluated.columns or "cost_return" not in evaluated.columns:
        return '<p class="empty">结果文件缺少 gross_return / cost_return，无法拆分信号层与组合层。</p>'
    gross = _clean(evaluated["gross_return"])
    cost = _clean(evaluated["cost_return"])
    gross_equity: list[float] = []
    comp = 1.0
    for value in gross:
        comp *= (1.0 + value)
        gross_equity.append(comp)
    cumulative_cost = [g - n for g, n in zip(gross_equity, equity)]
    layer_chart = _line_chart(
        dates, [("信号层（未扣成本）", gross_equity, GREY), ("组合层（已扣成本）", equity, BLUE)],
        height=250, y_fmt="{:.1f}")
    cost_chart = _line_chart(dates, [("累计成本拖累", cumulative_cost, RED)],
                             height=200, y_fmt="{:.2f}", fill=True)
    turnover = float(evaluated["turnover"].mean()) if "turnover" in evaluated.columns else float("nan")
    rows = [
        ["信号层累计收益", "按 gross_return 复利", f'<span class="{_tone(gross_equity[-1] - 1)}">{_fmt_pct(gross_equity[-1] - 1, 1)}</span>'],
        ["组合层累计收益", "按 net_return 复利（报表口径）", f'<span class="{_tone(equity[-1] - 1)}">{_fmt_pct(equity[-1] - 1, 1)}</span>'],
        ["累计成本拖累", "信号层净值 − 组合层净值", f'<span class="down">{cumulative_cost[-1]:.3f}</span>'],
        ["平均单边换手", "每期 |w_t − w_{t-1}| 之和的均值", _fmt_pct(turnover)],
        ["单期成本合计", "成本序列求和", f'{sum(cost):.4f}'],
    ]
    return (_legend([("信号层（未扣成本）", GREY), ("组合层（已扣成本）", BLUE)])
            + layer_chart + "<h3>成本与换手</h3>"
            + _legend([("累计成本拖累", RED)]) + cost_chart
            + _table(["层级", "口径", "数值"], rows, numeric_from=2)
            + '<p class="sub">券商研究里「单因子多头超额」与「带约束指增超额」往往差一个数量级，'
              '差额就来自行业/市值中性、换手上限、买卖成本与涨跌停约束。这里只拆分了成本与换手，'
              '尚未实现组合约束，因此两层差额小于券商口径。</p>')


def _cross_section_html(payload: dict | None) -> str:
    if not payload:
        return ('<p class="empty">尚未生成横截面单因子检验结果，运行 <code>clock-data run</code> '
                '的第 3 步后会出现在这里。</p>')
    rows = []
    for name, report in payload.get("factors", {}).items():
        ic = report.get("ic_summary", {})
        fm = report.get("fama_macbeth", {})
        shape = report.get("layer_monotonicity", {}).get("shape", "—")
        rows.append([_esc(name),
                     f'<span class="{_tone(ic.get("ic_mean"))}">{_fmt_num(ic.get("ic_mean"), 4)}</span>',
                     _fmt_num(ic.get("icir"), 3),
                     _fmt_pct(ic.get("positive_ratio")),
                     _fmt_num(fm.get("t_stat"), 2),
                     _esc(shape)])
    caveats = "".join(f"<li>{_esc(c)}</li>" for c in payload.get("caveats", []))
    return (f'<p class="sub">标的池 {_esc(payload.get("universe_type", "—"))} · '
            f'{len(payload.get("universe", []))} 个标的 · 样本 '
            f'{_esc(payload.get("sample_start", "—"))} ~ {_esc(payload.get("sample_end", "—"))} · '
            f'分 {payload.get("n_groups", "—")} 组</p>'
            + _table(["因子", "RankIC 均值", "ICIR", "IC 胜率", "FM t 值", "分层形态"], rows, numeric_from=1)
            + f'<ul>{caveats}</ul>')


def _diagnostics_html(payload: dict | None) -> str:
    if not payload or "calibration" not in payload:
        return '<p class="empty">尚未生成相位校准诊断。</p>'
    c = payload["calibration"]
    if not c or c.get("error"):
        return f'<p class="empty">{_esc(c.get("error", "诊断未完成"))}</p>'
    rows = [
        ["增长方向 Brier", _fmt_num(c.get("brier_growth_up"), 4)],
        ["通胀方向 Brier", _fmt_num(c.get("brier_inflation_up"), 4)],
        ["四相命中率", _fmt_pct(c.get("quadrant_hit_rate"))],
        ["常数基线命中率", _fmt_pct(c.get("constant_baseline_hit_rate"))],
        ["命中率－基线", f'<span class="{_tone(c.get("hit_rate_minus_baseline"))}">'
                    f'{_fmt_pct(c.get("hit_rate_minus_baseline"))}</span>'],
        ["平均归一化熵", _fmt_num(c.get("mean_normalized_entropy"), 4)],
        ["主导相位平均持续期", f'{_fmt_num(c.get("mean_dominant_regime_duration_periods"), 2)} 期'],
    ]
    sensitivity = payload.get("temperature_sensitivity", {}).get("by_temperature", {})
    sens_rows = [[_esc(k), _fmt_num(v.get("mean_max_probability"), 4),
                  _fmt_num(v.get("mean_entropy"), 4)]
                 for k, v in sensitivity.items()]
    sens_table = (_table(["temperature", "平均最大概率", "平均归一化熵"], sens_rows, numeric_from=1)
                  if sens_rows else "")
    caveats = "".join(f"<li>{_esc(x)}</li>" for x in c.get("caveats", []))
    return (_table(["诊断指标", "数值"], rows, numeric_from=1)
            + "<h3>温度参数敏感性</h3>" + sens_table
            + f'<p class="sub">Brier 越小越好；四相均匀分布的命中率参考值为 25%。</p><ul>{caveats}</ul>')


def _manifest_html(payload: dict | None) -> str:
    if not payload:
        return '<p class="empty">尚未生成 run_manifest.json。</p>'
    unconfirmed = [k for k, v in payload.get("parameter_confirmations", {}).items()
                   if v == "default"]
    rows = [
        ["工作站 / 运行编号", f'{_esc(payload.get("workstation", "—"))} / {_esc(payload.get("run_id", "—"))}'],
        ["数据集", ", ".join(map(str, payload.get("dataset_ids", [])))],
        ["样本区间", f'{_esc(payload.get("sample_start", "—"))} ~ {_esc(payload.get("sample_end", "—"))}'],
        ["成本（单边 bps）", _fmt_num(payload.get("cost_bps"), 1)],
        ["测试变体数", str(payload.get("tested_variants", "—"))],
        ["选优规则", _esc(payload.get("selection_rule", "—"))],
        ["未显式确认的参数", ", ".join(unconfirmed) if unconfirmed else "无"],
        ["点时口径", _esc(payload.get("as_of_rule", "—"))],
    ]
    warnings = "".join(f"<li>{_esc(w)}</li>" for w in payload.get("data_warnings", []))
    issues = "".join(f"<li>{_esc(i)}</li>" for i in payload.get("validation_issues", []))
    return (_table(["契约字段", "取值"], rows, numeric_from=99)
            + "<h3>数据声明与校验提示</h3><ul>" + warnings + issues + "</ul>")


def build_dashboard(out_dir: Path, db_path: Path, target: Path | None = None) -> Path:
    """Build a single-file HTML dashboard from saved research outputs."""
    out_dir = Path(out_dir)
    metrics_path = out_dir / "metrics.json"
    backtest_path = out_dir / "clock_backtest.csv"
    missing = [p.name for p in (metrics_path, backtest_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {out_dir}. Run first: clock-data run --output {out_dir}"
        )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(backtest_path, parse_dates=["date"])
    if "evaluated" in frame.columns:
        evaluated = frame[frame["evaluated"] == 1].copy()
    else:
        evaluated = frame.dropna(subset=["net_return"]).copy()
    if evaluated.empty:
        raise ValueError("No evaluated months in clock_backtest.csv")

    dates = [d.strftime("%Y-%m-%d") for d in evaluated["date"]]
    equity = _clean(evaluated["equity"])
    drawdown = []
    peak = -math.inf
    for v in equity:
        peak = max(peak, v)
        drawdown.append((v / peak - 1.0) if peak > 0 else 0.0)

    # Equal-weight monthly-rebalanced benchmark, rebuilt from the saved monthly returns.
    benchmark: list[float] = []
    monthly_path = out_dir / "monthly_asset_returns.csv"
    if monthly_path.exists():
        try:
            monthly = pd.read_csv(monthly_path, parse_dates=["date"]).set_index("date").sort_index()
            core = [c for c in ("SPY", "IEF", "GLD", "DBC") if c in monthly.columns]
            bm_returns = monthly[core].mean(axis=1).reindex(evaluated["date"]).fillna(0.0)
            benchmark = _clean((1.0 + bm_returns).cumprod())
        except Exception:
            benchmark = []

    attribution_path = out_dir / "factor_attribution.json"
    attribution = None
    if attribution_path.exists():
        try:
            attribution = json.loads(attribution_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            attribution = None

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    source = metrics.get("data_source", "unknown")
    cagr = metrics.get("cagr")
    sharpe = metrics.get("sharpe_zero_rf")
    max_dd = metrics.get("max_drawdown")
    bm_cagr = metrics.get("benchmark_cagr")
    bm_sharpe = metrics.get("benchmark_sharpe_zero_rf")
    bm_dd = metrics.get("benchmark_max_drawdown")

    equity_series = [("策略净值", equity, BLUE)]
    if benchmark and len(benchmark) == len(equity):
        equity_series.append(("等权基准", benchmark, GREY))
    equity_chart = _line_chart(dates, equity_series, height=300, y_fmt="{:.1f}")
    dd_chart = _line_chart(dates, [("回撤", drawdown, BLUE)], height=230,
                           y_fmt="{:.0%}", fill=True)

    phase_layers = []
    for col, color in PHASE_COLORS.items():
        if col in evaluated.columns:
            phase_layers.append((PHASE_LABELS[col], _clean(evaluated[col]), color))
    phase_chart = _stacked_chart(dates, phase_layers, height=250) if phase_layers else ""
    phase_legend = _legend([(PHASE_LABELS[c], color) for c, color in PHASE_COLORS.items()
                            if c in evaluated.columns])

    weight_layers = []
    for col, color in WEIGHT_COLORS.items():
        if col in evaluated.columns:
            weight_layers.append((WEIGHT_LABELS[col], _clean(evaluated[col]), color))
    weight_chart = _stacked_chart(dates, weight_layers, height=250) if weight_layers else ""
    weight_legend = _legend([(WEIGHT_LABELS[c], color) for c, color in WEIGHT_COLORS.items()
                             if c in evaluated.columns])

    perf_rows = [
        ["年化收益 CAGR", f'<span class="{_tone(cagr)}">{_fmt_pct(cagr)}</span>',
         f'<span class="{_tone(bm_cagr)}">{_fmt_pct(bm_cagr)}</span>',
         f'<span class="{_tone((cagr or 0) - (bm_cagr or 0))}">{_fmt_pct((cagr or 0) - (bm_cagr or 0))}</span>'],
        ["Sharpe（零无风险利率）", _fmt_num(sharpe), _fmt_num(bm_sharpe),
         f'<span class="{_tone((sharpe or 0) - (bm_sharpe or 0))}">{(sharpe or 0) - (bm_sharpe or 0):+.3f}</span>'],
        ["最大回撤", f'<span class="down">{_fmt_pct(max_dd)}</span>',
         f'<span class="down">{_fmt_pct(bm_dd)}</span>',
         f'<span class="{_tone((max_dd or 0) - (bm_dd or 0))}">{_fmt_pct((max_dd or 0) - (bm_dd or 0))}</span>'],
        ["年化波动", _fmt_pct(metrics.get("annualized_volatility")), "—", "—"],
        ["累计收益", f'<span class="up">{_fmt_pct(metrics.get("total_return"), 1)}</span>', "—", "—"],
    ]
    perf_table = _table(["指标", "策略", "等权基准", "策略－基准"], perf_rows, numeric_from=1)

    factor_table = '<p class="empty">尚未生成因子归因结果，运行 <code>clock-data run</code> 后会出现在这里。</p>'
    if attribution:
        rows = []
        for name, item in attribution.get("coefficients", {}).items():
            label = "Alpha（截距）" if name == "const" else _esc(name)
            coef, t, p = item["coef"], item["t"], item["p"]
            mark = ' class="sig"' if p < 0.05 else ""
            rows.append([f"<span{mark}>{label}</span>",
                         f'<span class="{_tone(coef)}">{coef:+.4f}</span>',
                         f"{t:+.2f}", f"{p:.4f}"])
        factor_table = (
            f'<p class="sub">观测 {attribution.get("nobs", "—")} 个月 · '
            f'样本 {_esc(attribution.get("sample_start", "—"))} ~ {_esc(attribution.get("sample_end", "—"))} · '
            f'R² {_fmt_num(attribution.get("rsquared"), 4)} · HAC maxlags {attribution.get("maxlags", "—")} · '
            f'日期对齐：{_esc(attribution.get("alignment_note", "—"))}</p>'
            + _table(["因子", "载荷", "HAC t 值", "p 值"], rows, numeric_from=1)
            + '<p class="sub">载荷与截距只描述该历史样本，不是预测值，也不代表因果效应；'
              '显著性标记仅表示 p &lt; 0.05。</p>'
        )

    dataset_rows = _dataset_rows(Path(db_path))
    dataset_table = (_table(["数据集", "来源", "起始", "截止", "观测数"], dataset_rows, numeric_from=4)
                     if dataset_rows else '<p class="empty">本地数据库暂不可读。</p>')

    signal_layer_html = _signal_layer_html(out_dir, evaluated, dates, equity)
    cross_section_html = _cross_section_html(_load_json(out_dir / "cross_section_report.json"))
    diagnostics_html = _diagnostics_html(_load_json(out_dir / "phase_diagnostics.json"))
    manifest_html = _manifest_html(_load_json(out_dir / "run_manifest.json"))

    notes = [
        ("数据口径", metrics.get("data_warning")),
        ("权重性质", metrics.get("weights")),
        ("现金处理", metrics.get("cash_sleeve")),
        ("样本筛选", metrics.get("output_flag")),
        ("基准定义", metrics.get("benchmark")),
        ("交易成本", f"单边换手 {_fmt_num(metrics.get('cost_bps_per_one_way_turnover'), 1)} bps"),
        ("缺失月份", f"因核心 ETF 价格缺失丢弃 {metrics.get('months_dropped_for_missing_core_prices', '—')} 个月；"
                     f"预热期排除 {metrics.get('warmup_months_excluded', '—')} 个月"),
    ]
    note_html = "".join(
        f"<li><strong>{_esc(k)}：</strong>{_esc(v)}</li>"
        for k, v in notes if v
    )

    css = """
  :root{--navy:#0b2545;--navy2:#13315c;--blue:#1b4f8a;--accent:#c0392b;
    --bg:#f4f6f9;--card:#fff;--line:#dde3ec;--text:#1c2733;--muted:#5d6b7d;
    --good:#1a7f52;--warn:#a86400}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;
    font-size:15px;line-height:1.72}
  .wrap{max-width:1120px;margin:0 auto;padding:34px 26px 56px}
  header{background:linear-gradient(135deg,var(--navy),var(--blue));color:#fff;
    border-radius:14px;padding:28px 34px;margin-bottom:20px}
  header h1{margin:0 0 8px;font-size:26px;letter-spacing:.4px}
  header p{margin:0;color:#c9dcf2;font-size:13.5px}
  .tag{display:inline-block;background:rgba(255,255,255,.16);
    border:1px solid rgba(255,255,255,.28);border-radius:999px;
    padding:2px 11px;font-size:12px;margin-right:8px}
  .startbar{background:#fff8ec;border:1px solid #f0dcb6;border-left:4px solid var(--warn);
    border-radius:0 10px 10px 0;padding:13px 18px;margin-bottom:20px;font-size:14px}
  .startbar code{background:var(--navy);color:#e6eefb;padding:3px 9px;border-radius:5px;
    font-size:13px;margin-right:6px}
  h2{font-size:19px;color:var(--navy);margin:30px 0 12px;padding-left:11px;
    border-left:4px solid var(--accent)}
  .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-bottom:8px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:16px 18px}
  .card .num{font-size:25px;font-weight:700;color:var(--navy);line-height:1.3}
  .card .lbl{font-size:12.5px;color:var(--muted);margin-top:3px}
  .panel{background:var(--card);border:1px solid var(--line);border-radius:11px;
    padding:17px 20px;margin-bottom:16px}
  .legend{display:flex;flex-wrap:wrap;gap:16px;font-size:12.5px;color:var(--muted);margin:2px 0 8px}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .legend i{width:12px;height:12px;border-radius:3px;display:inline-block}
  .sub{font-size:12.8px;color:var(--muted);margin:8px 0}
  .empty{color:var(--muted);font-size:13px}
  table{width:100%;border-collapse:collapse;background:#fff;font-size:13.5px}
  th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}
  th{background:#eef2f8;color:var(--navy2);font-weight:600;white-space:nowrap}
  td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
  .up{color:var(--accent);font-weight:600}
  .down{color:var(--good);font-weight:600}
  .sig{color:var(--accent);font-weight:600}
  code{font-family:"SFMono-Regular",Menlo,Consolas,monospace;background:#eef2f8;
    padding:1px 5px;border-radius:4px;font-size:12.5px;color:var(--navy2)}
  pre{background:var(--navy);color:#e6eefb;border-radius:9px;padding:15px 18px;
    overflow-x:auto;font-size:12.8px;line-height:1.7;margin:10px 0}
  pre code{background:none;color:inherit;padding:0;font-size:inherit}
  ul{margin:8px 0 0;padding-left:22px}li{margin:5px 0}
  footer{border-top:1px solid var(--line);margin-top:32px;padding-top:15px;
    color:var(--muted);font-size:12.5px;display:flex;justify-content:space-between;gap:18px}
  @media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}.wrap{padding:18px 14px}
    footer{flex-direction:column}}
"""

    commands = """clock                    # 一键：跑完整离线链路 + 生成并打开本页（最常用）
clock menu                # 交互式菜单，不想记命令就用它
clock update              # 更新本地金融数据库（联网同步全部数据源）
clock status              # 查看数据库覆盖区间与同步日志
clock report              # 不重算，只用已有结果重新生成本页并打开
clock open                # 打开输出文件夹
clock help                # 完整命令速查

# 等价的底层命令
clock-data run --output ./outputs/local-run
clock-data dashboard --output ./outputs/local-run --open
merrill-clock-quant run --data-source auto"""

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>美林时钟研究台 · 结果仪表盘 · Justin</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>美林时钟研究台 · 结果仪表盘</h1>
  <p><span class="tag">merrill-clock-quant</span><span class="tag">merrill-clock-factor-regression</span><span class="tag">merrill-clock-data</span></p>
  <p style="margin-top:12px">样本 {_esc(metrics.get("sample_start", "—"))} ~ {_esc(metrics.get("sample_end", "—"))} ·
     {int(metrics.get("observations", 0))} 个有效月 · 数据来源 <strong>{_esc(source)}</strong> · 生成于 {_esc(generated)}</p>
</header>

<div class="startbar">
  <strong>怎么启动：</strong>打开「终端」，敲 <code>clock</code> 回车即可——自动跑完离线回测与因子归因、生成本页并打开。
  不想记命令就敲 <code>clock menu</code> 用菜单选。本页是单文件 HTML，可直接发送给别人或归档，无需联网。
</div>

<div class="cards">
  <div class="card"><div class="num">{_fmt_pct(cagr)}</div><div class="lbl">策略年化收益 CAGR</div></div>
  <div class="card"><div class="num">{_fmt_num(sharpe)}</div><div class="lbl">Sharpe（零无风险利率）</div></div>
  <div class="card"><div class="num">{_fmt_pct(max_dd)}</div><div class="lbl">最大回撤</div></div>
  <div class="card"><div class="num">{int(metrics.get("observations", 0))}</div><div class="lbl">有效样本月数</div></div>
</div>

<h2>一、净值走势</h2>
<div class="panel">
  {_legend([("策略净值", BLUE)] + ([("等权基准", GREY)] if benchmark else []))}
  {equity_chart}
  <p class="sub">净值以 1.0 为起点，含单边 10 bps 换手成本；等权基准为 SPY / IEF / GLD / DBC 月频等权再平衡、不计成本，按同期起点归一化。</p>
</div>

<h2>二、回撤</h2>
<div class="panel">
  {dd_chart}
  <p class="sub">阴影为净值相对历史高点的回撤幅度，全部为非正值；峰值回撤 {_fmt_pct(max_dd)}。</p>
</div>

<h2>三、绩效对比</h2>
<div class="panel">{perf_table}
  <p class="sub">差值列「策略－基准」按中国习惯标注：<span class="up">正值为红</span>、<span class="down">负值为绿</span>。基准的最大回撤越小越好，差值仅作数值对照。</p>
</div>

<h2>四、信号层 / 组合层 / 成本拆分</h2>
<div class="panel">{signal_layer_html}</div>

<h2>五、美林时钟相位概率</h2>
<div class="panel">{phase_legend}{phase_chart}
  <p class="sub">按宏观增长／通胀加速度滚动估计的四相概率，仅使用决策时点可得的信息（含发布滞后）。</p>
</div>

<h2>六、相位校准诊断</h2>
<div class="panel">{diagnostics_html}</div>

<h2>七、资产权重配置</h2>
<div class="panel">{weight_legend}{weight_chart}
  <p class="sub">权重为示例研究先验，不是投资建议；现金 sleeve 按显式零收益计提。</p>
</div>

<h2>八、FF5 因子归因（时间序列归因轨）</h2>
<div class="panel">{factor_table}</div>

<h2>九、横截面单因子检验（检验轨）</h2>
<div class="panel">{cross_section_html}</div>

<h2>十、研究契约 run_manifest</h2>
<div class="panel">{manifest_html}
  <p class="sub">成本、时点、标的池、约束、测试变体数随结果一起落盘；未显式确认的参数会被标记，不会被默认值悄悄掩盖。</p>
</div>

<h2>十一、本地金融数据库</h2>
<div class="panel">{dataset_table}
  <p class="sub">路径 <code>~/.local/share/clockdata/finance.db</code>；更新数据用 <code>clock update</code>。</p>
</div>

<h2>十二、命令速查</h2>
<div class="panel"><pre><code>{_esc(commands)}</code></pre></div>

<h2>附录 · 口径与限制</h2>
<div class="panel"><ul>
  {note_html}
  <li><strong>点时数据：</strong>FRED 返回最新修订值，不是发布当时的点时（vintage）数据；发布滞后为近似假设，严肃检验需接 ALFRED。</li>
  <li><strong>行情性质：</strong>Yahoo 为第三方复权价代理，非交易所认证行情；上游偶发限流，此时用 <code>auto</code> / <code>local</code> 模式。</li>
  <li><strong>样本长度：</strong>部分 ETF 上市较晚，样本实际从 2008-03 起；BIL 上市前现金按零收益计提，会低估当期国库券利息。</li>
  <li><strong>结论边界：</strong>绩效与因子载荷只描述历史样本，不构成投资建议或策略有效性证据，过往表现不预示未来收益。</li>
</ul></div>

<footer><span>美林时钟研究台 · 本地离线生成 · Justin</span><span>仅供研究参考，不构成投资建议；市场有风险，投资需谨慎</span></footer>

</div>
</body>
</html>
"""

    target = Path(target) if target else out_dir / "dashboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc, encoding="utf-8")
    return target
