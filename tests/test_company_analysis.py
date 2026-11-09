import numpy as np
import pandas as pd
from pathlib import Path

from quantclock.company_analysis import (
    technical_analysis, flow_analysis, market_sentiment_analysis,
    classify_announcement, match_policy_documents, market_model,
    split_incomplete_daily_bar, build_company_snapshot,
)
from quantclock.company_report import render_markdown, render_html, write_snapshot
from quantclock.company_data import read_money_flow_csv, _parse_eastmoney_display_time


def sample_daily(n=150):
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = pd.Series(50 + np.arange(n) * 0.08 + np.sin(np.arange(n) / 4), index=idx)
    return pd.DataFrame({
        "open": close * 0.995, "close": close, "high": close * 1.02,
        "low": close * 0.98, "volume_raw": 1000 + np.arange(n),
        "amount_cny": 1e8 + np.arange(n) * 1e6, "turnover_pct": 1.2 + np.arange(n) * .001,
    }, index=idx)


def test_technical_analysis_reports_descriptive_metrics():
    result = technical_analysis(sample_daily())
    assert result["as_of"] == sample_daily().index[-1].strftime("%Y-%m-%d")
    assert np.isfinite(result["rsi14"])
    assert result["return_20d"] is not None
    assert "买卖" in result["window_note"]


def test_flow_missing_is_not_misreported_as_zero():
    result = flow_analysis(pd.DataFrame())
    assert result["available"] is False
    assert "不以0代替缺失" in result["note"]


def test_vendor_flow_window_sums_are_explicitly_separated():
    idx = pd.bdate_range("2025-01-01", periods=25)
    flow = pd.DataFrame({"main_net_inflow_cny": np.ones(25) * 1_000_000,
                         "main_net_inflow_ratio_pct": np.ones(25)}, index=idx)
    result = flow_analysis(flow)
    assert result["main_net_5d_cny"] == 5_000_000
    assert result["main_net_20d_cny"] == 20_000_000
    assert "估算" in result["note"]


def test_short_flow_sample_does_not_claim_a_20_day_total():
    idx = pd.bdate_range("2026-11-02", periods=5)
    flow = pd.DataFrame({"main_net_inflow_cny": np.arange(1, 6) * 1_000_000,
                         "main_net_inflow_ratio_pct": np.ones(5)}, index=idx)
    result = flow_analysis(flow)
    assert result["main_net_5d_cny"] == 15_000_000
    assert result["main_net_20d_cny"] is None
    assert result["positive_days_20d"] is None


def test_synthetic_flow_csv_is_read_without_bundled_vendor_data(tmp_path):
    from quantclock.company_data import FLOW_COLUMNS

    dates = pd.bdate_range("2026-11-02", periods=5)
    rows = [{"date": day.strftime("%Y-%m-%d"), **{column: 0 for column in FLOW_COLUMNS[1:]}}
            for day in dates]
    path = tmp_path / "synthetic_flow.csv"
    pd.DataFrame(rows, columns=FLOW_COLUMNS).to_csv(path, index=False)
    frame = read_money_flow_csv(str(path), "300136")
    assert len(frame) == 5
    assert frame.index[0].strftime("%Y-%m-%d") == "2026-11-02"
    assert frame.attrs["retrieved_at"] == "未提供采集时间"
    assert frame.attrs["provider"] == "User-supplied public-data CSV"


def test_eastmoney_display_time_milliseconds_are_parsed():
    result = _parse_eastmoney_display_time("2026-10-29 16:30:36:518")
    assert result.isoformat() == "2026-10-29T16:30:36.518000"


def test_same_day_bar_before_exchange_close_is_separated():
    idx = pd.to_datetime(["2026-11-06", "2026-11-09"])
    daily = pd.DataFrame({"open": [42.01, 44.26], "close": [42.30, 50.76],
                          "high": [43.63, 50.76], "low": [41.95, 44.25]}, index=idx)
    closed, provisional = split_incomplete_daily_bar(
        daily, retrieved_at="2026-11-09T10:21:39+08:00"
    )
    assert closed.index[-1].strftime("%Y-%m-%d") == "2026-11-06"
    assert provisional.index[-1].strftime("%Y-%m-%d") == "2026-11-09"
    assert provisional.iloc[-1]["close"] == 50.76


def test_same_day_bar_after_exchange_close_is_a_completed_session():
    idx = pd.to_datetime(["2026-11-06", "2026-11-09"])
    daily = pd.DataFrame({"close": [42.30, 50.76]}, index=idx)
    closed, provisional = split_incomplete_daily_bar(
        daily, retrieved_at="2026-11-09T15:01:00+08:00"
    )
    assert len(closed) == 2
    assert provisional.empty


def test_company_snapshot_excludes_intraday_bar_from_metrics_and_exports_it(monkeypatch, tmp_path):
    import quantclock.company_analysis as analysis

    idx = pd.bdate_range("2025-01-01", "2026-11-09")
    daily = sample_daily(len(idx))
    daily.index = idx
    daily.iloc[-1, daily.columns.get_loc("close")] = 50.76
    daily.iloc[-1, daily.columns.get_loc("high")] = 50.76
    daily.attrs.update({"retrieved_at": "2026-11-09T10:21:39+08:00", "name": "信维通信"})
    benchmark = pd.DataFrame({"close": np.linspace(100, 180, len(idx))}, index=idx)
    benchmark.attrs["name"] = "创业板指"
    flow_idx = pd.bdate_range("2026-11-02", periods=5)
    flow = pd.DataFrame({"main_net_inflow_cny": [-1.0] * 5,
                         "main_net_inflow_ratio_pct": [-0.1] * 5}, index=flow_idx)
    announcements = pd.DataFrame(columns=["published_at", "notice_date", "title", "category", "article_code", "source_url"])
    monkeypatch.setattr(analysis, "fetch_daily_ohlcv", lambda code, start_date: daily)
    monkeypatch.setattr(analysis, "fetch_money_flow", lambda code, limit: flow)
    monkeypatch.setattr(analysis, "fetch_announcements", lambda code, limit: announcements)
    monkeypatch.setattr(analysis, "fetch_benchmark_daily", lambda code, start_date: benchmark)

    summary = build_company_snapshot("300136", include_us_clock=False)
    assert summary["metadata"]["price_end"] == "2026-11-06"
    assert summary["metadata"]["raw_price_end"] == "2026-11-09"
    assert summary["technical"]["as_of"] == "2026-11-06"
    assert summary["frames"]["benchmark"].index[-1].strftime("%Y-%m-%d") == "2026-11-06"
    assert summary["intraday_reference"]["price_qfq"] == 50.76
    md = render_markdown(summary)
    page = render_html(summary)
    assert "盘中参考（未完成K线，不纳入日频分析）" in md
    assert "最近完整收盘日 2026-11-06" in md
    assert "盘中参考（未完成K线）" in page and "50.76" in page
    paths = write_snapshot(tmp_path, summary)
    raw = pd.read_csv(paths["daily_ohlcv_qfq.csv"])
    assert raw.iloc[-1]["bar_status"] == "incomplete_intraday_bar"
    assert Path(paths["intraday_reference.csv"]).exists()


def test_announcement_labels_are_event_categories_not_sentiment():
    assert classify_announcement("信维通信：2026年三季度报告") == "定期报告/业绩"
    assert classify_announcement("投资者关系活动记录表") == "投资者沟通"
    assert classify_announcement("重大风险提示公告") == "风险/监管事项"


def test_policy_matching_is_keyword_retrieval_and_keeps_scope_note():
    policies = pd.DataFrame([{"title": "电子信息制造业稳增长方案", "published_at": "2025-09-04",
                              "publisher": "部委", "keywords": "电子信息;元器件",
                              "summary": "推进智能终端", "source_url": "https://example.org",
                              "scope_note": "仅行业背景"}])
    result = match_policy_documents(policies)
    assert len(result) == 1
    assert "电子信息" in result.iloc[0]["relevant_terms"]
    assert result.iloc[0]["scope_note"] == "仅行业背景"


def test_market_model_runs_on_aligned_daily_returns():
    daily = sample_daily(160)
    idx = daily.index
    bench_close = pd.Series(100 * (1 + .0004 + .002 * np.sin(np.arange(len(idx)))).cumprod(), index=idx)
    benchmark = pd.DataFrame({"close": bench_close}, index=idx)
    benchmark.attrs["name"] = "test index"
    result = market_model(daily, benchmark)
    assert result["available"]
    assert result["observations"] == 120
    assert "不是未来alpha" in result["note"]


def test_report_has_required_disclaimers_and_uses_no_single_total_score():
    daily = sample_daily(100)
    idx = daily.index
    frame = {"daily": daily, "flow": pd.DataFrame(), "announcements": pd.DataFrame(),
             "benchmark": pd.DataFrame(), "policies": pd.DataFrame()}
    summary = {
        "metadata": {"code": "300136", "name": "信维通信", "retrieved_at": "2026-01-01", "price_start": "2025-01-01",
                     "price_end": "2025-05-01", "price_observations": 100, "flow_start": None, "flow_end": None,
                     "flow_observations": 0, "announcement_count": 0},
        "technical": technical_analysis(daily), "flow": flow_analysis(pd.DataFrame()),
        "announcements": {"count": 0, "latest_title": None, "latest_published_at": None, "categories": {}},
        "sentiment": {"classification": "量价信号混合", "scope_note": "不含社交媒体数据。", "return_5d": 0.01,
                      "amount_5d_vs_prior20d_median": 1.1, "turnover_5d_vs_prior20d_median": 1.0,
                      "relative_return_20d_vs_benchmark": 0.0},
        "policies": [], "market_model": {"available": False, "note": "不可用"},
        "merrill_clock": {"available": False, "note": "未连接"}, "frames": frame,
    }
    md = render_markdown(summary)
    page = render_html(summary)
    assert "不构成投资建议" in md
    assert "不构成投资建议" in page
    assert "五个维度分开呈现" in md
    assert "综合评分" not in md
    assert "viewport" in page
