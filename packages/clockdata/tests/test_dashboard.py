"""Tests for the self-contained HTML result dashboard."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clockdata.dashboard import _fmt_pct, _tone, build_dashboard
from clockdata.sources import SOURCES
from clockdata.store import Store


@pytest.fixture
def outputs(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    rng = np.random.default_rng(7)
    net = rng.normal(0.004, 0.02, len(dates))
    frame = pd.DataFrame({
        "date": dates,
        "growth_accel": rng.normal(0, 1, len(dates)),
        "inflation_accel": rng.normal(0, 1, len(dates)),
        "p_recovery": 0.25, "p_overheat": 0.25,
        "p_stagflation": 0.25, "p_reflation": 0.25,
        "w_SPY": 0.30, "w_IEF": 0.25, "w_GLD": 0.20,
        "w_DBC": 0.15, "w___cash_zero__": 0.10,
        "gross_return": net + 0.001,
        "turnover": 0.1,
        "cost_return": 0.0001,
        "net_return": net,
        "equity": np.cumprod(1 + net),
        "evaluated": 1,
    })
    frame.to_csv(tmp_path / "clock_backtest.csv", index=False)

    monthly = pd.DataFrame({
        "date": dates,
        "SPY": net + 0.002, "IEF": -net * 0.2,
        "GLD": net * 0.5, "DBC": net * 0.8,
    })
    monthly.to_csv(tmp_path / "monthly_asset_returns.csv", index=False)

    metrics = {
        "cagr": 0.05, "sharpe_zero_rf": 0.7, "max_drawdown": -0.12,
        "annualized_volatility": 0.08, "total_return": 0.18,
        "observations": 30, "sample_start": "2020-01-31", "sample_end": "2022-06-30",
        "benchmark_cagr": 0.04, "benchmark_sharpe_zero_rf": 0.6,
        "benchmark_max_drawdown": -0.18,
        "data_source": "local", "data_warning": "unit-test fixture",
        "weights": "fixture tilts", "cash_sleeve": "zero accrual",
        "output_flag": "fixture flag", "benchmark": "fixture benchmark",
        "cost_bps_per_one_way_turnover": 10.0,
        "months_dropped_for_missing_core_prices": 3,
        "warmup_months_excluded": 2,
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    attribution = {
        "nobs": 30, "sample_start": "2020-01-31", "sample_end": "2022-06-30",
        "rsquared": 0.42, "maxlags": 3, "alignment_note": "exact date match",
        "coefficients": {
            "const": {"coef": 0.0001, "t": 0.2, "p": 0.84},
            "Mkt-RF": {"coef": 0.38, "t": 5.1, "p": 0.0001},
        },
    }
    (tmp_path / "factor_attribution.json").write_text(json.dumps(attribution), encoding="utf-8")
    return tmp_path


def test_dashboard_is_self_contained_and_complete(outputs: Path) -> None:
    target = build_dashboard(outputs, outputs / "no-such.db")
    text = target.read_text(encoding="utf-8")
    assert target.name == "dashboard.html"
    assert text.count("<svg") == 6                 # equity / drawdown / phases / weights / signal layer / cost drag
    assert text.count("<table>") == 3              # performance / factors / signal-layer decomposition
    assert "本地数据库暂不可读" in text            # missing db degrades gracefully
    assert "Mkt-RF" in text and "0.4200" in text  # attribution rendered
    assert "信号层" in text and "累计成本拖累" in text
    assert "尚未生成横截面单因子检验结果" in text  # optional report degrades gracefully
    assert "尚未生成相位校准诊断" in text
    assert "尚未生成 run_manifest.json" in text
    assert "怎么启动" in text and "clock menu" in text
    assert "http://" not in text and "https://" not in text  # no CDN or external assets
    assert "<script" not in text
    assert "<html" in text and "</html>" in text


def test_dashboard_renders_catalogue_when_database_is_readable(outputs: Path) -> None:
    db = outputs / "finance.db"
    frame = pd.DataFrame({"Mkt-RF": [0.01, -0.02]},
                         index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    Store(db).init_schema(sources=SOURCES).upsert_frame(
        "ff5_factors", "fama_french", "factors", frame, frequency="月")
    text = build_dashboard(outputs, db).read_text(encoding="utf-8")
    assert text.count("<table>") == 4
    assert "ff5_factors" in text and "fama_french" in text
    assert "本地数据库暂不可读" not in text


def test_dashboard_custom_target_and_missing_attribution(outputs: Path) -> None:
    (outputs / "factor_attribution.json").unlink()
    target = outputs / "reports" / "custom.html"
    result = build_dashboard(outputs, outputs / "no-such.db", target=target)
    text = result.read_text(encoding="utf-8")
    assert result == target
    assert "尚未生成因子归因结果" in text
    assert "<svg" in text


def test_dashboard_requires_saved_outputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="clock_backtest.csv"):
        build_dashboard(tmp_path, tmp_path / "no-such.db")


def test_number_formatting_follows_chinese_market_convention() -> None:
    assert _fmt_pct(0.0599) == "+5.99%"
    assert _fmt_pct(-0.1901) == "-19.01%"
    assert _tone(0.01) == "up"
    assert _tone(-0.01) == "down"
    assert _fmt_pct(float("nan")) == "—"
