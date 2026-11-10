"""Tests for the constellation graph model, clouds, forecasts and renderer."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from starmap import cloud as cloud_mod
from starmap import forecast as fc_mod
from starmap import graph as graph_mod
from starmap.viz import render_html, write_html


# --------------------------------------------------------------------------- #
# graph model
# --------------------------------------------------------------------------- #
def test_node_size_scales_with_magnitude():
    small = graph_mod.Node(id="a", kind="factor", label="a", value=0.01)
    large = graph_mod.Node(id="b", kind="factor", label="b", value=0.9)
    assert large.size > small.size
    assert graph_mod.Node(id="c", kind="factor", label="c", value=0.0).size == 4.0


def test_node_colour_by_kind():
    assert graph_mod.Node(id="a", kind="macro_phase", label="x").color == "#5ad7ff"
    assert graph_mod.Node(id="b", kind="instrument", label="x").color == "#ffd166"


def test_constellation_validation_detects_dangling_link():
    c = graph_mod.Constellation()
    c.add_node(graph_mod.Node(id="phase:recovery", kind="macro_phase", label="复苏"))
    c.add_link(graph_mod.Link(source="phase:recovery", target="asset:SPY",
                              kind="phase_alloc"))
    issues = c.validate()
    assert any("asset:SPY" in i for i in issues)


def test_constellation_validation_detects_unknown_kinds():
    c = graph_mod.Constellation()
    c.add_node(graph_mod.Node(id="x", kind="not_a_kind", label="x"))
    c.add_link(graph_mod.Link(source="x", target="x", kind="not_a_link"))
    issues = c.validate()
    assert len(issues) >= 2


def test_valid_constellation_has_no_issues():
    c = graph_mod.Constellation()
    c.add_node(graph_mod.Node(id="phase:recovery", kind="macro_phase", label="复苏"))
    c.add_node(graph_mod.Node(id="asset:SPY", kind="asset", label="SPY"))
    c.add_link(graph_mod.Link(source="phase:recovery", target="asset:SPY",
                              kind="phase_alloc", weight=0.3))
    assert c.validate() == []


def test_build_phase_nodes_sizes_by_probability():
    nodes = graph_mod.build_phase_nodes({"recovery": 0.6, "overheat": 0.4},
                                        {"recovery": "复苏"})
    assert len(nodes) == 2
    assert nodes[0].label == "复苏"
    assert nodes[0].size > nodes[1].size


def test_build_factor_links_point_at_instrument():
    links = graph_mod.build_factor_links({"mom_1m": 0.05}, "300136")
    assert links[0].source == "factor:mom_1m"
    assert links[0].target == "instrument:300136"


def test_transition_links_skip_self_transitions():
    matrix = {"recovery": {"recovery": 0.7, "overheat": 0.3}}
    links = graph_mod.build_transition_links(matrix)
    assert len(links) == 1
    assert links[0].target == "phase:overheat"


# --------------------------------------------------------------------------- #
# clouds
# --------------------------------------------------------------------------- #
def test_correlation_cloud_requires_enough_samples():
    out = cloud_mod.correlation_cloud(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 3.0]}),
                                      "a", "b")
    assert out["available"] is False
    assert "note" in out


def test_correlation_cloud_produces_normalised_grid():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    out = cloud_mod.correlation_cloud(frame, "a", "b", width=24, height=16)
    assert out["available"] is True
    assert len(out["density"]) == 16
    assert len(out["density"][0]) == 24
    values = [v for row in out["density"] for v in row]
    assert max(values) <= 1.0
    assert min(values) >= 0.0


def test_correlation_cloud_missing_column_is_reported():
    out = cloud_mod.correlation_cloud(pd.DataFrame({"a": range(50)}), "a", "missing")
    assert out["available"] is False


def test_phase_transition_matrix_rows_sum_to_one():
    series = pd.Series(["a", "b", "a", "a", "b", "b"],
                        index=pd.date_range("2024-01-31", periods=6, freq="ME"))
    matrix = cloud_mod.phase_transition_matrix(series)
    assert matrix
    for row in matrix.values():
        assert abs(sum(row.values()) - 1.0) < 1e-6


def test_regime_density_cloud_marks_current_position():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "growth_accel": rng.normal(size=120).cumsum() * 0.01,
        "inflation_accel": rng.normal(size=120).cumsum() * 0.01,
    }, index=pd.date_range("2010-01-31", periods=120, freq="ME"))
    out = cloud_mod.regime_density_cloud(frame)
    assert out["available"] is True
    assert "growth" in out["current"] and "inflation" in out["current"]
    assert len(out["trajectory"]) == 120


def test_soften_edges_removes_hard_boundary():
    grid = np.ones((20, 20))
    softened = cloud_mod._soften_edges(grid)
    assert softened[0, 0] < 1.0
    assert softened[10, 10] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# forecasts
# --------------------------------------------------------------------------- #
def test_phase_path_probabilities_sum_to_one():
    matrix = {"recovery": {"recovery": 0.8, "overheat": 0.2},
              "overheat": {"overheat": 0.6, "recovery": 0.4}}
    out = fc_mod.project_phase_path("recovery", matrix, horizon=4)
    assert out["available"] is True
    assert len(out["steps"]) == 4
    for step in out["steps"]:
        assert abs(sum(step["distribution"].values()) - 1.0) < 1e-6
    assert out["steps"][0]["modal_phase"] == "recovery"
    assert out["caveats"], "a projection must carry caveats"


def test_phase_path_without_matrix_is_reported():
    out = fc_mod.project_phase_path("recovery", {}, horizon=3)
    assert out["available"] is False
    assert out["steps"] == []


def test_phase_path_entropy_rises_over_horizon():
    matrix = {"recovery": {"recovery": 0.5, "overheat": 0.5},
              "overheat": {"recovery": 0.5, "overheat": 0.5}}
    out = fc_mod.project_phase_path("recovery", matrix, horizon=5)
    entropies = [s["entropy"] for s in out["steps"]]
    assert entropies[-1] >= entropies[0]


def test_factor_expectations_band_widens():
    stats = {"mom_1m": {"ic_mean": 0.06, "icir": 0.12, "n_periods": 200}}
    out = fc_mod.project_factor_expectations(stats, horizon=4)
    factor = out["factors"][0]
    widths = [p["upper"] - p["lower"] for p in factor["path"]]
    assert widths[-1] > widths[0]
    assert factor["path"][0]["expected_ic"] < 0.06


def test_price_path_requires_history():
    out = fc_mod.project_price_paths(pd.Series(dtype=float), horizon=6)
    assert out["available"] is False


def test_price_path_band_widens_with_horizon():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2018-01-31", periods=120, freq="ME")
    series = pd.Series(rng.normal(0.005, 0.04, 120), index=idx)
    out = fc_mod.project_price_paths(series, horizon=12, n_paths=200)
    assert out["available"] is True
    widths = [s["upper"] - s["lower"] for s in out["steps"]]
    assert widths[-1] > widths[0]
    assert all(0.0 <= s["prob_up"] <= 1.0 for s in out["steps"])


def test_price_path_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2018-01-31", periods=60, freq="ME")
    series = pd.Series(rng.normal(0.0, 0.03, 60), index=idx)
    a = fc_mod.project_price_paths(series, horizon=6, seed=42)
    b = fc_mod.project_price_paths(series, horizon=6, seed=42)
    assert a["steps"] == b["steps"]


def test_signal_confidence_measures_direction():
    idx = pd.date_range("2024-01-31", periods=10, freq="ME")
    pred = pd.Series([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], index=idx, dtype=float)
    real = pd.Series([2, -2, 3, -1, 1, -2, 4, -3, 2, -1], index=idx, dtype=float)
    out = fc_mod.signal_confidence(pred, real)
    assert out["available"] is True
    assert out["direction_accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# renderer
# --------------------------------------------------------------------------- #
def _sample_payload() -> dict:
    c = graph_mod.Constellation()
    c.add_node(graph_mod.Node(id="phase:recovery", kind="macro_phase", label="复苏",
                               value=0.6, metrics={"probability": 0.6}))
    c.add_node(graph_mod.Node(id="asset:SPY", kind="asset", label="SPY", value=0.3))
    c.add_node(graph_mod.Node(id="factor:mom_1m", kind="factor", label="mom_1m", value=0.06))
    c.add_node(graph_mod.Node(id="instrument:300136", kind="instrument", label="信维通信", value=0.05))
    c.add_node(graph_mod.Node(id="source:akshare", kind="source", label="akshare", value=0.4))
    c.add_link(graph_mod.Link(source="phase:recovery", target="asset:SPY",
                              kind="phase_alloc", weight=0.3))
    c.add_link(graph_mod.Link(source="factor:mom_1m", target="instrument:300136",
                              kind="factor_load", weight=0.06))
    return {
        "meta": {"title": "测试星图", "subtitle": "副标题",
                 "generated_at": "2026-11-10T00:00:00+08:00"},
        "badges": [("市场 a_share", "on")],
        "graph": {"nodes": [n.to_dict() for n in c.nodes],
                  "links": [l.to_dict() for l in c.links]},
        "phases": [{"label": "复苏", "probability": 0.6, "weight": 0.0}],
        "allocations": [{"label": "SPY", "weight": 0.3}],
        "factors": [{"name": "mom_1m", "ic_mean": 0.06, "icir": 0.12,
                     "positive_ratio": 0.54, "n_periods": 248}],
        "instrument": {"symbol": "300136", "label": "信维通信",
                       "metrics": [["最新收盘", "50.83"]]},
        "sources": [{"source_id": "akshare", "capabilities": "a_share",
                     "available": True, "status_detail": "可用", "note": "免费"}],
        "clouds": {},
        "forecasts": {},
        "caveats": ["这是测试说明"],
        "legend": [{"label": "相位", "color": "#5ad7ff"}],
    }


def test_render_html_is_self_contained():
    html = render_html(_sample_payload())
    assert html.startswith("<!DOCTYPE html>")
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "cdn." not in html
    assert "src=\"http" not in html


def test_render_html_embeds_payload_as_json():
    html = render_html(_sample_payload())
    start = html.index('<script id="payload" type="application/json">')
    end = html.index("</script>", start)
    blob = html[start:end].split(">", 1)[1]
    payload = json.loads(blob)
    assert payload["graph"]["nodes"]
    assert payload["meta"]["title"] == "测试星图"


def test_render_html_escapes_payload_breaking_characters():
    payload = _sample_payload()
    payload["meta"]["title"] = "</script><script>alert(1)</script>"
    html = render_html(payload)
    start = html.index('<script id="payload" type="application/json">')
    end = html.index("</script>", start)
    blob = html[start:end].split(">", 1)[1]
    json.loads(blob)  # must still parse: the payload is JSON-encoded


def test_write_html_creates_file(tmp_path):
    target = tmp_path / "out" / "star.html"
    path = write_html(_sample_payload(), target)
    assert path.exists()
    assert path.stat().st_size > 1000


def test_render_html_handles_empty_graph():
    payload = _sample_payload()
    payload["graph"] = {"nodes": [], "links": []}
    html = render_html(payload)
    assert "<!DOCTYPE html>" in html
