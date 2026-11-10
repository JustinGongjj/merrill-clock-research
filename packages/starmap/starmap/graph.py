"""Constellation graph model: nodes, links and the semantic layer behind them.

The visualisation is a *view* of real measurements. Every node carries the
evidence that produced it, and every link carries a typed relationship with a
weight, so a reader can always trace a glowing node back to a number in a
report. Nothing here invents data -- the graph is assembled from the local
database, the factor lab output and the company snapshot.

Node kinds
----------
``macro_phase``   a Merrill-clock phase, sized by its probability
``asset``         a tradable asset in the strategy's asset map
``factor``       a cross-sectional factor, sized by |IC|
``instrument``   the subject under study (any ticker)
``source``       a data source actually used or available
``horizon``      a forward-looking projection point

Link kinds
----------
``phase_alloc``   phase -> asset, weight = target allocation
``factor_load``   factor -> instrument, weight = factor value / IC contribution
``data_lineage``  source -> node it fed
``phase_transition`` phase -> phase, weight = transition probability
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

NODE_KINDS = ("macro_phase", "asset", "factor", "instrument", "source", "horizon")
LINK_KINDS = ("phase_alloc", "factor_load", "data_lineage", "phase_transition")

#: Visual palette per node kind (hex, used by the HTML renderer).
KIND_COLORS = {
    "macro_phase": "#5ad7ff",
    "asset": "#9d7bff",
    "factor": "#ff6bd6",
    "instrument": "#ffd166",
    "source": "#5b8c5a",
    "horizon": "#ff8a5c",
}


@dataclass
class Node:
    """One point in the constellation."""

    id: str
    kind: str
    label: str
    value: float = 0.0            # signed magnitude that drives size
    detail: str = ""              # short human-readable description
    metrics: dict[str, Any] = field(default_factory=dict)
    x: float = 0.0                # layout coordinates (filled by the renderer)
    y: float = 0.0
    z: float = 0.0                # depth hint for perspective

    @property
    def size(self) -> float:
        """Node radius in layout units. Uses sqrt so area scales with |value|."""
        magnitude = abs(self.value)
        if magnitude <= 0:
            return 4.0
        return float(np.clip(6.0 + 26.0 * math.sqrt(min(magnitude, 1.0)), 4.0, 34.0))

    @property
    def color(self) -> str:
        return KIND_COLORS.get(self.kind, "#8fa3b8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "value": _clean_float(self.value),
            "detail": self.detail,
            "metrics": self.metrics,
            "size": round(self.size, 2),
            "color": self.color,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
        }


@dataclass
class Link:
    """One relation between two nodes."""

    source: str
    target: str
    kind: str
    weight: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "weight": _clean_float(self.weight),
            "detail": self.detail,
        }


@dataclass
class Constellation:
    """A complete graph plus the provenance of everything in it."""

    nodes: list[Node] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> Node:
        self.nodes.append(node)
        return node

    def add_link(self, link: Link) -> Link:
        self.links.append(link)
        return link

    def node(self, node_id: str) -> Node | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def validate(self) -> list[str]:
        """Return structural problems (dangling links, unknown kinds, ...)."""
        issues: list[str] = []
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            issues.append("存在重复的节点 id")
        for n in self.nodes:
            if n.kind not in NODE_KINDS:
                issues.append(f"未知节点类型 {n.kind!r}（节点 {n.id}）")
        for link in self.links:
            if link.kind not in LINK_KINDS:
                issues.append(f"未知连线类型 {link.kind!r}")
            if link.source not in ids:
                issues.append(f"连线起点不存在: {link.source}")
            if link.target not in ids:
                issues.append(f"连线终点不存在: {link.target}")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [l.to_dict() for l in self.links],
            "stats": {
                "n_nodes": len(self.nodes),
                "n_links": len(self.links),
                "kinds": {k: sum(1 for n in self.nodes if n.kind == k) for k in NODE_KINDS},
            },
        }


def _clean_float(value: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return round(f, 6)


# --------------------------------------------------------------------------- #
# Builders -- each takes real measured data and produces graph elements
# --------------------------------------------------------------------------- #
def build_phase_nodes(probabilities: dict[str, float],
                      phase_cn: dict[str, str] | None = None) -> list[Node]:
    """One node per Merrill-clock phase, sized by its probability."""
    phase_cn = phase_cn or {}
    nodes = []
    for phase, prob in probabilities.items():
        value = float(prob)
        nodes.append(Node(
            id=f"phase:{phase}",
            kind="macro_phase",
            label=phase_cn.get(phase, phase),
            value=value,
            detail=f"相位概率 {value:.1%}",
            metrics={"phase": phase, "probability": _clean_float(value)},
            z=1.0,
        ))
    return nodes


def build_asset_nodes(allocations: dict[str, float]) -> list[Node]:
    """One node per asset, sized by its target weight."""
    nodes = []
    for asset, weight in allocations.items():
        value = float(weight)
        nodes.append(Node(
            id=f"asset:{asset}",
            kind="asset",
            label=asset,
            value=value,
            detail=f"目标权重 {value:.1%}",
            metrics={"asset": asset, "weight": _clean_float(value)},
            z=0.55,
        ))
    return nodes


def build_factor_nodes(factors: dict[str, dict[str, float]]) -> list[Node]:
    """One node per factor, sized by |IC| and coloured by sign via value."""
    nodes = []
    for name, stats in factors.items():
        ic = float(stats.get("ic_mean", stats.get("rank_ic", 0.0)) or 0.0)
        nodes.append(Node(
            id=f"factor:{name}",
            kind="factor",
            label=name,
            value=ic,
            detail=f"RankIC {ic:+.4f} · ICIR {float(stats.get('icir', 0.0) or 0.0):+.3f}",
            metrics={k: _clean_float(v) for k, v in stats.items()
                     if isinstance(v, (int, float))},
            z=0.8,
        ))
    return nodes


def build_instrument_node(symbol: str, label: str, snapshot: dict[str, Any]) -> Node:
    """The subject instrument, carrying whatever the snapshot measured."""
    metrics: dict[str, Any] = {}
    detail = ""
    value = 0.0
    metrics_block = snapshot.get("metrics") or {}
    if isinstance(metrics_block, dict):
        for key, val in metrics_block.items():
            if isinstance(val, (int, float)):
                metrics[key] = _clean_float(val)
    price = snapshot.get("last_price")
    if isinstance(price, (int, float)):
        value = abs(float(price)) / 1000.0
        detail = f"最新价 {float(price):.2f}"
    return Node(
        id=f"instrument:{symbol}",
        kind="instrument",
        label=label or symbol,
        value=value,
        detail=detail,
        metrics=metrics,
        z=1.3,
    )


def build_source_nodes(sources: list[dict[str, Any]], used: set[str] | None = None) -> list[Node]:
    """One node per data source; size reflects how much it contributed."""
    used = used or set()
    nodes = []
    for src in sources:
        sid = str(src.get("source_id") or src.get("source") or "")
        if not sid:
            continue
        available = bool(src.get("available", src.get("status") == "ok"))
        weight = 1.0 if sid in used else (0.45 if available else 0.15)
        nodes.append(Node(
            id=f"source:{sid}",
            kind="source",
            label=sid,
            value=weight,
            detail=str(src.get("display_name") or src.get("detail") or ""),
            metrics={
                "available": available,
                "latency": str(src.get("latency") or ""),
                "tier": _clean_float(src.get("tier", 0) or 0),
            },
            z=0.25,
        ))
    return nodes


def build_phase_links(allocations: dict[str, float],
                      phase: str) -> list[Link]:
    """phase -> asset links carrying the target allocation."""
    links = []
    for asset, weight in allocations.items():
        links.append(Link(
            source=f"phase:{phase}",
            target=f"asset:{asset}",
            kind="phase_alloc",
            weight=float(weight),
            detail=f"配置权重 {float(weight):.1%}",
        ))
    return links


def build_factor_links(factor_values: dict[str, float],
                       symbol: str) -> list[Link]:
    """factor -> instrument links carrying the measured factor value."""
    links = []
    for name, value in factor_values.items():
        links.append(Link(
            source=f"factor:{name}",
            target=f"instrument:{symbol}",
            kind="factor_load",
            weight=float(value),
            detail=f"因子暴露 {float(value):+.4f}",
        ))
    return links


def build_lineage_links(source_ids: list[str], node_ids: list[str]) -> list[Link]:
    """source -> node links marking which upstream fed which node."""
    links = []
    for sid in source_ids:
        for nid in node_ids:
            links.append(Link(
                source=f"source:{sid}",
                target=nid,
                kind="data_lineage",
                weight=0.2,
                detail="数据来源",
            ))
    return links


def build_transition_links(transition_matrix: dict[str, dict[str, float]]) -> list[Link]:
    """phase -> phase links carrying observed transition probability."""
    links = []
    for src, targets in transition_matrix.items():
        for tgt, prob in targets.items():
            if src == tgt:
                continue
            links.append(Link(
                source=f"phase:{src}",
                target=f"phase:{tgt}",
                kind="phase_transition",
                weight=float(prob),
                detail=f"历史转移概率 {float(prob):.1%}",
            ))
    return links
