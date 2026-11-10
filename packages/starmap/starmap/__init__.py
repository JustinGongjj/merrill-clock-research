"""Sci-fi constellation visualisation for the Merrill-clock research stack.

The package turns measured research output -- macro phase probabilities,
cross-sectional factor statistics, instrument snapshots and data-source health
-- into a self-contained HTML star map. It is instrument-agnostic: the same
code path renders any symbol.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .assemble import build_payload
from .cloud import correlation_cloud, regime_density_cloud
from .forecast import (
    project_factor_expectations,
    project_phase_path,
    project_price_paths,
)
from .graph import Constellation, Link, Node
from .viz import render_html, write_html

__all__ = [
    "build_payload",
    "correlation_cloud",
    "regime_density_cloud",
    "project_factor_expectations",
    "project_phase_path",
    "project_price_paths",
    "Constellation",
    "Link",
    "Node",
    "render_html",
    "write_html",
    "__version__",
]
