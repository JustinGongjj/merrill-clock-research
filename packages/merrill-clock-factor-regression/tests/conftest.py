"""Keep pytest's temporary files inside the project tree.

WorkBuddy's Python shim intercepts mkdir under /private/var/folders, which
breaks pytest's default tmp_path factory. A project-local fixture avoids the
sandbox path entirely so `pytest -q` works without extra flags.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp"


@pytest.fixture
def tmp_path():
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="factorlab-", dir=_TMP_ROOT))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
