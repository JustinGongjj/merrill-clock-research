"""Keep pytest's temporary files inside the project tree.

WorkBuddy's Python shim intercepts mkdir under /private/var/folders, which
breaks pytest's default tmp_path factory. Project-local fixtures avoid the
sandbox path entirely so `pytest -q` works without extra flags.

Both ``tmp_path`` and ``tmp_path_factory`` are overridden: the latter backs
the former in stock pytest, so overriding only one still leaves the base
temp directory outside the project tree.
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
    path = Path(tempfile.mkdtemp(prefix="clockdata-", dir=_TMP_ROOT))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def tmp_path_factory():
    """A ``tmp_path_factory`` rooted inside the project tree."""
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="clockdata-factory-", dir=_TMP_ROOT))
    counter = {"n": 0}

    class _Factory:
        def mktemp(self, name: str, numbered: bool = True) -> Path:
            counter["n"] += 1
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
            path = root / f"{safe}-{counter['n']}"
            path.mkdir(parents=True, exist_ok=True)
            return path

        def getbasetemp(self) -> Path:
            return root

    try:
        yield _Factory()
    finally:
        shutil.rmtree(root, ignore_errors=True)
