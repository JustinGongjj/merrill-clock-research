"""Tests for the data-source registry and capability routing."""
from __future__ import annotations

import pandas as pd
import pytest

from clockdata import registry as reg


def test_catalogue_ids_are_unique():
    ids = [s.source_id for s in reg.CATALOGUE]
    assert len(ids) == len(set(ids))


def test_every_capability_is_served_by_some_source():
    matrix = reg.capability_matrix()
    for cap in reg.ALL_CAPABILITIES:
        assert matrix[cap], f"no source serves {cap}"


def test_unknown_capability_raises():
    with pytest.raises(KeyError):
        reg.sources_for("not_a_capability")


def test_key_gated_sources_are_unavailable_without_key(monkeypatch):
    spec = reg.BY_ID["tushare"]
    assert spec.requires_key
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    assert not spec.available()
    assert "TUSHARE_TOKEN" in spec.status_detail()


def test_key_gated_source_available_with_key(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "dummy-token-for-test")
    spec = reg.BY_ID["tushare"]
    assert spec.key_present()


def test_registry_rows_expose_capabilities_and_status():
    rows = reg.registry_rows()
    assert len(rows) == len(reg.CATALOGUE)
    for row in rows:
        assert set(row["capabilities"])
        assert "available" in row and "status_detail" in row


def test_sources_for_orders_by_tier():
    chain = reg.sources_for(reg.CAP_A_SHARE, require_available=False)
    tiers = [s.tier for s in chain]
    assert tiers == sorted(tiers), "sources must be ordered best-tier first"


def test_routed_fetch_returns_first_success():
    """The first source in tier order is the one that answers."""
    def good():
        return pd.DataFrame({"A": [1, 2, 3]})

    outcome = reg.routed_fetch(reg.CAP_A_SHARE, {"akshare": good})
    assert outcome.source_id == "akshare"
    assert not outcome.degraded
    assert outcome.attempts[-1]["outcome"] == "ok"


def test_routed_falls_back_to_next_source_and_marks_degraded():
    def bad():
        raise RuntimeError("upstream down")

    def good():
        return pd.DataFrame({"A": [1, 2, 3]})

    outcome = reg.routed_fetch(reg.CAP_A_SHARE,
                                {"baostock": bad, "akshare": good})
    assert outcome.source_id == "akshare"
    assert outcome.degraded, "a fallback must be reported as degraded"
    assert outcome.attempts[0]["outcome"] == "error"


def test_routed_fetch_raises_when_all_fail():
    def bad():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        reg.routed_fetch(reg.CAP_A_SHARE, {"akshare": bad})


def test_routed_fetch_skips_empty_frames():
    """An empty answer is recorded as 'empty' and the chain moves on."""
    def empty():
        return pd.DataFrame()

    def good():
        return pd.DataFrame({"A": [1, 2]})

    # baostock is tier 1 for a_share, so it is attempted before akshare.
    outcome = reg.routed_fetch(reg.CAP_A_SHARE, {"baostock": empty, "akshare": good})
    assert outcome.source_id == "akshare"
    assert any(a["outcome"] == "empty" for a in outcome.attempts)
    assert outcome.degraded, "falling back after an empty frame is a degradation"


def test_local_db_is_registered_as_offline_fallback():
    spec = reg.BY_ID["local_db"]
    assert reg.CAP_A_SHARE in spec.capabilities
    assert reg.CAP_US_EQUITY in spec.capabilities
    assert spec.transport == "file"


def test_probe_all_returns_one_row_per_source():
    rows = reg.probe_all(timeout=1)
    assert len(rows) == len(reg.CATALOGUE)
    for row in rows:
        assert "source" in row and "status" in row and "detail" in row


def test_probe_does_not_leave_a_gil_spinning_thread():
    """A probe must not starve the rest of the process.

    Regression test for two defects that made the whole backend crawl after
    ``probe_all`` ran:

    * Baostock's cursor only advances when ``get_row_data()`` is called, so a
      loop over ``next()`` alone spins forever and holds the GIL.
    * An import performed inside the probe's worker thread could be abandoned
      mid-import, keeping that module's import lock and blocking every later
      ``import_module`` call.

    A background thread that spins without releasing the GIL must therefore not
    exist once the probe has returned.
    """
    import threading
    import time

    before = threading.active_count()
    reg.probe_all(timeout=1)
    time.sleep(0.3)
    assert threading.active_count() <= before + 1, (
        "probe_all left a background thread behind; it may be spinning and "
        "holding the GIL"
    )

    # A tight Python loop in another thread starves the main thread.
    # Guard the whole process against that: the GIL must still be handed back
    # often enough for this thread to make progress.
    spin = threading.Event()
    spinner = threading.Thread(
        target=lambda: spin.wait(5), daemon=True)   # cooperative, releases GIL
    spinner.start()
    try:
        start = time.perf_counter()
        for _ in range(200_000):
            pass
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"main thread was starved by a background thread ({elapsed:.1f}s)"
    finally:
        spin.set()


def test_guarded_returns_timeout_without_hanging():
    """A probe that never returns must be reported as a timeout."""
    import time

    def never_returns(deadline):
        while not deadline.expired:
            time.sleep(0.05)
        return True, "should not get here"

    started = time.perf_counter()
    ok, detail = reg._guarded(never_returns, timeout=0.3)
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, "the guard did not enforce its deadline"
    assert not ok or "should not get here" not in detail


def test_import_status_is_cached_and_never_reimports():
    """Availability checks must not re-enter the import system each time."""
    calls = []
    real_import = reg.importlib.import_module

    def counting_import(name, *args, **kwargs):
        calls.append(name)
        return real_import(name, *args, **kwargs)

    reg.importlib.import_module = counting_import
    try:
        reg._IMPORT_CACHE.pop("json", None)
        reg._import_status("json")
        reg._import_status("json")
        reg._import_status("json")
    finally:
        reg.importlib.import_module = real_import
    assert calls.count("json") == 1, "the import cache did not prevent a re-import"


def test_deadline_expires_and_reports_remaining():
    import time

    deadline = reg._Deadline(0.05)
    assert not deadline.expired
    assert deadline.remaining() <= 0.05
    time.sleep(0.08)
    assert deadline.expired
