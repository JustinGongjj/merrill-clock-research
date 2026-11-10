import numpy as np
import pandas as pd
import pytest

from clockdata.sources import DATASETS, FETCHERS, SOURCES, check_connectivity, fetch_dataset
from clockdata.store import Store


def _panel() -> pd.DataFrame:
    idx = pd.date_range("2024-01-31", periods=4, freq="ME")
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.1, np.nan, 0.3, 0.4]}, index=idx)


def test_roundtrip_preserves_dates_columns_and_values(tmp_path):
    store = Store(tmp_path / "test.db").init_schema(sources=SOURCES)
    frame = _panel()
    assert store.upsert_frame("demo", "local_csv", "macro", frame) == 7  # one NaN skipped
    loaded = store.load_frame("demo")
    assert list(loaded.columns) == ["a", "b"]
    assert loaded.index.equals(frame.index)
    assert loaded.loc["2024-01-31", "a"] == 1.0
    assert pd.isna(loaded.loc["2024-02-29", "b"])


def test_upsert_overwrites_values_and_keeps_earlier_history(tmp_path):
    store = Store(tmp_path / "test.db").init_schema(sources=SOURCES)
    store.upsert_frame("demo", "local_csv", "macro", _panel())
    later = pd.DataFrame({"a": [9.0]}, index=pd.DatetimeIndex(["2024-01-31"]))
    store.upsert_frame("demo", "local_csv", "macro", later)
    loaded = store.load_frame("demo")
    assert loaded.loc["2024-01-31", "a"] == 9.0
    assert len(loaded) == 4  # earlier rows were not deleted


def test_load_frame_date_and_series_filters(tmp_path):
    store = Store(tmp_path / "test.db").init_schema(sources=SOURCES)
    store.upsert_frame("demo", "local_csv", "macro", _panel())
    filtered = store.load_frame("demo", start="2024-02-01", end="2024-03-31", series=["a"])
    assert list(filtered.columns) == ["a"]
    assert len(filtered) == 2


def test_dataset_meta_and_status_coverage(tmp_path):
    store = Store(tmp_path / "test.db").init_schema(sources=SOURCES)
    store.upsert_frame("demo", "fred", "macro", _panel(), unit="index", frequency="月")
    meta = store.dataset_meta("demo")
    assert meta["first_date"] == "2024-01-31" and meta["last_date"] == "2024-04-30"
    assert meta["n_obs"] == 7 and meta["unit"] == "index"
    assert set(store.datasets()["dataset_id"]) == {"demo"}
    assert "fred" in set(store.sources()["source_id"])


def test_sync_log_and_readonly_sql_guard(tmp_path):
    store = Store(tmp_path / "test.db").init_schema(sources=SOURCES)
    store.log_sync("demo", "fred", "2026-01-01T00:00:00+00:00", "ok", 7, "test")
    assert store.sync_history().iloc[0]["rows_written"] == 7
    result = store.query("SELECT COUNT(*) AS n FROM sync_log")
    assert int(result.iloc[0]["n"]) == 1
    with pytest.raises(ValueError):
        store.query("DELETE FROM sync_log")


def test_empty_frame_rejected(tmp_path):
    store = Store(tmp_path / "test.db").init_schema(sources=SOURCES)
    with pytest.raises(ValueError):
        store.upsert_frame("demo", "local_csv", "macro", pd.DataFrame())


def test_source_catalogue_is_consistent():
    ids = {s["source_id"] for s in SOURCES}
    assert ids >= {"fred", "yahoo", "fama_french", "wind_mcp", "local_csv"}
    assert set(FETCHERS) == set(DATASETS)
    for name, spec in DATASETS.items():
        assert spec["source_id"] in ids, name
    with pytest.raises(KeyError):
        fetch_dataset("does_not_exist")


def test_yahoo_sync_refuses_to_hang_when_probe_fails(monkeypatch):
    import quantclock.data as qdata
    from clockdata import sources

    monkeypatch.setattr(qdata, "yahoo_healthy", lambda: False)
    monkeypatch.setattr(qdata, "fetch_market_prices",
                        lambda *a, **k: pytest.fail("should not start a bulk download"))
    with pytest.raises(RuntimeError, match="health probe failed"):
        sources._fetch_yahoo("2000-01-01")


def test_connectivity_probe_reports_every_registered_python_source(monkeypatch):
    monkeypatch.setattr("clockdata.sources.requests.get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr("clockdata.sources.requests.head", lambda *a, **k: _FakeResponse())
    probes = check_connectivity(timeout=1)
    # akshare and windpy probes are registered alongside the original five.
    assert {p["source"] for p in probes} == {
        "fred", "yahoo", "fama_french", "wind_mcp", "local_csv", "akshare", "windpy",
    }
    assert all("detail" in p and "status" in p for p in probes)


class _FakeResponse:
    ok = True
    status_code = 200
    content = b"date,INDPRO\n2025-01-01,100.0\n"
    text = "date,INDPRO\n2025-01-01,100.0\n"
