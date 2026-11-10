"""Tests for the local backend data service (REST API)."""
from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from clockdata.server import DataService, _Handler


@pytest.fixture(scope="module")
def service_url(tmp_path_factory):
    """Start the API on an ephemeral port for the duration of the module."""
    db = tmp_path_factory.mktemp("db") / "finance.db"
    state = DataService(db_path=db)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.state = state          # type: ignore[attr-defined]
    httpd.verbose = False        # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def test_index_lists_endpoints(service_url):
    payload = _get_json(f"{service_url}/")
    assert "endpoints" in payload
    assert any("/api/health" in e for e in payload["endpoints"])


def test_health_reports_database_state(service_url):
    payload = _get_json(f"{service_url}/api/health")
    assert payload["status"] in ("ok", "degraded")
    assert "database" in payload
    assert "generated_at" in payload


def test_sources_endpoint_returns_catalogue(service_url):
    payload = _get_json(f"{service_url}/api/sources")
    assert payload["sources"]
    row = payload["sources"][0]
    assert "source_id" in row and "capabilities" in row


def test_capabilities_endpoint_returns_routing(service_url):
    payload = _get_json(f"{service_url}/api/capabilities")
    assert "a_share" in payload["routing"]
    assert payload["routing"]["a_share"]


def test_datasets_endpoint_returns_list(service_url):
    payload = _get_json(f"{service_url}/api/datasets")
    assert "datasets" in payload


def test_instruments_endpoint_documents_formats(service_url):
    payload = _get_json(f"{service_url}/api/instruments")
    assert payload["builtins"]
    assert payload["accepted_formats"]


def test_instrument_endpoint_resolves_symbol(service_url):
    payload = _get_json(f"{service_url}/api/instrument/300136")
    inst = payload["instrument"]
    assert inst["symbol"] == "300136"
    assert inst["market"] == "a_share"


def test_unknown_endpoint_returns_404(service_url):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{service_url}/api/nope", timeout=10)
    assert exc.value.code == 404


def test_constellation_json_is_valid(service_url):
    payload = _get_json(f"{service_url}/api/constellation?symbol=300136")
    assert "graph" in payload
    assert "caveats" in payload
    assert isinstance(payload["graph"]["nodes"], list)


def test_constellation_html_is_a_document(service_url):
    html = _get_text(f"{service_url}/api/constellation.html?symbol=300136")
    assert html.startswith("<!DOCTYPE html>")
    assert "constellation" in html


def test_factors_endpoint_handles_missing_report(service_url):
    payload = _get_json(f"{service_url}/api/factors")
    assert "available" in payload


def test_forecast_endpoint_returns_payload(service_url):
    payload = _get_json(f"{service_url}/api/forecast?symbol=300136")
    assert "forecasts" in payload
    assert "clouds" in payload
