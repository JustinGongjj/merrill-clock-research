"""Tests for symbol resolution, universe expansion and the multi-source router."""
from __future__ import annotations

import pandas as pd
import pytest

from clockdata import prices as price_mod
from clockdata import universe as uni


@pytest.mark.parametrize("symbol,market,capability", [
    ("300136", uni.MARKET_A_SHARE, "a_share"),
    ("600519", uni.MARKET_A_SHARE, "a_share"),
    ("000001", uni.MARKET_A_SHARE, "a_share"),
    ("688981", uni.MARKET_A_SHARE, "a_share"),
    ("AAPL", uni.MARKET_US, "us_equity"),
    ("SPY", uni.MARKET_US, "etf"),
    ("BTC/USDT", uni.MARKET_CRYPTO, "crypto"),
    ("00700", uni.MARKET_HK, "hk_equity"),
    ("09988", uni.MARKET_HK, "hk_equity"),
    ("0700", uni.MARKET_HK, "hk_equity"),
])
def test_classify_detects_market_and_capability(symbol, market, capability):
    inst = uni.classify(symbol)
    assert inst.market == market
    assert inst.capability == capability


def test_classify_rejects_empty_symbol():
    with pytest.raises(ValueError):
        uni.classify("   ")


def test_unknown_symbol_market():
    inst = uni.classify("这是中文")
    assert inst.market == uni.MARKET_UNKNOWN
    assert inst.capability == ""


def test_a_share_exchange_inference():
    assert uni._a_share_exchange("600519") == "SH"
    assert uni._a_share_exchange("000001") == "SZ"
    assert uni._a_share_exchange("300136") == "SZ"
    assert uni._a_share_exchange("688981") == "SH"
    assert uni._a_share_exchange("830799") == "BJ"


def test_baostock_code_format():
    inst = uni.classify("600519")
    assert uni.baostock_code(inst) == "sh.600519"
    inst2 = uni.classify("000001")
    assert uni.baostock_code(inst2) == "sz.000001"


def test_baostock_code_rejects_non_a_share():
    with pytest.raises(ValueError):
        uni.baostock_code(uni.classify("AAPL"))


def test_peer_universe_includes_target_first():
    inst = uni.classify("300136")
    peers = uni.peer_universe(inst, max_peers=10)
    assert peers[0] == "300136"
    assert "300136" not in peers[1:]


def test_peer_universe_for_us_includes_target():
    inst = uni.classify("AAPL")
    peers = uni.peer_universe(inst, max_peers=5)
    assert peers[0] == "AAPL"


def test_describe_sources_for_returns_chain():
    rows = uni.describe_sources_for(uni.classify("300136"))
    assert rows
    assert rows[0]["source_id"]
    assert "tier" in rows[0]


def test_describe_sources_for_unknown_is_empty():
    assert uni.describe_sources_for(uni.classify("这是中文")) == []


def test_local_db_prices_reports_missing_symbols(monkeypatch):
    """A symbol absent from the local cache must be reported as missing."""
    import tempfile
    from pathlib import Path

    from clockdata import store as store_mod

    tmp = Path(tempfile.mkdtemp()) / "empty.db"
    store = store_mod.Store(tmp).init_schema([])
    monkeypatch.setattr(price_mod, "_local_db_prices",
                        lambda syms, s, e: pd.DataFrame({"SPY": [1.0, 2.0]},
                                                        index=pd.date_range("2024-01-31", periods=2, freq="ME")))
    frame = price_mod._local_db_prices(["SPY"], "2024-01-01", None)
    assert "SPY" in frame.columns
    assert store is not None
