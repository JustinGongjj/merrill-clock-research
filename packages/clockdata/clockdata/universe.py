"""Symbol resolution and universe expansion for any tradable instrument.

The point of this module is that nothing downstream needs to special-case a
particular ticker. A user supplies ``300136`` or ``AAPL`` or ``BTC/USDT`` and
this layer decides what it is, which data source can serve it, and which
related instruments belong in the same panel.

Resolution rules (all deterministic, no network calls):

* 6 digits -> A-share (Shanghai 60xxxx/68xxxx, Shenzhen 00xxxx/30xxxx, Beijing 8xxxxx/4xxxxx)
* 6 digits ending in SH/SZ prefix hints -> same
* 4 digits + .SS/.SZ -> ETF/index
* contains a slash -> crypto pair
* alphabetic 1-6 chars -> US equity or ETF (tried against the US route)
* anything else -> unresolved

The expansion step deliberately mixes sources: A-share peers come from the
CSI 300 constituent list (current snapshot only -- survivorship bias is
documented, never hidden), US peers from the live price panel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import registry as reg

MARKET_A_SHARE = "a_share"
MARKET_US = "us"
MARKET_HK = "hk"
MARKET_CRYPTO = "crypto"
MARKET_ETF = "etf"
MARKET_UNKNOWN = "unknown"

#: Instruments the pipeline can price locally without any upstream call.
BUILTIN_PROXIES = {
    "SPY": {"market": MARKET_US, "label": "S&P 500 ETF", "kind": "equity_index"},
    "QQQ": {"market": MARKET_US, "label": "Nasdaq 100 ETF", "kind": "equity_index"},
    "IWM": {"market": MARKET_US, "label": "Russell 2000 ETF", "kind": "equity_index"},
    "EFA": {"market": MARKET_US, "label": "MSCI EAFE ETF", "kind": "equity_index"},
    "EEM": {"market": MARKET_US, "label": "MSCI EM ETF", "kind": "equity_index"},
    "TLT": {"market": MARKET_US, "label": "20+ Year Treasury ETF", "kind": "bond"},
    "IEF": {"market": MARKET_US, "label": "7-10 Year Treasury ETF", "kind": "bond"},
    "GLD": {"market": MARKET_US, "label": "Gold ETF", "kind": "commodity"},
    "DBC": {"market": MARKET_US, "label": "Broad Commodity ETF", "kind": "commodity"},
    "USO": {"market": MARKET_US, "label": "WTI Crude ETF", "kind": "commodity"},
    "CPER": {"market": MARKET_US, "label": "Copper ETF", "kind": "commodity"},
    "DBA": {"market": MARKET_US, "label": "Agriculture ETF", "kind": "commodity"},
    "BIL": {"market": MARKET_US, "label": "1-3 Month Treasury ETF", "kind": "cash"},
}


@dataclass
class Instrument:
    """A resolved tradable instrument."""

    symbol: str
    market: str
    kind: str
    label: str = ""
    provider_hint: str = ""
    capability: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market": self.market,
            "kind": self.kind,
            "label": self.label,
            "provider_hint": self.provider_hint,
            "capability": self.capability,
            "meta": self.meta,
        }


def classify(symbol: str) -> Instrument:
    """Classify a symbol into market + kind without touching the network."""
    raw = (symbol or "").strip()
    if not raw:
        raise ValueError("symbol 不能为空")

    upper = raw.upper()

    if "/" in upper:
        base, quote = upper.split("/", 1)
        return Instrument(symbol=upper, market=MARKET_CRYPTO, kind="crypto_pair",
                          label=f"{base}/{quote}", provider_hint="ccxt",
                          capability=reg.CAP_CRYPTO,
                          meta={"base": base, "quote": quote})

    if upper in BUILTIN_PROXIES:
        info = BUILTIN_PROXIES[upper]
        return Instrument(symbol=upper, market=info["market"], kind=info["kind"],
                          label=info["label"], provider_hint="yahoo",
                          capability=reg.CAP_ETF if info["kind"] != "bond" else reg.CAP_ETF,
                          meta={"builtin": True})

    # A-share style: 6 digits
    if re.fullmatch(r"\d{6}", raw):
        return Instrument(symbol=raw, market=MARKET_A_SHARE, kind="stock",
                          label="", provider_hint="akshare/baostock",
                          capability=reg.CAP_A_SHARE,
                          meta={"exchange": _a_share_exchange(raw)})

    # Explicit suffix hints
    if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", upper):
        base = upper.split(".")[0]
        return Instrument(symbol=base, market=MARKET_A_SHARE, kind="stock",
                          provider_hint="akshare/baostock",
                          capability=reg.CAP_A_SHARE,
                          meta={"exchange": _a_share_exchange(base)})

    # Hong Kong codes are 5 digits (00700, 09988) or 4 digits (0700).
    if re.fullmatch(r"\d{4,5}", raw):
        return Instrument(symbol=raw.zfill(5), market=MARKET_HK, kind="stock",
                          provider_hint="akshare", capability=reg.CAP_HK_EQUITY)

    # US ticker
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", upper):
        return Instrument(symbol=upper, market=MARKET_US, kind="stock",
                          provider_hint="yahoo", capability=reg.CAP_US_EQUITY)

    return Instrument(symbol=raw, market=MARKET_UNKNOWN, kind="unknown",
                      label="", provider_hint="", capability="")


def _a_share_exchange(code: str) -> str:
    """Infer the exchange suffix for a 6-digit A-share code."""
    if code.startswith(("60", "68", "5", "9")):
        return "SH"
    if code.startswith(("00", "30", "2", "1")):
        return "SZ"
    if code.startswith(("4", "8")):
        return "BJ"
    return "SZ"


def baostock_code(instrument: Instrument) -> str:
    """Convert to Baostock's ``sh.600000`` format."""
    if instrument.market != MARKET_A_SHARE:
        raise ValueError(f"{instrument.symbol} 不是 A 股，无法转换为 baostock 代码")
    return f"{instrument.meta.get('exchange', 'SZ').lower()}.{instrument.symbol}"


def resolve(symbol: str) -> Instrument:
    """Public entry point: classify and enrich with a display name when possible."""
    inst = classify(symbol)
    if inst.market == MARKET_A_SHARE:
        try:
            inst.label = _a_share_name(inst.symbol)
            inst.meta["name_source"] = "akshare 东方财富"
        except Exception as exc:  # noqa: BLE001 - name is optional metadata
            inst.meta["name_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    return inst


def _a_share_name(code: str) -> str:
    import akshare as ak
    df = ak.stock_individual_info_em(symbol=code)
    if df is None or df.empty:
        return ""
    mapping = dict(zip(df["item"], df["value"]))
    return str(mapping.get("股票简称", ""))


def peer_universe(instrument: Instrument, *, max_peers: int = 30) -> list[str]:
    """Return a related-instrument list suitable for a cross-sectional panel.

    For A-shares this is the current CSI 300 snapshot (documented survivorship
    bias). For US instruments it is a fixed liquid universe. The target symbol
    is always included first so a panel is never empty.
    """
    if instrument.market == MARKET_A_SHARE:
        try:
            from .providers import csi300_universe
            peers = [c for c in csi300_universe() if c != instrument.symbol]
        except Exception:  # noqa: BLE001 - fall back to a static liquid list
            peers = []
        if not peers:
            peers = [c for c in _STATIC_A_SHARE_LIQUID if c != instrument.symbol]
        return [instrument.symbol] + peers[:max_peers]

    if instrument.market in (MARKET_US, MARKET_ETF):
        peers = [s for s in BUILTIN_PROXIES if s != instrument.symbol]
        return [instrument.symbol] + peers[:max_peers]

    if instrument.market == MARKET_CRYPTO:
        return [instrument.symbol, "BTC/USDT", "ETH/USDT"][:max_peers]

    return [instrument.symbol]


_STATIC_A_SHARE_LIQUID = [
    "600519", "000001", "600036", "601318", "600276", "000858", "601012",
    "600900", "000333", "601166", "600030", "601288", "000651", "002594",
    "600887", "601888", "002415", "000725", "600585", "601601",
]


def describe_sources_for(instrument: Instrument) -> list[dict[str, Any]]:
    """Which registered sources can serve this instrument, best first."""
    if not instrument.capability:
        return []
    rows = []
    for spec in reg.sources_for(instrument.capability, require_available=False):
        rows.append({
            "source_id": spec.source_id,
            "display_name": spec.display_name,
            "tier": spec.tier,
            "latency": spec.latency,
            "fee": spec.fee,
            "available": spec.available(),
            "status_detail": spec.status_detail(),
            "note": spec.note,
        })
    return rows
