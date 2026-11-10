"""Multi-source price fetching with capability-based fallback.

Every provider is wrapped into the same shape: a wide ``date x symbol`` frame
of adjusted closes. :func:`fetch_prices` asks the registry which sources can
serve the requested market and tries them in order, recording each attempt so
the caller can report which upstream actually answered.

This is what makes the research stack reusable: adding a market (US equities,
crypto, ETFs) means adding a provider function here, not touching the backtest,
the factor lab or the visualisation.
"""
from __future__ import annotations

import time
from typing import Iterable

import pandas as pd

from . import registry as reg
from .universe import (
    MARKET_A_SHARE,
    MARKET_CRYPTO,
    MARKET_HK,
    MARKET_US,
    Instrument,
    baostock_code,
    classify,
)


# --------------------------------------------------------------------------- #
# Individual providers
# --------------------------------------------------------------------------- #
def _akshare_ashare(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    import akshare as ak

    frames, failed = [], []
    end_compact = (end or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    start_compact = start.replace("-", "")
    for code in symbols:
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start_compact, end_date=end_compact,
                                    adjust="qfq")
            if df is None or df.empty:
                failed.append(code)
                continue
            series = df.set_index("日期")["收盘"].rename(code)
            series.index = pd.to_datetime(series.index)
            frames.append(series.to_frame())
        except Exception:  # noqa: BLE001 - one bad code must not kill the pull
            failed.append(code)
        time.sleep(0.12)
    if not frames:
        raise RuntimeError(f"AkShare 未返回任何 A 股数据（尝试 {len(list(symbols))} 只）")
    panel = pd.concat(frames, axis=1).sort_index()
    panel.attrs["failed"] = failed
    return panel


def _baostock_ashare(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    import baostock as bs

    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {login.error_msg}")
    frames, failed = [], []
    # Baostock's query API expects YYYY-MM-DD dates (the compact YYYYMMDD form
    # it uses for its own upload endpoints is rejected here).
    start_date = str(pd.Timestamp(start).date())
    end_date = str(pd.Timestamp(end or pd.Timestamp.today()).date())
    try:
        for code in symbols:
            inst = classify(code)
            query = bs.query_history_k_data_plus(
                baostock_code(inst), "date,close",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2")
            if query.error_code != "0":
                failed.append(code)
                continue
            rows = []
            while query.next():
                rows.append(query.get_row_data())
            if not rows:
                failed.append(code)
                continue
            frame = pd.DataFrame(rows, columns=["date", "close"])
            frame["date"] = pd.to_datetime(frame["date"])
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            series = frame.dropna().set_index("date")["close"].rename(code)
            frames.append(series.to_frame())
    finally:
        bs.logout()
    if not frames:
        raise RuntimeError(f"Baostock 未返回任何 A 股数据（失败 {len(failed)} 只）")
    panel = pd.concat(frames, axis=1).sort_index()
    panel.attrs["failed"] = failed
    return panel


def _yahoo(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    tickers = [s for s in symbols]
    if not tickers:
        raise RuntimeError("yfinance 需要非空标的列表")
    raw = yf.download(tickers, start=start, end=end, progress=False,
                      auto_adjust=True, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance 返回空数据")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all")
    close.columns = [str(c) for c in close.columns]
    return close


def _efinance_ashare(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    import efinance as ef

    frames = []
    for code in symbols:
        try:
            df = ef.stock.get_quote_history(code, beg=start.replace("-", ""),
                                            end=(end or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", ""))
            if df is None or df.empty:
                continue
            df = df.copy()
            df["日期"] = pd.to_datetime(df["日期"])
            series = df.set_index("日期")["收盘"].rename(code)
            frames.append(series.to_frame())
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        raise RuntimeError("efinance 未返回任何 A 股数据")
    return pd.concat(frames, axis=1).sort_index()


def _mootdx_ashare(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    from mootdx.quotes import Quotes

    client = Quotes.factory(market="std")
    frames = []
    for code in symbols:
        try:
            frame = client.bars(symbol=code, frequency=9,
                                offset=800, start=start.replace("-", ""),
                                end=(end or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", ""))
            if frame is None or frame.empty:
                continue
            series = frame.set_index("datetime")["close"].rename(code)
            frames.append(series.to_frame())
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        raise RuntimeError("mootdx 未返回任何 A 股数据")
    return pd.concat(frames, axis=1).sort_index()


def _adata_ashare(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    import adata

    frames = []
    for code in symbols:
        try:
            df = adata.stock.info.market.get_market_ths(symbol=code, start_date=start,
                                                         k_type=1)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            series = df.set_index("trade_date")["close"].rename(code)
            frames.append(series.to_frame())
        except Exception:  # noqa: BLE001
            continue
    if not frames:
        raise RuntimeError("adata 未返回任何 A 股数据")
    return pd.concat(frames, axis=1).sort_index()


#: CCXT exchanges probed in order. Some are geo-restricted from mainland
#: networks (Binance returns HTTP 451), so the provider walks this list and
#: uses the first one that actually answers.
CCXT_EXCHANGE_PRIORITY = (
    "okx", "gate", "kucoin", "bitstamp", "bitfinex", "kraken",
    "mexc", "htx", "coinbase", "binance",
)


def _ccxt_crypto(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    import ccxt

    since = None
    frames = []
    attempted: list[str] = []
    for exchange_id in CCXT_EXCHANGE_PRIORITY:
        try:
            exchange = ccxt.__dict__[exchange_id]({"enableRateLimit": True})
        except Exception:  # noqa: BLE001 - unknown id in this ccxt build
            continue
        try:
            since = since or exchange.parse8601(f"{start}T00:00:00Z")
            ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1d", since=since, limit=3)
            if not ohlcv:
                attempted.append(f"{exchange_id}:empty")
                continue
        except Exception as exc:  # noqa: BLE001 - try the next exchange
            attempted.append(f"{exchange_id}:{type(exc).__name__}")
            continue

        # This exchange answered the probe; pull the requested symbols.
        for symbol in symbols:
            try:
                data = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since, limit=1000)
                if not data:
                    continue
                frame = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "vol"])
                frame["date"] = pd.to_datetime(frame["ts"], unit="ms")
                series = frame.set_index("date")["close"].rename(symbol)
                frames.append(series.to_frame())
            except Exception:  # noqa: BLE001
                continue
        if frames:
            panel = pd.concat(frames, axis=1).sort_index()
            panel.attrs["exchange"] = exchange_id
            panel.attrs["attempted"] = attempted
            return panel
    raise RuntimeError(f"所有候选交易所均不可用：{attempted}")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
_MARKET_FETCHERS = {
    MARKET_A_SHARE: {
        "akshare": lambda syms, s, e: _akshare_ashare(syms, s, e),
        "baostock": lambda syms, s, e: _baostock_ashare(syms, s, e),
        "efinance": lambda syms, s, e: _efinance_ashare(syms, s, e),
        "mootdx": lambda syms, s, e: _mootdx_ashare(syms, s, e),
        "adata": lambda syms, s, e: _adata_ashare(syms, s, e),
    },
    MARKET_US: {"yahoo": lambda syms, s, e: _yahoo(syms, s, e)},
    MARKET_HK: {"akshare": lambda syms, s, e: _akshare_ashare(syms, s, e),
                "yahoo": lambda syms, s, e: _yahoo(syms, s, e)},
    MARKET_CRYPTO: {"ccxt": lambda syms, s, e: _ccxt_crypto(syms, s, e),
                    "yahoo": lambda syms, s, e: _yahoo(syms, s, e)},
}


def _local_db_prices(symbols: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    """Read a price panel from the local SQLite database.

    This is the offline fallback: whatever was synced earlier stays usable
    even when every online upstream is unreachable. Every stored dataset is
    searched for the requested symbols -- not just the two well-known ones --
    so a symbol synced in any dataset is found.
    """
    from .store import Store, default_db_path

    syms = [str(s) for s in symbols]
    store = Store(default_db_path())
    datasets = store.datasets()
    dataset_ids = list(datasets["dataset_id"]) if not datasets.empty else []
    # Prefer the market-price datasets but fall back to any dataset that
    # happens to carry the requested columns.
    ordered = [d for d in ("market_prices", "ashare_prices") if d in dataset_ids]
    ordered += [d for d in dataset_ids if d not in ordered]

    found: dict[str, pd.Series] = {}
    for dataset_id in ordered:
        frame = store.load_frame(dataset_id, start=start, end=end)
        if frame.empty:
            continue
        for sym in syms:
            if sym in frame.columns and sym not in found:
                series = frame[sym].dropna()
                if not series.empty:
                    found[sym] = series.rename(sym)
    if not found:
        raise RuntimeError(f"本地库中没有 {syms} 的价格数据（已检索 {len(ordered)} 个数据集）")
    panel = pd.concat(list(found.values()), axis=1).sort_index()
    panel.attrs["missing"] = [s for s in syms if s not in found]
    return panel


def fetch_prices(
    symbols: Iterable[str],
    *,
    start: str = "2015-01-01",
    end: str | None = None,
    market: str | None = None,
    use_local_first: bool = True,
) -> reg.FetchOutcome:
    """Fetch a price panel for ``symbols``, falling back across data sources.

    ``market`` is inferred from the first symbol when not given. The returned
    outcome records which source answered and what happened to the others.
    With ``use_local_first`` the local SQLite database is consulted before the
    network, which keeps the pipeline reproducible and offline-capable.
    """
    syms = [str(s) for s in symbols]
    if not syms:
        raise ValueError("symbols 不能为空")
    resolved_market = market or classify(syms[0]).market
    fetchers = _MARKET_FETCHERS.get(resolved_market)
    if not fetchers:
        raise ValueError(f"市场 {resolved_market!r} 暂无可用取数通道")

    capability = {
        MARKET_A_SHARE: reg.CAP_A_SHARE,
        MARKET_US: reg.CAP_US_EQUITY,
        MARKET_HK: reg.CAP_HK_EQUITY,
        MARKET_CRYPTO: reg.CAP_CRYPTO,
    }[resolved_market]

    # Only keep fetchers whose source is registered for this capability and is
    # actually installed on this host. ``raw`` keeps the unbound provider so
    # the partial-cache path can re-call it with a different symbol list.
    raw: dict[str, Callable[[list[str], str, str | None], pd.DataFrame]] = {}
    for source_id, fn in fetchers.items():
        spec = reg.BY_ID.get(source_id)
        if spec is None:
            continue
        if capability not in spec.capabilities:
            continue
        if not spec.available():
            continue
        raw[source_id] = fn

    if use_local_first:
        local_frame = None
        local_error = ""
        try:
            local_frame = _local_db_prices(syms, start, end)
        except Exception as exc:  # noqa: BLE001 - fall through to online sources
            local_error = f"{type(exc).__name__}: {str(exc)[:120]}"
        else:
            missing = list(local_frame.attrs.get("missing", []))
            if not missing:
                return reg.FetchOutcome(
                    frame=local_frame, source_id="local_db",
                    attempts=[{"source": "local_db", "outcome": "ok",
                               "detail": f"{local_frame.shape[0]} 行 × "
                                         f"{local_frame.shape[1]} 列"}],
                    degraded=True,
                )
            local_error = (f"本地库命中 {local_frame.shape[1]} 只，"
                           f"缺失 {len(missing)} 只：{missing[:5]}")

        # Local cache covers part of the universe: fetch the rest online and
        # merge, so a partially-synced database is still useful.
        if local_frame is not None and raw:
            missing = list(local_frame.attrs.get("missing", []))
            if missing:
                online = reg.routed_fetch(
                    capability,
                    {sid: (lambda fn=fn, syms=missing: fn(syms, start, end))
                     for sid, fn in raw.items()})
                merged = pd.concat([local_frame, online.frame], axis=1)
                merged = merged.loc[:, ~merged.columns.duplicated()].sort_index()
                return reg.FetchOutcome(
                    frame=merged, source_id=f"local_db+{online.source_id}",
                    attempts=[{"source": "local_db", "outcome": "partial",
                               "detail": local_error},
                              {"source": online.source_id, "outcome": "ok",
                               "detail": f"补齐 {missing[:5]}"}],
                    degraded=True,
                )
    else:
        local_error = "未启用本地库优先"

    if not raw:
        raise RuntimeError(
            f"市场 {resolved_market} 的在线取数通道均不可用（本地库：{local_error}）"
        )
    usable = {sid: (lambda fn=fn: fn(syms, start, end)) for sid, fn in raw.items()}
    outcome = reg.routed_fetch(capability, usable)
    outcome.attempts.insert(0, {"source": "local_db", "outcome": "miss",
                                 "detail": local_error})
    return outcome
