"""Local financial database: SQLite storage for macro, price and factor series.

The database is deliberately a single long-form ``observations`` table so that
any wide panel (FRED macro, daily ETF prices, Fama-French factors, Wind EDB
exports) can be stored without schema changes. Every dataset carries its own
source, frequency, coverage and fetch time for auditability.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    endpoint      TEXT,
    transport     TEXT NOT NULL,
    requires_key  INTEGER NOT NULL DEFAULT 0,
    note          TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id  TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES sources(source_id),
    kind        TEXT NOT NULL,
    description TEXT,
    unit        TEXT,
    frequency   TEXT,
    params      TEXT,
    first_date  TEXT,
    last_date   TEXT,
    n_obs       INTEGER NOT NULL DEFAULT 0,
    fetched_at  TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    dataset_id TEXT NOT NULL,
    obs_date   TEXT NOT NULL,
    series_id  TEXT NOT NULL,
    value      REAL NOT NULL,
    PRIMARY KEY (dataset_id, obs_date, series_id)
);
CREATE INDEX IF NOT EXISTS ix_obs_dataset_date
    ON observations (dataset_id, obs_date);

CREATE TABLE IF NOT EXISTS sync_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id   TEXT,
    source_id    TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    status       TEXT,
    rows_written INTEGER,
    message      TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_db_path() -> Path:
    return Path.home() / ".local" / "share" / "clockdata" / "finance.db"


class Store:
    """Thin SQLite wrapper. Each call opens its own short-lived connection."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else default_db_path()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def init_schema(self, sources: list[dict[str, Any]] | None = None) -> "Store":
        with self._connect() as con:
            con.executescript(SCHEMA)
            for src in sources or []:
                con.execute(
                    """INSERT INTO sources(source_id, display_name, endpoint, transport, requires_key, note)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(source_id) DO UPDATE SET
                         display_name=excluded.display_name, endpoint=excluded.endpoint,
                         transport=excluded.transport, requires_key=excluded.requires_key,
                         note=excluded.note""",
                    (src["source_id"], src["display_name"], src.get("endpoint"),
                     src["transport"], int(src.get("requires_key", 0)), src.get("note")),
                )
        return self

    def upsert_frame(
        self,
        dataset_id: str,
        source_id: str,
        kind: str,
        frame: pd.DataFrame,
        *,
        description: str = "",
        unit: str = "",
        frequency: str = "",
        params: dict[str, Any] | None = None,
    ) -> int:
        """Write a wide DatetimeIndex DataFrame as long-form observations."""
        if frame.empty:
            raise ValueError(f"Refusing to store an empty frame for {dataset_id!r}")
        wide = frame.copy()
        wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index)).tz_localize(None).normalize()
        wide.index.name = "date"
        wide.columns = [str(c) for c in wide.columns]
        wide = wide.apply(pd.to_numeric, errors="coerce")
        wide = wide[~wide.index.duplicated(keep="last")].sort_index()

        long = wide.reset_index().melt(id_vars="date", var_name="series_id", value_name="value")
        long["date"] = long["date"].dt.strftime("%Y-%m-%d")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["value"])
        rows = [
            (dataset_id, str(d), str(s), float(v))
            for d, s, v in long[["date", "series_id", "value"]].itertuples(index=False, name=None)
        ]
        if not rows:
            raise ValueError(f"No numeric observations in {dataset_id!r}")

        first_date, last_date = long["date"].min(), long["date"].max()
        with self._connect() as con:
            con.execute(
                """INSERT INTO datasets(dataset_id, source_id, kind, description, unit, frequency, params,
                                        first_date, last_date, n_obs, fetched_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(dataset_id) DO UPDATE SET
                     source_id=excluded.source_id, kind=excluded.kind,
                     description=excluded.description, unit=excluded.unit,
                     frequency=excluded.frequency, params=excluded.params,
                     first_date=MIN(datasets.first_date, excluded.first_date),
                     last_date=MAX(datasets.last_date, excluded.last_date),
                     n_obs=(SELECT COUNT(*) FROM observations WHERE dataset_id=excluded.dataset_id),
                     fetched_at=excluded.fetched_at""",
                (dataset_id, source_id, kind, description, unit, frequency,
                 json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
                 first_date, last_date, len(rows), utcnow()),
            )
            con.executemany(
                """INSERT INTO observations(dataset_id, obs_date, series_id, value)
                   VALUES(?,?,?,?)
                   ON CONFLICT(dataset_id, obs_date, series_id) DO UPDATE SET value=excluded.value""",
                rows,
            )
            con.execute(
                "UPDATE datasets SET first_date=(SELECT MIN(obs_date) FROM observations WHERE dataset_id=?), "
                "last_date=(SELECT MAX(obs_date) FROM observations WHERE dataset_id=?), "
                "n_obs=(SELECT COUNT(*) FROM observations WHERE dataset_id=?), fetched_at=? "
                "WHERE dataset_id=?",
                (dataset_id, dataset_id, dataset_id, utcnow(), dataset_id),
            )
        return len(rows)

    def load_frame(
        self,
        dataset_id: str,
        start: str | None = None,
        end: str | None = None,
        series: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read a dataset back into the original wide shape."""
        query = "SELECT obs_date, series_id, value FROM observations WHERE dataset_id=?"
        params: list[Any] = [dataset_id]
        if start:
            query += " AND obs_date>=?"
            params.append(pd.Timestamp(start).strftime("%Y-%m-%d"))
        if end:
            query += " AND obs_date<=?"
            params.append(pd.Timestamp(end).strftime("%Y-%m-%d"))
        if series:
            query += f" AND series_id IN ({','.join('?' * len(series))})"
            params.extend(series)
        with self._connect() as con:
            long = pd.read_sql_query(query, con, params=params)
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="obs_date", columns="series_id", values="value").sort_index()
        wide.index = pd.DatetimeIndex(pd.to_datetime(wide.index))
        wide.index.name = "date"
        return wide

    def datasets(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """SELECT d.dataset_id, d.source_id, s.display_name, d.kind, d.frequency,
                          d.first_date, d.last_date, d.n_obs, d.fetched_at, d.description
                   FROM datasets d LEFT JOIN sources s ON s.source_id=d.source_id
                   ORDER BY d.dataset_id""", con)

    def sources(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query("SELECT * FROM sources ORDER BY source_id", con)

    def dataset_meta(self, dataset_id: str) -> dict[str, Any] | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if row is None:
            return None
        keys = ["dataset_id", "source_id", "kind", "description", "unit", "frequency",
                "params", "first_date", "last_date", "n_obs", "fetched_at"]
        result = dict(zip(keys, row))
        result["params"] = json.loads(result.get("params") or "{}")
        return result

    def log_sync(self, dataset_id: str, source_id: str, started_at: str,
                 status: str, rows_written: int, message: str = "") -> None:
        with self._connect() as con:
            con.execute(
                """INSERT INTO sync_log(dataset_id, source_id, started_at, finished_at,
                                        status, rows_written, message)
                   VALUES(?,?,?,?,?,?,?)""",
                (dataset_id, source_id, started_at, utcnow(), status, rows_written, message),
            )

    def sync_history(self, limit: int = 20) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                "SELECT * FROM sync_log ORDER BY id DESC LIMIT ?", con, params=(limit,))

    def query(self, sql: str) -> pd.DataFrame:
        """Read-only SQL helper for ad-hoc inspection."""
        if not sql.lstrip().lower().startswith(("select", "with", "pragma", "explain")):
            raise ValueError("Only read-only SELECT/WITH/PRAGMA/EXPLAIN statements are allowed")
        with self._connect() as con:
            return pd.read_sql_query(sql, con)

    def db_size_bytes(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0
