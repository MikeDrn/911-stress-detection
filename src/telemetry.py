"""Asynchronous SQLite telemetry & audit log.

Writes happen in FastAPI ``BackgroundTasks`` *after* the HTTP response has been
sent, through ``aiosqlite`` (SQLite on a worker thread), so a slow disk never
adds latency to a dispatcher's alert. WAL journaling lets the API write while
readers (``/telemetry`` endpoints) query concurrently.

Schema
------
calls(call_id PK, received_at, filename, duration_s, n_windows, max_score,
      alert, n_alert_segments, threshold, consecutive_windows, model_version,
      model_source, latency_ms, client)
windows(call_id FK, window_idx, start_s, end_s, duress_score, above_threshold,
        in_alert)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    call_id             TEXT PRIMARY KEY,
    received_at         TEXT NOT NULL,
    filename            TEXT,
    duration_s          REAL,
    n_windows           INTEGER,
    max_score           REAL,
    alert               INTEGER NOT NULL,
    n_alert_segments    INTEGER,
    threshold           REAL,
    consecutive_windows INTEGER,
    model_version       TEXT,
    model_source        TEXT,
    latency_ms          REAL,
    client              TEXT
);
CREATE TABLE IF NOT EXISTS windows (
    call_id         TEXT NOT NULL REFERENCES calls(call_id) ON DELETE CASCADE,
    window_idx      INTEGER NOT NULL,
    start_s         REAL,
    end_s           REAL,
    duress_score    REAL,
    above_threshold INTEGER,
    in_alert        INTEGER,
    PRIMARY KEY (call_id, window_idx)
);
CREATE INDEX IF NOT EXISTS idx_calls_received ON calls(received_at DESC);
"""


class TelemetryStore:
    def __init__(self, db_path: str | os.PathLike):
        self.db_path = Path(db_path)

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.executescript(SCHEMA)
            await db.commit()

    async def log_call(self, call: dict[str, Any], windows: list[dict[str, Any]]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO calls VALUES
                   (:call_id, :received_at, :filename, :duration_s, :n_windows, :max_score,
                    :alert, :n_alert_segments, :threshold, :consecutive_windows,
                    :model_version, :model_source, :latency_ms, :client)""",
                {**call, "alert": int(call["alert"])},
            )
            await db.executemany(
                """INSERT OR REPLACE INTO windows VALUES
                   (:call_id, :index, :start_s, :end_s, :duress_score, :above_threshold,
                    :in_alert)""",
                [
                    {
                        **w,
                        "call_id": call["call_id"],
                        "above_threshold": int(w["above_threshold"]),
                        "in_alert": int(w["in_alert"]),
                    }
                    for w in windows
                ],
            )
            await db.commit()

    async def recent_calls(self, limit: int = 50) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM calls ORDER BY received_at DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in await cur.fetchall()]

    async def get_call(self, call_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,))
            row = await cur.fetchone()
            if row is None:
                return None
            cur = await db.execute(
                "SELECT * FROM windows WHERE call_id = ? ORDER BY window_idx", (call_id,)
            )
            return {**dict(row), "windows": [dict(r) for r in await cur.fetchall()]}

    async def stats(self) -> dict[str, Any]:
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT COUNT(*), COALESCE(SUM(alert),0), AVG(latency_ms), MAX(received_at) "
                "FROM calls"
            )
            total, alerts, avg_latency, last = await cur.fetchone()
            return {
                "total_calls": total,
                "alert_calls": alerts,
                "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
                "last_call_at": last,
            }
