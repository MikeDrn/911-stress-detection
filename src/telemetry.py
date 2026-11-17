"""Asynchronous SQLite telemetry, audit log and online-evaluation store.

Writes happen in FastAPI ``BackgroundTasks`` *after* the HTTP response has been
sent, through ``aiosqlite`` (SQLite on a worker thread), so a slow disk never
adds latency to a dispatcher's alert. WAL journaling lets the API write while
readers query concurrently.

Tables
------
calls            one row per scored call (+ which model served it, its routing
                 propensity, shadow result, dispatcher feedback)
windows          per-window scores of the model that served the call
rollout_state    the single current release experiment (mode, canary step ...)
rollout_events   audit trail of every release decision (started, advanced,
                 rolled back, promoted ...)

``init()`` migrates databases created by earlier versions by adding any
missing column, so an existing audit log survives the upgrade.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

CALL_COLUMNS: dict[str, str] = {
    "call_id": "TEXT PRIMARY KEY",
    "received_at": "TEXT NOT NULL",
    "filename": "TEXT",
    "duration_s": "REAL",
    "n_windows": "INTEGER",
    "max_score": "REAL",
    "alert": "INTEGER NOT NULL",
    "n_alert_segments": "INTEGER",
    "threshold": "REAL",
    "consecutive_windows": "INTEGER",
    "model_version": "TEXT",
    "model_source": "TEXT",
    "latency_ms": "REAL",
    "client": "TEXT",
    # --- online evaluation (Course 4) ---
    "experiment_id": "TEXT",
    "mode": "TEXT",
    "arm": "TEXT",
    "propensity": "REAL",
    "canary_step": "INTEGER",
    "model_ms_per_window": "REAL",
    "error": "INTEGER DEFAULT 0",
    "shadow_version": "TEXT",
    "shadow_alert": "INTEGER",
    "shadow_max_score": "REAL",
    "shadow_ms_per_window": "REAL",
    "shadow_error": "INTEGER",
    # --- content + fusion (what was said, and the combined decision) ---
    "transcript": "TEXT",
    "transcript_source": "TEXT",
    "content_label": "TEXT",
    "p_covert": "REAL",
    "stress_score": "REAL",
    "threat_score": "REAL",
    "threat_level": "TEXT",
    "reason": "TEXT",
    "asr_ms": "REAL",
    # --- delayed ground truth from the dispatcher ---
    "true_label": "TEXT",
    "correct": "INTEGER",
    "shadow_correct": "INTEGER",
    "feedback_at": "TEXT",
}

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS calls (
    {", ".join(f"{k} {v}" for k, v in CALL_COLUMNS.items())}
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
CREATE TABLE IF NOT EXISTS rollout_state (
    id    INTEGER PRIMARY KEY CHECK (id = 1),
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rollout_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL,
    experiment_id TEXT,
    event         TEXT NOT NULL,
    details       TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_received ON calls(received_at DESC);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class TelemetryStore:
    def __init__(self, db_path: str | os.PathLike):
        self.db_path = Path(db_path)

    def _connect(self):
        return aiosqlite.connect(self.db_path, timeout=30)

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as db:
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.executescript(SCHEMA)
            cur = await db.execute("PRAGMA table_info(calls)")
            existing = {row[1] for row in await cur.fetchall()}
            for col, decl in CALL_COLUMNS.items():
                if col not in existing:  # migrate an older audit database in place
                    decl = decl.replace("PRIMARY KEY", "").replace("NOT NULL", "")
                    await db.execute(f"ALTER TABLE calls ADD COLUMN {col} {decl}")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_calls_exp ON calls(experiment_id)")
            await db.commit()

    # ----------------------------------------------------------------- calls
    async def log_call(self, call: dict[str, Any], windows: list[dict[str, Any]]) -> None:
        row = {k: call.get(k) for k in CALL_COLUMNS}
        row["alert"] = int(bool(call["alert"]))
        row["error"] = int(call.get("error") or 0)
        cols = ", ".join(row)
        marks = ", ".join(f":{k}" for k in row)
        async with self._connect() as db:
            await db.execute(f"INSERT OR REPLACE INTO calls ({cols}) VALUES ({marks})", row)
            await db.executemany(
                """INSERT OR REPLACE INTO windows VALUES
                   (:call_id, :index, :start_s, :end_s, :duress_score, :above_threshold,
                    :in_alert)""",
                [
                    {
                        **w,
                        "call_id": call["call_id"],
                        "above_threshold": int(w["above_threshold"]),
                        "in_alert": int(w.get("in_alert", w.get("sustained", False))),
                    }
                    for w in windows
                ],
            )
            await db.commit()

    async def update_call(self, call_id: str, **fields: Any) -> bool:
        fields = {k: v for k, v in fields.items() if k in CALL_COLUMNS}
        if not fields:
            return False
        sets = ", ".join(f"{k} = :{k}" for k in fields)
        async with self._connect() as db:
            cur = await db.execute(
                f"UPDATE calls SET {sets} WHERE call_id = :call_id", {**fields, "call_id": call_id}
            )
            await db.commit()
            return cur.rowcount > 0

    async def recent_calls(self, limit: int = 50) -> list[dict[str, Any]]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM calls ORDER BY received_at DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in await cur.fetchall()]

    async def get_call(self, call_id: str) -> dict[str, Any] | None:
        async with self._connect() as db:
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
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT COUNT(*), COALESCE(SUM(alert),0), AVG(latency_ms), MAX(received_at), "
                "COUNT(true_label) FROM calls"
            )
            total, alerts, avg_latency, last, labelled = await cur.fetchone()
            return {
                "total_calls": total,
                "alert_calls": alerts,
                "labelled_calls": labelled,
                "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
                "last_call_at": last,
            }

    # ------------------------------------------------------ experiment data
    async def experiment_calls(
        self, experiment_id: str, canary_step: int | None = None
    ) -> list[dict[str, Any]]:
        q = "SELECT * FROM calls WHERE experiment_id = ?"
        args: list[Any] = [experiment_id]
        if canary_step is not None:
            q += " AND canary_step = ?"
            args.append(canary_step)
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(q + " ORDER BY received_at", args)
            return [dict(r) for r in await cur.fetchall()]

    async def window_scores_by_arm(
        self, experiment_id: str, canary_step: int | None = None
    ) -> dict[str, list[float]]:
        q = (
            "SELECT c.arm, w.duress_score FROM windows w JOIN calls c USING (call_id) "
            "WHERE c.experiment_id = ? AND c.error = 0"
        )
        args: list[Any] = [experiment_id]
        if canary_step is not None:
            q += " AND c.canary_step = ?"
            args.append(canary_step)
        out: dict[str, list[float]] = {"champion": [], "challenger": []}
        async with self._connect() as db:
            cur = await db.execute(q, args)
            for arm, score in await cur.fetchall():
                out.setdefault(arm, []).append(score)
        return out

    async def bandit_counts(self, experiment_id: str) -> dict[str, tuple[int, int]]:
        """{arm: (successes, labelled pulls)} - the bandit's reward table."""
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT arm, COALESCE(SUM(correct),0), COUNT(correct) FROM calls "
                "WHERE experiment_id = ? GROUP BY arm",
                (experiment_id,),
            )
            return {arm: (int(s), int(n)) for arm, s, n in await cur.fetchall()}

    async def experiments(self) -> list[dict[str, Any]]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT experiment_id, mode, COUNT(*) AS calls, MIN(received_at) AS started_at "
                "FROM calls WHERE experiment_id IS NOT NULL GROUP BY experiment_id, mode "
                "ORDER BY started_at DESC"
            )
            return [dict(r) for r in await cur.fetchall()]

    async def last_event(self, experiment_id: str, prefix: str) -> dict[str, Any] | None:
        async with self._connect() as db:
            cur = await db.execute(
                "SELECT at, event, details FROM rollout_events WHERE experiment_id = ? "
                "AND event LIKE ? ORDER BY id DESC LIMIT 1",
                (experiment_id, f"{prefix}%"),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return {"at": row[0], "event": row[1], "details": json.loads(row[2]) if row[2] else None}

    # --------------------------------------------------------- rollout state
    async def load_rollout_state(self) -> dict[str, Any] | None:
        async with self._connect() as db:
            cur = await db.execute("SELECT state FROM rollout_state WHERE id = 1")
            row = await cur.fetchone()
            return json.loads(row[0]) if row else None

    async def save_rollout_state(self, state: dict[str, Any]) -> None:
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO rollout_state (id, state) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET state = excluded.state",
                (json.dumps(state),),
            )
            await db.commit()

    async def log_event(self, experiment_id: str | None, event: str, details: Any = None) -> None:
        async with self._connect() as db:
            await db.execute(
                "INSERT INTO rollout_events (at, experiment_id, event, details) VALUES (?,?,?,?)",
                (now_iso(), experiment_id, event, json.dumps(details, default=str)),
            )
            await db.commit()

    async def events(self, limit: int = 200) -> list[dict[str, Any]]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM rollout_events ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = [dict(r) for r in await cur.fetchall()]
        for r in rows:
            r["details"] = json.loads(r["details"]) if r["details"] else None
        return rows
