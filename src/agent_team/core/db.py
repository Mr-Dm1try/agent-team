"""One SQLite file for runs, tasks, artifacts, checkpoints, decisions, feedback and
verdicts. Write-ahead logging is on, so every committed row survives a crash.

The connection is shared and guarded by a lock. Calls run in a worker thread so the
event loop never blocks on disk.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    status TEXT NOT NULL,
    phase TEXT NOT NULL,
    created_at TEXT NOT NULL,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tasks_by_run ON tasks (run_id, status);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    version INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, kind, version)
);
CREATE TABLE IF NOT EXISTS lineage (
    run_id TEXT NOT NULL,
    child TEXT NOT NULL,
    parent TEXT NOT NULL,
    PRIMARY KEY (run_id, child, parent)
);
CREATE INDEX IF NOT EXISTS lineage_by_parent ON lineage (run_id, parent);
CREATE TABLE IF NOT EXISTS checkpoints (
    task_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    step INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    messages TEXT NOT NULL,
    PRIMARY KEY (task_id, attempt, step)
);
CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_by_run ON decisions (run_id, seq);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    artifact_kind TEXT,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_by_kind ON feedback (run_id, artifact_kind, seq);
CREATE TABLE IF NOT EXISTS verdicts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    task_id TEXT,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS verdicts_by_task ON verdicts (task_id, seq);
"""

Params = Sequence[Any]


class Database:
    """Thin async wrapper around one sqlite3 connection."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    @property
    def journal_mode(self) -> str:
        """SQLite journal mode in effect (``wal`` for a file database)."""
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return str(row[0])

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            self._conn.close()

    # --- sync helpers, run in a worker thread ------------------------------------------

    def _fetchall(self, sql: str, params: Params) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def _transaction[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(self._conn)
                self._conn.execute("COMMIT")
            except BaseException:
                # Also covers a failed COMMIT, so the connection never stays mid-transaction.
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
            return result

    def _write(self, statements: Iterable[tuple[str, Params]]) -> None:
        def run(conn: sqlite3.Connection) -> None:
            for sql, params in statements:
                conn.execute(sql, params)

        self._transaction(run)

    # --- async API ---------------------------------------------------------------------

    async def fetchall(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        """Rows for a read query."""
        return await asyncio.to_thread(self._fetchall, sql, params)

    async def fetchone(self, sql: str, params: Params = ()) -> sqlite3.Row | None:
        """First row of a read query, or None."""
        rows = await self.fetchall(sql, params)
        return rows[0] if rows else None

    async def execute(self, sql: str, params: Params = ()) -> None:
        """Run one write statement in its own transaction."""
        await self.write([(sql, params)])

    async def write(self, statements: Iterable[tuple[str, Params]]) -> None:
        """Run several statements in one transaction."""
        await asyncio.to_thread(self._write, list(statements))

    async def transaction[T](self, fn: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``fn`` on the connection inside one ``BEGIN IMMEDIATE`` transaction, so
        reads and writes in it are atomic across processes."""
        return await asyncio.to_thread(self._transaction, fn)
