"""The memory store: a memories table and an FTS5 index in the run database
(docs/DESIGN.md, "Memory").

Keyword matches come from SQLite's built-in BM25 over the FTS5 index; meaning matches
from a numpy dot product over the stored unit vectors, which takes milliseconds for a
few thousand memories. Both are deterministic, and ties break on age and id. The
memory tables are created here, next to the run tables, so a database written before
memory existed simply gains them.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Sequence
from typing import Any

import numpy as np

from agent_team.core.db import Database
from agent_team.core.models import Memory, MemoryKind, MemoryMatch

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        norm TEXT NOT NULL,
        created_at TEXT NOT NULL,
        usefulness INTEGER NOT NULL DEFAULT 0,
        vector BLOB,
        data TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS memories_by_text ON memories (kind, norm)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(id UNINDEXED, text, tokenize = 'porter unicode61')""",
    """CREATE TABLE IF NOT EXISTS memory_fetches (
        task_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        run_id TEXT NOT NULL,
        query TEXT NOT NULL,
        ids TEXT NOT NULL,
        credited INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (task_id, attempt)
    )""",
)

#: Words too common to help a keyword match.
_COMMON = (
    "a an and are as at be by for from has have in into is it its of on or that the "
    "this to was were will with"
)
STOPWORDS = frozenset(_COMMON.split())
#: A word, for keyword matching and the fake embedder.
WORD = re.compile(r"[a-z0-9]+")
#: Terms sent to FTS5 from one query, so a long objective stays a cheap search.
MAX_TERMS = 32


def normalize(text: str) -> str:
    """The text as compared for exact duplicates: lower case, words only."""
    return " ".join(WORD.findall(text.lower()))


def terms(text: str) -> set[str]:
    """The text's distinct words, without the common ones and a plural ``s``."""
    words = (w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1)
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words}


def overlap(a: set[str], b: set[str]) -> float:
    """Words in common, 0 to 1: the cosine of the two word sets."""
    return len(a & b) / math.sqrt(len(a) * len(b)) if a and b else 0.0


def match_query(text: str) -> str | None:
    """An FTS5 query that matches any of the text's words, each quoted so no word is
    read as FTS5 syntax; None when no word is left."""
    terms = [w for w in dict.fromkeys(WORD.findall(text.lower())) if w not in STOPWORDS]
    terms = [w for w in terms if len(w) > 1][:MAX_TERMS]
    return " OR ".join(f'"{w}"' for w in terms) or None


def _blob(vector: Sequence[float]) -> bytes:
    """A vector scaled to length 1, as float32 bytes."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return (arr / norm if norm else arr).tobytes()


def _unit(vector: Sequence[float]) -> np.ndarray[Any, np.dtype[np.float32]]:
    """A query vector scaled to length 1."""
    return np.frombuffer(_blob(vector), dtype=np.float32)


def _memory(row: sqlite3.Row) -> Memory:
    """A memory from its row; usefulness lives in its own column."""
    data = json.loads(row["data"])
    data["usefulness"] = row["usefulness"]
    return Memory.model_validate(data)


def _data(memory: Memory) -> str:
    """The JSON kept for a memory, without the columns stored apart."""
    return memory.model_dump_json(exclude={"usefulness"})


class SqliteMemoryIndex:
    """Implements ``MemoryIndex`` over the run database. Create it with ``open``."""

    def __init__(self, db: Database) -> None:
        """Use ``db``, whose memory tables ``open`` has created."""
        self.db = db

    @classmethod
    async def open(cls, db: Database) -> SqliteMemoryIndex:
        """Create the memory tables in ``db`` if they're missing, and return the index."""

        def create(conn: sqlite3.Connection) -> None:
            """Run each statement of the memory schema."""
            for statement in SCHEMA:
                conn.execute(statement)

        await db.transaction(create)
        return cls(db)

    async def save(
        self, memory: Memory, vector: Sequence[float] | None, *, merge_at: float
    ) -> tuple[Memory, bool]:
        """Insert ``memory``, or merge it into the oldest memory of its kind with the
        same words, else the closest one at a cosine of ``merge_at`` or more. A merge
        keeps the older id and text and adds the new sources, in one transaction."""
        norm = normalize(memory.text)
        blob = _blob(vector) if vector is not None else None

        def run(conn: sqlite3.Connection) -> tuple[Memory, bool]:
            """Find a duplicate and merge into it, or insert."""
            row = conn.execute(
                "SELECT * FROM memories WHERE kind = ? AND norm = ? ORDER BY created_at, id",
                (memory.kind.value, norm),
            ).fetchone()
            if row is None and blob is not None:
                row = _closest(conn, memory.kind, blob, merge_at)
            if row is None:
                conn.execute(
                    "INSERT INTO memories (id, kind, norm, created_at, usefulness, vector, data) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        memory.id,
                        memory.kind.value,
                        norm,
                        memory.created_at.isoformat(),
                        memory.usefulness,
                        blob,
                        _data(memory),
                    ),
                )
                conn.execute(
                    "INSERT INTO memories_fts (id, text) VALUES (?, ?)", (memory.id, memory.text)
                )
                return memory, False
            kept = _memory(row)
            new = [s for s in memory.sources if s not in kept.sources]
            kept = kept.model_copy(update={"sources": [*kept.sources, *new]})
            conn.execute(
                "UPDATE memories SET data = ?, vector = COALESCE(vector, ?) WHERE id = ?",
                (_data(kept), blob, kept.id),
            )
            return kept, True

        return await self.db.transaction(run)

    async def search(
        self,
        text: str,
        vector: Sequence[float] | None,
        kinds: Sequence[MemoryKind],
        *,
        limit: int,
    ) -> list[MemoryMatch]:
        """The top ``limit`` keyword matches and the top ``limit`` vector matches among
        ``kinds``, each with both scores where they exist, in id order."""
        if not kinds:
            return []
        marks = ", ".join("?" for _ in kinds)
        kind_values = [k.value for k in kinds]
        bm25: dict[str, float] = {}
        fts = match_query(text)
        if fts is not None:
            rows = await self.db.fetchall(
                "SELECT f.id AS id, -bm25(memories_fts) AS score FROM memories_fts AS f "
                f"JOIN memories AS m ON m.id = f.id WHERE memories_fts MATCH ? "
                f"AND m.kind IN ({marks}) ORDER BY bm25(memories_fts), f.id LIMIT ?",
                (fts, *kind_values, limit),
            )
            bm25 = {r["id"]: max(float(r["score"]), 0.0) for r in rows}
        cosine: dict[str, float] = {}
        query = _unit(vector) if vector is not None else None
        if query is not None:
            rows = await self.db.fetchall(
                f"SELECT id, vector FROM memories WHERE kind IN ({marks}) "
                "AND vector IS NOT NULL ORDER BY id",
                tuple(kind_values),
            )
            # Vectors from another embedding model (another length) can't be compared.
            rows = [r for r in rows if len(r["vector"]) == query.nbytes]
            if rows:
                matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
                scores = matrix @ query
                # Stable sort on the id-ordered rows: equal scores keep id order.
                best = np.argsort(-scores, kind="stable")[:limit]
                cosine = {rows[i]["id"]: float(scores[i]) for i in best}
        ids = sorted(set(bm25) | set(cosine))
        if not ids:
            return []
        found = await self._rows(ids)
        asked = terms(text)
        out: list[MemoryMatch] = []
        for memory_id in ids:
            row = found[memory_id]
            cos = cosine.get(memory_id)
            if cos is None and query is not None and row["vector"] is not None:
                # A keyword match outside the top vector matches still gets its cosine.
                stored = np.frombuffer(row["vector"], dtype=np.float32)
                cos = float(stored @ query) if stored.shape == query.shape else None
            memory = _memory(row)
            words = overlap(asked, terms(memory.text)) if memory_id in bm25 else 0.0
            out.append(
                MemoryMatch(memory=memory, cosine=cos, bm25=bm25.get(memory_id, 0.0), words=words)
            )
        return out

    async def _rows(self, ids: Sequence[str]) -> dict[str, sqlite3.Row]:
        """The memory rows with these ids, by id."""
        marks = ", ".join("?" for _ in ids)
        rows = await self.db.fetchall(f"SELECT * FROM memories WHERE id IN ({marks})", tuple(ids))
        return {r["id"]: r for r in rows}

    async def memories(self, kinds: Sequence[MemoryKind] | None = None) -> list[Memory]:
        """Every memory, or those of ``kinds``, oldest first."""
        if kinds is None:
            rows = await self.db.fetchall("SELECT * FROM memories ORDER BY created_at, id")
        else:
            marks = ", ".join("?" for _ in kinds) or "NULL"
            rows = await self.db.fetchall(
                f"SELECT * FROM memories WHERE kind IN ({marks}) ORDER BY created_at, id",
                tuple(k.value for k in kinds),
            )
        return [_memory(r) for r in rows]

    async def without_vectors(self, limit: int) -> list[Memory]:
        """Memories saved while the embedder was down, oldest first."""
        rows = await self.db.fetchall(
            "SELECT * FROM memories WHERE vector IS NULL ORDER BY created_at, id LIMIT ?",
            (limit,),
        )
        return [_memory(r) for r in rows]

    async def set_vector(self, memory_id: str, vector: Sequence[float]) -> None:
        """Store the vector of a memory saved without one; a vector already there stays."""
        await self.db.execute(
            "UPDATE memories SET vector = COALESCE(vector, ?) WHERE id = ?",
            (_blob(vector), memory_id),
        )

    async def forget(self, memory_id: str) -> Memory | None:
        """Delete one memory and its keyword entry in one transaction; returns what was
        deleted, or None when there is no such memory."""

        def run(conn: sqlite3.Connection) -> Memory | None:
            """Read the row, then delete it from both tables."""
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
            return _memory(row)

        return await self.db.transaction(run)

    async def record_fetch(
        self, run_id: str, task_id: str, attempt: int, query: str, ids: Sequence[str]
    ) -> None:
        """Remember which memories one attempt was given; a resumed attempt's newer
        fetch replaces the one before it, and a task already credited stays credited."""
        await self.db.execute(
            "INSERT INTO memory_fetches (task_id, attempt, run_id, query, ids) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (task_id, attempt) DO UPDATE SET "
            "query = excluded.query, ids = excluded.ids",
            (task_id, attempt, run_id, query, json.dumps(list(ids))),
        )

    async def credit(self, task_id: str) -> list[str]:
        """Add one to the usefulness of each memory the task's first attempt was given,
        once per task, in one transaction."""

        def run(conn: sqlite3.Connection) -> list[str]:
            """Mark the fetch credited and bump its memories."""
            row = conn.execute(
                "SELECT ids FROM memory_fetches WHERE task_id = ? AND attempt = 1 AND credited = 0",
                (task_id,),
            ).fetchone()
            if row is None:
                return []
            ids: list[str] = json.loads(row["ids"])
            conn.execute(
                "UPDATE memory_fetches SET credited = 1 WHERE task_id = ? AND attempt = 1",
                (task_id,),
            )
            conn.executemany(
                "UPDATE memories SET usefulness = usefulness + 1 WHERE id = ?",
                [(i,) for i in ids],
            )
            return ids

        return await self.db.transaction(run)


def _closest(
    conn: sqlite3.Connection, kind: MemoryKind, blob: bytes, merge_at: float
) -> sqlite3.Row | None:
    """The memory of ``kind`` whose vector is closest to ``blob``, if its cosine is at
    least ``merge_at``; on a tie the oldest."""
    rows = conn.execute(
        "SELECT * FROM memories WHERE kind = ? AND vector IS NOT NULL ORDER BY created_at, id",
        (kind.value,),
    ).fetchall()
    query = np.frombuffer(blob, dtype=np.float32)
    rows = [r for r in rows if len(r["vector"]) == len(blob)]  # same embedding model only
    if not rows:
        return None
    matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    scores = matrix @ query
    best = int(np.argmax(scores))  # the first of equal maxima, so the oldest
    return rows[best] if float(scores[best]) >= merge_at else None
