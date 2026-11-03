"""SQLite implementations of the store Protocols in ``agent_team.core.interfaces``.

Every row keeps the full model as JSON plus a few indexed columns. Rows are the
source of truth; artifact files under ``runs/<run>/artifacts`` are written for people
and tools to read.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import sqlite3
from collections import deque
from pathlib import Path

from pydantic import TypeAdapter

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.db import Database
from agent_team.core.errors import InvalidTransition
from agent_team.core.interfaces import EventBus
from agent_team.core.models import (
    Decision,
    Event,
    EventName,
    Feedback,
    Phase,
    Run,
    Task,
    TaskStatus,
    Verdict,
    can_transition,
    utcnow,
)
from agent_team.llm.types import Message, ToolResultBlock

_ARTIFACT = TypeAdapter[Artifact](Artifact)
_MESSAGES = TypeAdapter(list[Message])

#: Statuses that stamp ``finished_at``.
_FINISHED = frozenset({TaskStatus.DONE, TaskStatus.ESCALATED, TaskStatus.SUPERSEDED})


def _seq(table: str) -> str:
    """SQL for the table's next sequence number, so rows keep insert order."""
    return f"(SELECT COALESCE(MAX(seq), 0) + 1 FROM {table})"


# --- runs and tasks ----------------------------------------------------------------


class SqliteTaskStore:
    """Runs and tasks. Implements ``TaskStore``; status changes only through ``transition``."""

    def __init__(self, db: Database, bus: EventBus | None = None) -> None:
        """Use ``db``; ``bus`` gets task_state_changed events. The lock serialises transitions."""
        self.db = db
        self.bus = bus
        self._lock = asyncio.Lock()

    # runs

    async def create_run(self, run: Run) -> Run:
        """Insert a new run."""
        await self.db.execute(
            "INSERT INTO runs (id, status, created_at, data) VALUES (?, ?, ?, ?)",
            (run.id, run.status.value, run.created_at.isoformat(), run.model_dump_json()),
        )
        return run

    async def get_run(self, run_id: str) -> Run:
        """Load a run; raises KeyError if it doesn't exist."""
        row = await self.db.fetchone("SELECT data FROM runs WHERE id = ?", (run_id,))
        if row is None:
            raise KeyError(f"no run {run_id!r}")
        return Run.model_validate_json(row["data"])

    async def update_run(self, run: Run) -> Run:
        """Persist a run and stamp ``updated_at``."""
        await self.get_run(run.id)
        run = run.model_copy(update={"updated_at": utcnow()})
        await self.db.execute(
            "UPDATE runs SET status = ?, data = ? WHERE id = ?",
            (run.status.value, run.model_dump_json(), run.id),
        )
        return run

    async def list_runs(self) -> list[Run]:
        """Every run, oldest first."""
        rows = await self.db.fetchall("SELECT data FROM runs ORDER BY created_at, id")
        return [Run.model_validate_json(r["data"]) for r in rows]

    # tasks

    async def create_task(self, task: Task) -> Task:
        """Insert a new task."""
        await self.db.execute(
            "INSERT INTO tasks (id, run_id, status, phase, created_at, seq, data) "
            f"VALUES (?, ?, ?, ?, ?, {_seq('tasks')}, ?)",
            (
                task.id,
                task.run_id,
                task.status.value,
                task.phase.value,
                task.created_at.isoformat(),
                task.model_dump_json(),
            ),
        )
        return task

    async def get_task(self, task_id: str) -> Task:
        """Load a task; raises KeyError if it doesn't exist."""
        row = await self.db.fetchone("SELECT data FROM tasks WHERE id = ?", (task_id,))
        if row is None:
            raise KeyError(f"no task {task_id!r}")
        return Task.model_validate_json(row["data"])

    async def _put(self, task: Task) -> None:
        """Overwrite a stored task with its new state."""
        await self.db.execute(
            "UPDATE tasks SET status = ?, phase = ?, data = ? WHERE id = ?",
            (task.status.value, task.phase.value, task.model_dump_json(), task.id),
        )

    async def update_task(self, task: Task) -> Task:
        """Persist non-status fields; raises InvalidTransition if the status differs."""
        async with self._lock:
            current = await self.get_task(task.id)
            if current.status is not task.status:
                raise InvalidTransition(
                    f"update_task can't change status of {task.id} "
                    f"({current.status} -> {task.status}); use transition()"
                )
            await self._put(task)
        return task

    async def transition(self, task_id: str, to: TaskStatus, *, reason: str) -> Task:
        """Move a task to ``to`` if ALLOWED_TRANSITIONS permits, stamp its times and
        emit ``task_state_changed``."""
        async with self._lock:
            task = await self.get_task(task_id)
            src = task.status
            if not can_transition(src, to):
                raise InvalidTransition(f"{task_id}: {src} -> {to} is not allowed ({reason})")
            now = utcnow()
            update: dict[str, object] = {"status": to}
            if to is TaskStatus.RUNNING:
                update["started_at"] = now
                update["finished_at"] = None
            if to in _FINISHED:
                update["finished_at"] = now
            task = task.model_copy(update=update)
            await self._put(task)
        if self.bus is not None:
            await self.bus.emit(
                Event(
                    name=EventName.TASK_STATE_CHANGED,
                    run_id=task.run_id,
                    task_id=task.id,
                    attempt=task.attempt,
                    agent=task.agent,
                    trace_id=task.trace_id,
                    span_id=task.span_id,
                    data={"from": src.value, "to": to.value, "reason": reason},
                )
            )
        return task

    async def list_tasks(
        self,
        run_id: str,
        *,
        status: TaskStatus | None = None,
        phase: Phase | None = None,
    ) -> list[Task]:
        """A run's tasks in creation order, optionally filtered by status and phase."""
        sql = "SELECT data FROM tasks WHERE run_id = ?"
        params: list[str] = [run_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        if phase is not None:
            sql += " AND phase = ?"
            params.append(phase.value)
        rows = await self.db.fetchall(sql + " ORDER BY seq", params)
        return [Task.model_validate_json(r["data"]) for r in rows]

    async def ready_tasks(self, run_id: str) -> list[Task]:
        """Promote pending tasks whose dependencies are all done, then return every
        ready task in the run (newly promoted and already ready), oldest first."""
        tasks = await self.list_tasks(run_id)
        done = {t.id for t in tasks if t.status is TaskStatus.DONE}
        for task in tasks:
            if task.status is TaskStatus.PENDING and all(d in done for d in task.depends_on):
                # A concurrent caller may have promoted or superseded it already.
                with contextlib.suppress(InvalidTransition):
                    await self.transition(task.id, TaskStatus.READY, reason="dependencies done")
        return await self.list_tasks(run_id, status=TaskStatus.READY)


# --- artifacts ---------------------------------------------------------------------


class SqliteArtifactStore:
    """Versioned artifacts with lineage. Implements ``ArtifactStore``."""

    def __init__(self, db: Database, runs_dir: str | Path) -> None:
        """Use ``db`` for rows and ``runs_dir`` for the artifact JSON copies."""
        self.db = db
        self.runs_dir = Path(runs_dir)

    def path_for(self, run_id: str, ref: ArtifactRef) -> Path:
        """Where the JSON copy of one artifact version lives."""
        return self.runs_dir / run_id / "artifacts" / f"{ref.kind}.v{ref.version}.json"

    async def save(self, artifact: Artifact) -> ArtifactRef:
        """Assign the next version, store the row and lineage in one transaction, then
        write the JSON file. The version is read and claimed atomically, so two
        processes saving the same kind get different versions."""
        if not artifact.run_id:
            raise ValueError("artifact.run_id must be set before saving")
        run_id = artifact.run_id
        created = utcnow().isoformat()

        def insert(conn: sqlite3.Connection) -> tuple[ArtifactRef, str]:
            """Claim the next version and insert its rows in one transaction."""
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM artifacts WHERE run_id = ? AND kind = ?",
                (run_id, artifact.kind),
            ).fetchone()
            version = int(row[0]) + 1
            saved = artifact.model_copy(update={"version": version})
            ref = ArtifactRef(kind=saved.kind, version=version)
            body = saved.model_dump_json(indent=2)
            conn.execute(
                "INSERT INTO artifacts (run_id, kind, version, task_id, path, created_at, data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    ref.kind,
                    version,
                    saved.task_id,
                    str(self.path_for(run_id, ref)),
                    created,
                    body,
                ),
            )
            for parent in saved.built_from:
                conn.execute(
                    "INSERT OR IGNORE INTO lineage (run_id, child, parent) VALUES (?, ?, ?)",
                    (run_id, str(ref), str(parent)),
                )
            return ref, body

        ref, body = await self.db.transaction(insert)
        # The row is the source of truth; the file is a readable copy written after it.
        await asyncio.to_thread(_write_file, self.path_for(run_id, ref), body)
        return ref

    async def load(self, run_id: str, ref: ArtifactRef) -> Artifact:
        """Load one artifact version; raises KeyError if it doesn't exist."""
        row = await self.db.fetchone(
            "SELECT data FROM artifacts WHERE run_id = ? AND kind = ? AND version = ?",
            (run_id, ref.kind, ref.version),
        )
        if row is None:
            raise KeyError(f"no artifact {ref} in run {run_id!r}")
        return _ARTIFACT.validate_json(row["data"])

    async def latest(self, run_id: str, kind: str) -> Artifact | None:
        """The highest version of a kind in a run, or None."""
        row = await self.db.fetchone(
            "SELECT data FROM artifacts WHERE run_id = ? AND kind = ? "
            "ORDER BY version DESC LIMIT 1",
            (run_id, kind),
        )
        return None if row is None else _ARTIFACT.validate_json(row["data"])

    async def dependents(self, run_id: str, ref: ArtifactRef) -> list[ArtifactRef]:
        """Every artifact version built, directly or not, from ``ref``."""
        rows = await self.db.fetchall(
            "SELECT child, parent FROM lineage WHERE run_id = ?", (run_id,)
        )
        children: dict[str, list[str]] = {}
        for r in rows:
            children.setdefault(r["parent"], []).append(r["child"])
        seen: dict[str, None] = {}
        queue = deque([str(ref)])
        while queue:
            for child in children.get(queue.popleft(), []):
                if child not in seen:
                    seen[child] = None
                    queue.append(child)
        return [ArtifactRef.parse(c) for c in seen]


def _digest(text: str) -> str:
    """Short content hash used to name blob files."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _write_file(path: Path, body: str) -> None:
    """Write a temp file, then rename it over the target, so a crash leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(path)


# --- checkpoints -------------------------------------------------------------------


class SqliteCheckpointStore:
    """One row per (task, attempt, step). Tool results longer than ``inline_limit``
    characters are written to ``runs/<run>/blobs/`` and the row keeps the path and a
    preview."""

    def __init__(
        self,
        db: Database,
        runs_dir: str | Path,
        *,
        bus: EventBus | None = None,
        inline_limit: int = 20_000,
        preview_chars: int = 2_000,
    ) -> None:
        """Use ``db`` for rows and ``runs_dir`` for blobs of results over ``inline_limit``."""
        self.db = db
        self.runs_dir = Path(runs_dir)
        self.bus = bus
        self.inline_limit = inline_limit
        self.preview_chars = preview_chars

    async def _run_id(self, task_id: str) -> str:
        """The task's run id (it names the blob folder), or ``_unassigned`` if unknown."""
        row = await self.db.fetchone("SELECT run_id FROM tasks WHERE id = ?", (task_id,))
        return str(row["run_id"]) if row else "_unassigned"

    async def save_step(
        self, task_id: str, attempt: int, step: int, messages: list[Message]
    ) -> None:
        """Store the attempt's messages after one step, offloading large tool results."""
        run_id = await self._run_id(task_id)
        stored = [await self._offload(run_id, task_id, attempt, m) for m in messages]
        await self.db.execute(
            "INSERT OR REPLACE INTO checkpoints (task_id, attempt, step, created_at, messages) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, attempt, step, utcnow().isoformat(), _MESSAGES.dump_json(stored).decode()),
        )
        if self.bus is not None:
            await self.bus.emit(
                Event(
                    name=EventName.CHECKPOINT_SAVED,
                    run_id=run_id,
                    task_id=task_id,
                    attempt=attempt,
                    level="debug",
                    data={"step": step, "messages": len(messages)},
                )
            )

    async def _offload(self, run_id: str, task_id: str, attempt: int, message: Message) -> Message:
        """Move large tool results to blob files, leaving a preview and the path."""
        blocks = []
        changed = False
        for block in message.content:
            if (
                isinstance(block, ToolResultBlock)
                and block.stored_at is None
                and len(block.content) > self.inline_limit
            ):
                path = (
                    self.runs_dir
                    / run_id
                    / "blobs"
                    / f"{task_id}.a{attempt}.{block.tool_use_id}.{_digest(block.content)}.txt"
                )
                # The agent resaves its whole history every step. Naming by content
                # writes each result once and keeps reused tool call ids apart.
                if not await asyncio.to_thread(path.exists):
                    await asyncio.to_thread(_write_file, path, block.content)
                # This rewrites history the model has already seen. On resume, Claude's
                # thinking blocks after this point no longer match their history; with
                # on_history_edit set to drop_block the API drops them and the call
                # goes on (otherwise an enforced account gets a 400).
                preview = block.content[: min(self.preview_chars, self.inline_limit)]
                block = block.model_copy(
                    update={
                        "content": f"{preview}\n[... truncated; full result at {path}]",
                        "stored_at": str(path),
                    }
                )
                changed = True
            blocks.append(block)
        return message.model_copy(update={"content": blocks}) if changed else message

    async def load_latest(self, task_id: str, attempt: int) -> tuple[int, list[Message]] | None:
        """(step, messages) of the attempt's last saved step, or None."""
        row = await self.db.fetchone(
            "SELECT step, messages FROM checkpoints WHERE task_id = ? AND attempt = ? "
            "ORDER BY step DESC LIMIT 1",
            (task_id, attempt),
        )
        if row is None:
            return None
        return int(row["step"]), _MESSAGES.validate_json(row["messages"])


# --- decisions, feedback, verdicts -------------------------------------------------


class SqliteDecisionLog:
    """The orchestrator's decision log. Implements ``DecisionLog``."""

    def __init__(self, db: Database, bus: EventBus | None = None) -> None:
        """Use ``db``; ``bus`` gets a decision event per record."""
        self.db = db
        self.bus = bus

    async def record(self, decision: Decision) -> None:
        """Store a decision and emit a ``decision`` event."""
        await self.db.execute(
            f"INSERT INTO decisions (id, run_id, task_id, seq, data) VALUES (?, ?, ?, "
            f"{_seq('decisions')}, ?)",
            (decision.id, decision.run_id, decision.task_id, decision.model_dump_json()),
        )
        if self.bus is not None:
            await self.bus.emit(
                Event(
                    name=EventName.DECISION,
                    run_id=decision.run_id,
                    task_id=decision.task_id,
                    agent=None,
                    data=json.loads(decision.model_dump_json()),
                )
            )

    async def list(self, run_id: str) -> list[Decision]:
        """A run's decisions in the order they were made."""
        rows = await self.db.fetchall(
            "SELECT data FROM decisions WHERE run_id = ? ORDER BY seq", (run_id,)
        )
        return [Decision.model_validate_json(r["data"]) for r in rows]


class SqliteFeedbackStore:
    """User answers, comments and findings per artifact kind. Implements ``FeedbackStore``."""

    def __init__(self, db: Database) -> None:
        """Use ``db`` for feedback rows."""
        self.db = db

    async def add(self, run_id: str, feedback: Feedback) -> None:
        """Store one piece of feedback for a run."""
        kind = feedback.artifact.kind if feedback.artifact else None
        await self.db.execute(
            f"INSERT INTO feedback (id, run_id, artifact_kind, seq, data) VALUES (?, ?, ?, "
            f"{_seq('feedback')}, ?)",
            (feedback.id, run_id, kind, feedback.model_dump_json()),
        )

    async def for_artifact(self, run_id: str, kind: str) -> list[Feedback]:
        """Feedback on one artifact kind, oldest first."""
        rows = await self.db.fetchall(
            "SELECT data FROM feedback WHERE run_id = ? AND artifact_kind = ? ORDER BY seq",
            (run_id, kind),
        )
        return [Feedback.model_validate_json(r["data"]) for r in rows]


class SqliteVerdictStore:
    """Audit verdicts per task and for the whole run. Implements ``VerdictStore``."""

    def __init__(self, db: Database) -> None:
        """Use ``db`` for verdict rows."""
        self.db = db

    async def save(self, verdict: Verdict) -> None:
        """Store a verdict."""
        await self.db.execute(
            f"INSERT INTO verdicts (id, run_id, task_id, scope, phase, seq, data) "
            f"VALUES (?, ?, ?, ?, ?, {_seq('verdicts')}, ?)",
            (
                verdict.id,
                verdict.run_id,
                verdict.task_id,
                verdict.scope.value,
                verdict.phase.value if verdict.phase else None,
                verdict.model_dump_json(),
            ),
        )

    async def latest_for_task(self, task_id: str) -> Verdict | None:
        """The task's most recent verdict, or None."""
        row = await self.db.fetchone(
            "SELECT data FROM verdicts WHERE task_id = ? ORDER BY seq DESC LIMIT 1", (task_id,)
        )
        return None if row is None else Verdict.model_validate_json(row["data"])

    async def for_task(self, task_id: str) -> list[Verdict]:
        """Every verdict for the task, oldest first."""
        rows = await self.db.fetchall(
            "SELECT data FROM verdicts WHERE task_id = ? ORDER BY seq", (task_id,)
        )
        return [Verdict.model_validate_json(r["data"]) for r in rows]
