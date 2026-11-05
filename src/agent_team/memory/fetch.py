"""Searching memory, and the automatic fetch before each task (docs/DESIGN.md,
"Memory", steps 1 to 7).

``MemorySearch`` embeds a query, asks the index for keyword and vector matches and
ranks them; with the embedder down it ranks on BM25 alone and says so. ``MemoryFetch``
is the dispatcher's memory hook: it builds the task's fixed query, keeps the kinds the
agent uses, fits the result in the caps, logs a ``memory_fetched`` event with the query
and ids, and records the fetch so the memories can be credited when the task passes.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime

from agent_team.core.artifacts import Artifact
from agent_team.core.config import MemoryConfig
from agent_team.core.interfaces import EventBus, MemoryIndex
from agent_team.core.models import Event, EventName, MemoryKind, Run, Task, utcnow
from agent_team.memory.embed import Embeddings
from agent_team.memory.rank import Mode, Ranked, build_query, fit, rank

#: Memories saved without a vector that one search embeds, oldest first.
BACKFILL = 8


@dataclass
class Found:
    """What one search found: the ranked memories, how it matched, and why vectors
    were missing when they were."""

    ranked: list[Ranked] = field(default_factory=list)
    mode: Mode = "hybrid"
    embedder_error: str | None = None


class MemorySearch:
    """Hybrid search over the memory index, ranked by the ``memory:`` config."""

    def __init__(self, index: MemoryIndex, embeddings: Embeddings, config: MemoryConfig) -> None:
        """Search ``index``, embedding queries with ``embeddings``."""
        self.index = index
        self.embeddings = embeddings
        self.config = config

    async def search(
        self,
        text: str,
        kinds: list[MemoryKind],
        *,
        run_id: str | None,
        now: datetime | None = None,
    ) -> Found:
        """Memories of ``kinds`` matching ``text``, best first. ``run_id`` is the run
        asking: decisions and facts from other runs need a strong match."""
        vectors, error = await self.embeddings.vectors([text], query=True)
        if vectors is not None:
            await self._backfill()
        vector = vectors[0] if vectors else None
        matches = await self.index.search(text, vector, kinds, limit=self.config.candidates)
        ranked = rank(matches, self.config, now=now or utcnow(), run_id=run_id)
        return Found(ranked=ranked, mode="hybrid" if vector else "bm25", embedder_error=error)

    async def _backfill(self) -> None:
        """Embed a few memories saved while the embedder was down."""
        missing = await self.index.without_vectors(BACKFILL)
        if not missing:
            return
        vectors, _ = await self.embeddings.vectors([m.text for m in missing], query=False)
        for memory, vector in zip(missing, vectors or [], strict=False):
            await self.index.set_vector(memory.id, vector)


class MemoryFetch:
    """The dispatcher's memory hook: the "Relevant memory" lines for one task."""

    def __init__(
        self, search: MemorySearch, index: MemoryIndex, bus: EventBus, config: MemoryConfig
    ) -> None:
        """Fetch with ``search``, record fetches in ``index`` and log them on ``bus``."""
        self.search = search
        self.index = index
        self.bus = bus
        self.config = config

    async def __call__(self, task: Task, run: Run, inputs: list[Artifact]) -> list[str]:
        """The lines for ``task``'s prompt. A failure is logged and gives no memories,
        so memory never fails a task."""
        kinds = self.config.kinds.get(task.agent, [])
        if not kinds or self.config.max_items == 0:
            return []
        query = build_query(task, inputs)
        try:
            found = await self.search.search(query, kinds, run_id=run.id)
            chosen = fit(
                found.ranked,
                run_id=run.id,
                max_items=self.config.max_items,
                max_tokens=self.config.max_tokens,
            )
            ids = [r.memory.id for r, _ in chosen]
            await self.index.record_fetch(run.id, task.id, task.attempt, query, ids)
        except Exception as exc:
            await self._emit(
                task,
                {"query": query, "ids": [], "error": f"{type(exc).__name__}: {exc}"},
                level="warning",
                trace=traceback.format_exc(),
            )
            return []
        await self._emit(
            task,
            {
                "query": query,
                "kinds": [k.value for k in kinds],
                "ids": ids,
                "scores": {r.memory.id: round(r.score, 4) for r, _ in chosen},
                "mode": found.mode,
                "embedder_error": found.embedder_error,
            },
            level="warning" if found.embedder_error else "info",
        )
        return [text for _, text in chosen]

    async def _emit(
        self, task: Task, data: dict[str, object], *, level: str, trace: str | None = None
    ) -> None:
        """Log one fetch on the task's attempt span."""
        if trace:
            data["traceback"] = trace
        await self.bus.emit(
            Event(
                name=EventName.MEMORY_FETCHED,
                run_id=task.run_id,
                task_id=task.id,
                attempt=task.attempt,
                agent=task.agent,
                trace_id=task.trace_id,
                span_id=task.span_id,
                level="warning" if level == "warning" else "info",
                data=data,
            )
        )
