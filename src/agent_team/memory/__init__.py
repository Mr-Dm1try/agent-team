"""Long-term memory (docs/DESIGN.md, "Memory"): the store, embeddings, ranking, the
automatic fetch before each task, the writer hook and the ``memory_search`` tool, built
from the ``memory:`` config by ``build_memory``."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_team.core.config import Config
from agent_team.core.db import Database
from agent_team.core.interfaces import (
    ArtifactStore,
    Embedder,
    EventBus,
    TaskStore,
    Tool,
    VerdictStore,
)
from agent_team.llm.types import LLMClient
from agent_team.memory.embed import Embeddings, OllamaEmbedder
from agent_team.memory.fetch import MemoryFetch, MemorySearch
from agent_team.memory.store import SqliteMemoryIndex
from agent_team.memory.tool import MemorySearchTool
from agent_team.memory.writer import MemoryWriter


@dataclass
class MemoryParts:
    """Memory as the app uses it: the index, search, the dispatcher's fetch hook, the
    writer hook and the agents' tool."""

    db: Database
    index: SqliteMemoryIndex
    search: MemorySearch
    fetch: MemoryFetch | None = None
    writer: MemoryWriter | None = None
    tools: dict[str, Tool] = field(default_factory=dict)

    def close(self) -> None:
        """Close the memory's database connection."""
        self.db.close()


async def open_search(config: Config, *, embedder: Embedder | None = None) -> MemoryParts:
    """The index and search alone, over the configured database, for the CLI."""
    db = Database(config.paths.db)
    try:
        index = await SqliteMemoryIndex.open(db)
    except BaseException:
        db.close()
        raise
    embedder = embedder or OllamaEmbedder(config.memory.embedder, config.providers.ollama.base_url)
    settings = config.memory.embedder
    embeddings = Embeddings(
        embedder, retry_after_s=settings.retry_after_s, timeout_s=settings.timeout_s
    )
    return MemoryParts(db=db, index=index, search=MemorySearch(index, embeddings, config.memory))


async def build_memory(
    config: Config,
    *,
    bus: EventBus,
    llm: LLMClient,
    tasks: TaskStore,
    artifacts: ArtifactStore,
    verdicts: VerdictStore,
    embedder: Embedder | None = None,
) -> MemoryParts:
    """Everything memory needs, with the writer attached to ``bus``. ``embedder``
    replaces the Ollama one (tests use FakeEmbedder)."""
    parts = await open_search(config, embedder=embedder)
    parts.fetch = MemoryFetch(parts.search, parts.index, bus, config.memory)
    parts.writer = MemoryWriter(
        index=parts.index,
        embeddings=parts.search.embeddings,
        tasks=tasks,
        artifacts=artifacts,
        verdicts=verdicts,
        llm=llm,
        bus=bus,
        config=config.memory,
    )
    parts.writer.attach(bus)
    tool = MemorySearchTool(parts.search, max_tokens=config.memory.max_tokens)
    parts.tools = {tool.spec.name: tool}
    return parts
