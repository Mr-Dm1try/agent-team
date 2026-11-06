"""Embeddings, search and the automatic fetch: the fallback to BM25 when the embedder
is down, the kinds and caps of the fetch, and its trace."""

from __future__ import annotations

from pathlib import Path

import pytest
from agents.fakes import FakeBus

from agent_team.core.config import EmbedderConfig, MemoryConfig
from agent_team.core.db import Database
from agent_team.core.errors import EmbedderUnavailable
from agent_team.core.models import (
    AgentName,
    EventName,
    Memory,
    MemoryKind,
    MemorySource,
    Phase,
    Run,
    Task,
)
from agent_team.memory.embed import Embeddings, FakeEmbedder, OllamaEmbedder
from agent_team.memory.fetch import MemoryFetch, MemorySearch
from agent_team.memory.store import SqliteMemoryIndex

LESSON = "Engineer, implementation: tests for the books endpoints need httpx installed"
OTHERS = [
    (MemoryKind.LESSON, "QA, implementation: read every acceptance criterion first"),
    (MemoryKind.DECISION, "Use SQLite for the books database"),
    (MemoryKind.PREFERENCE, "The user wants type hints"),
    (MemoryKind.RESEARCH_FACT, "httpx is the test client FastAPI documents"),
    (MemoryKind.RUN_SUMMARY, "Built a books API with two tests"),
]


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """The current time."""
        return self.now


async def test_the_switch_skips_a_down_embedder_until_it_may_be_back() -> None:
    fake, clock = FakeEmbedder(fail=True), Clock()
    embeddings = Embeddings(fake, retry_after_s=60, clock=clock)
    vectors, why = await embeddings.vectors(["x"], query=True)
    assert vectors is None and why and "down" in why
    fake.fail = False
    clock.now = 30
    assert (await embeddings.vectors(["x"], query=True))[0] is None  # still waiting
    assert fake.calls == 1
    clock.now = 61
    vectors, why = await embeddings.vectors(["x"], query=True)
    assert vectors is not None and why is None and fake.calls == 2


async def test_ollama_down_is_embedder_unavailable() -> None:
    config = EmbedderConfig(timeout_s=2)
    embedder = OllamaEmbedder(config, "http://127.0.0.1:9")  # nothing listens on 9
    with pytest.raises(EmbedderUnavailable):
        await embedder.embed(["hello"], query=True)


async def test_the_fake_embedder_is_deterministic() -> None:
    a, b = FakeEmbedder(), FakeEmbedder()
    assert await a.embed(["tests need httpx"], query=True) == await b.embed(
        ["tests need httpx"], query=False
    )


async def _filled(tmp_path: Path, embedder: FakeEmbedder) -> SqliteMemoryIndex:
    """An index with the lesson and one memory of each other kind, from run_old."""
    idx = await SqliteMemoryIndex.open(Database(tmp_path / "team.db"))
    for kind, text in [(MemoryKind.LESSON, LESSON), *OTHERS]:
        vectors = None if embedder.fail else await embedder.embed([text], query=False)
        memory = Memory(kind=kind, text=text, sources=[MemorySource(run_id="run_old")])
        await idx.save(memory, vectors[0] if vectors else None, merge_at=0.92)
    return idx


def _task(agent: AgentName = AgentName.ENGINEER) -> Task:
    """A task for the books endpoints in run_new."""
    return Task(
        run_id="run_new",
        phase=Phase.IMPLEMENTATION,
        agent=agent,
        objective="Build the books endpoints with tests",
        output_kind="code_change",
        attempt=1,
    )


RUN = Run(id="run_new", title="books", request="Build a books API", workspace="/tmp/ws")


def _fetch(idx: SqliteMemoryIndex, embedder: FakeEmbedder, **config: object) -> MemoryFetch:
    """A fetch over ``idx`` with the config's defaults, changed by ``config``."""
    settings = MemoryConfig(min_relevance=0.1, **config)  # type: ignore[arg-type]
    search = MemorySearch(idx, Embeddings(embedder, retry_after_s=60), settings)
    return MemoryFetch(search, idx, FakeBus(), settings)


async def test_the_fetch_keeps_the_agents_kinds_and_logs_the_query(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    idx = await _filled(tmp_path, embedder)
    fetch, task = _fetch(idx, embedder), _task()
    lines = await fetch(task, RUN, [])
    assert lines[0].startswith('Lesson: "Engineer, implementation: tests for the books')
    assert "(run run_old)" in lines[0]
    assert not any(line.startswith(("Run summary", "Research fact")) for line in lines)
    [event] = fetch.bus.events  # type: ignore[attr-defined]
    assert event.name is EventName.MEMORY_FETCHED and event.task_id == task.id
    assert event.data["query"] == "engineer, implementation, Build the books endpoints with tests"
    assert event.data["mode"] == "hybrid" and event.data["embedder_error"] is None
    assert len(event.data["ids"]) == len(lines)
    assert event.data["kinds"] == ["lesson", "decision", "preference"]


async def test_the_fetch_respects_the_item_and_token_caps(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    idx = await _filled(tmp_path, embedder)
    one = await _fetch(idx, embedder, max_items=1)(_task(), RUN, [])
    assert len(one) == 1
    tight = await _fetch(idx, embedder, max_tokens=5)(_task(), RUN, [])
    assert tight == []
    none = await _fetch(idx, embedder, kinds={AgentName.ENGINEER: []})(_task(), RUN, [])
    assert none == []


async def test_a_down_embedder_falls_back_to_bm25_and_says_so(tmp_path: Path) -> None:
    embedder = FakeEmbedder(fail=True)
    idx = await _filled(tmp_path, embedder)  # saved without vectors
    fetch = _fetch(idx, embedder)
    lines = await fetch(_task(), RUN, [])
    assert lines and lines[0].startswith('Lesson: "Engineer, implementation')
    [event] = fetch.bus.events  # type: ignore[attr-defined]
    assert event.data["mode"] == "bm25" and "down" in event.data["embedder_error"]
    assert event.level == "warning"


async def test_vectors_are_filled_in_once_the_embedder_is_back(tmp_path: Path) -> None:
    down = FakeEmbedder(fail=True)
    idx = await _filled(tmp_path, down)
    assert len(await idx.without_vectors(10)) == 6
    await _fetch(idx, FakeEmbedder())(_task(), RUN, [])
    assert await idx.without_vectors(10) == []


async def test_a_failing_fetch_gives_no_memory_and_logs_why(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    idx = await _filled(tmp_path, embedder)
    fetch = _fetch(idx, embedder)

    async def broken(*args: object, **kwargs: object) -> None:
        """The index fell over."""
        raise RuntimeError("disk on fire")

    idx.record_fetch = broken  # type: ignore[method-assign]
    assert await fetch(_task(), RUN, []) == []
    [event] = fetch.bus.events  # type: ignore[attr-defined]
    assert "disk on fire" in event.data["error"] and event.level == "warning"
