"""Embeddings, search and the automatic fetch: the fallback to BM25 when the embedder
is down, the kinds and caps of the fetch, and its trace."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from agents.fakes import FakeBus

from agent_team.core.config import EmbedderConfig, MemoryConfig
from agent_team.core.db import Database
from agent_team.core.errors import EmbedderUnavailable
from agent_team.core.models import (
    SESSION,
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
from agent_team.memory.fetch import MAX_QUERY, QUOTE, MemoryFetch, MemorySearch, earlier_run
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
    # The memories from other runs are named, with their kind and run, for the review.
    earlier = event.data["from_other_runs"]
    assert earlier and all(m["run_id"] != RUN.id for m in earlier)
    assert {m["id"] for m in earlier} <= set(event.data["ids"])
    assert earlier[0]["kind"] == "lesson" and earlier[0]["run_id"] == "run_old"


async def test_the_users_own_preferences_arent_from_other_runs(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    idx = await SqliteMemoryIndex.open(Database(tmp_path / "team.db"))
    text = "Engineer, implementation: always build the books endpoints with tests"
    vectors = await embedder.embed([text], query=False)
    memory = Memory(kind=MemoryKind.PREFERENCE, text=text, sources=[MemorySource(run_id=SESSION)])
    await idx.save(memory, vectors[0], merge_at=0.92)
    fetch = _fetch(idx, embedder)
    assert await fetch(_task(), RUN, [])
    [event] = fetch.bus.events  # type: ignore[attr-defined]
    assert event.data["ids"] == [memory.id] and event.data["from_other_runs"] == []


def test_a_long_memory_is_quoted_short_in_the_event() -> None:
    memory = Memory(
        kind=MemoryKind.DECISION, text="word " * 100, sources=[MemorySource(run_id="r")]
    )
    quoted = earlier_run(memory)
    assert len(quoted["text"]) == QUOTE and quoted["text"].endswith("…")
    assert quoted == {"id": memory.id, "kind": "decision", "run_id": "r", "text": quoted["text"]}


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


async def test_a_long_query_is_cut_before_it_is_embedded(tmp_path: Path) -> None:
    class Recording(FakeEmbedder):
        """The fake embedder, keeping what it was asked to embed."""

        def __init__(self) -> None:
            """Start with nothing seen."""
            super().__init__()
            self.seen: list[str] = []

        async def embed(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
            """Record, then embed as the fake does."""
            self.seen += list(texts)
            return await super().embed(texts, query=query)

    embedder = Recording()
    idx = await SqliteMemoryIndex.open(Database(tmp_path / "team.db"))
    search = MemorySearch(idx, Embeddings(embedder, retry_after_s=60), MemoryConfig())
    await search.search("books " * 5_000, [MemoryKind.LESSON], run_id=None)
    assert len(embedder.seen[0]) == MAX_QUERY


async def test_another_projects_decisions_need_more_than_one_shared_word(tmp_path: Path) -> None:
    # Memories from a library-API run, fetched for a vault MCP server run.
    embedder = FakeEmbedder(fail=True)  # keyword match only, where BM25 counts most
    idx = await SqliteMemoryIndex.open(Database(tmp_path / "team.db"))
    old = [
        (MemoryKind.DECISION, "Use FastAPI with SQLite for the library inventory API"),
        (MemoryKind.DECISION, "Store books in one SQLite table keyed by ISBN"),
        (MemoryKind.RESEARCH_FACT, "Python sqlite3 needs check_same_thread=False across threads"),
        (MemoryKind.LESSON, "Engineer, implementation: run pytest as python -m pytest"),
        (MemoryKind.RESEARCH_FACT, "Pydantic v2 uses model_dump instead of dict"),
        (MemoryKind.PREFERENCE, "The user wants type hints everywhere"),
        (MemoryKind.PREFERENCE, "Keep commits small"),
        (MemoryKind.RUN_SUMMARY, "Built the inventory service with 73 tests"),
        (MemoryKind.LESSON, "QA, implementation: start the server and call each endpoint"),
        (MemoryKind.DECISION, "Return 404 for a missing book"),
        (MemoryKind.RESEARCH_FACT, "httpx is the test client FastAPI documents"),
    ]
    for kind, text in old:
        memory = Memory(kind=kind, text=text, sources=[MemorySource(run_id="run_old")])
        await idx.save(memory, None, merge_at=0.92)
    vault = Run(id="run_vault", title="vault", request="An MCP vault server", workspace="/w")
    architect = Task(
        run_id=vault.id,
        phase=Phase.SPEC,
        agent=AgentName.ARCHITECT,
        objective="Which Python MCP SDK version and API should the server use to expose "
        "list, read and search tools over Markdown notes?",
        output_kind="spec",
        attempt=1,
    )
    settings = MemoryConfig()
    search = MemorySearch(idx, Embeddings(embedder, retry_after_s=60), settings)
    lines = await MemoryFetch(search, idx, FakeBus(), settings)(architect, vault, [])
    assert not [line for line in lines if line.startswith(("Decision", "Research fact"))], lines
