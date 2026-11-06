"""The SQLite memory index: saving, merging near-duplicates, hybrid search, the fetch
log and usefulness credit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_team.core.db import Database
from agent_team.core.models import Memory, MemoryKind, MemorySource
from agent_team.memory.embed import FakeEmbedder
from agent_team.memory.store import SqliteMemoryIndex, match_query, normalize

T0 = datetime(2026, 11, 1, tzinfo=UTC)
EMBED = FakeEmbedder()


def mem(
    text: str,
    kind: MemoryKind = MemoryKind.LESSON,
    *,
    run: str = "run_1",
    task: str | None = "task_1",
    ref: str | None = None,
    age_days: float = 0,
    id: str | None = None,
) -> Memory:
    """A memory saved ``age_days`` before T0."""
    extra = {"id": id} if id else {}
    return Memory(
        kind=kind,
        text=text,
        sources=[MemorySource(run_id=run, task_id=task, ref=ref)],
        created_at=T0 - timedelta(days=age_days),
        **extra,
    )


async def vec(text: str) -> list[float]:
    """The fake embedder's vector for a memory's text."""
    return (await EMBED.embed([text], query=False))[0]


async def index(tmp_path: Path) -> SqliteMemoryIndex:
    """A fresh index in its own database file."""
    return await SqliteMemoryIndex.open(Database(tmp_path / "team.db"))


async def test_the_tables_join_an_existing_run_database(tmp_path: Path) -> None:
    db = Database(tmp_path / "team.db")  # the run tables, written before memory existed
    first = await SqliteMemoryIndex.open(db)
    await first.save(mem("pytest needs httpx"), None, merge_at=0.92)
    again = await SqliteMemoryIndex.open(db)  # opening twice keeps what was there
    assert [m.text for m in await again.memories()] == ["pytest needs httpx"]


async def test_the_same_words_merge_and_keep_the_older_id_and_source(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    old, merged = await idx.save(mem("FastAPI tests need httpx."), None, merge_at=0.92)
    assert not merged
    new = mem("fastapi TESTS need httpx", run="run_2", task="task_9")
    kept, merged = await idx.save(new, None, merge_at=0.92)
    assert merged and kept.id == old.id and kept.text == old.text
    assert [s.run_id for s in kept.sources] == ["run_1", "run_2"]
    # Saving the same source again adds nothing.
    kept, _ = await idx.save(new, None, merge_at=0.92)
    assert len(kept.sources) == 2
    assert len(await idx.memories()) == 1


async def test_a_close_vector_merges_and_a_far_one_does_not(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    text = "Engineer implementation tests need httpx installed for FastAPI"
    first, _ = await idx.save(mem(text), await vec(text), merge_at=0.92)
    close = text + " here"  # one more word: cosine about 0.95
    kept, merged = await idx.save(mem(close, run="run_2"), await vec(close), merge_at=0.92)
    assert merged and kept.id == first.id
    far = "QA reads the acceptance criteria before testing"
    _, merged = await idx.save(mem(far, run="run_2"), await vec(far), merge_at=0.92)
    assert not merged
    # Another kind never merges, however close.
    _, merged = await idx.save(
        mem(text, MemoryKind.DECISION, run="run_3"), await vec(text), merge_at=0.92
    )
    assert not merged
    assert len(await idx.memories()) == 3


async def test_search_scores_keywords_and_meaning(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    texts = [
        "FastAPI tests need httpx installed",
        "Use SQLite for the mock database",
        "The user wants type hints and pytest",
        "Researcher opened the Python docs",
    ]
    for i, t in enumerate(texts):
        await idx.save(mem(t, id=f"mem_{i}"), await vec(t), merge_at=0.92)
    query = "engineer tests for the FastAPI endpoints"
    q = (await EMBED.embed([query], query=True))[0]
    found = await idx.search(query, q, [MemoryKind.LESSON], limit=10)
    by_id = {m.memory.id: m for m in found}
    assert by_id["mem_0"].bm25 > 0 and by_id["mem_0"].cosine is not None
    assert by_id["mem_0"].cosine > by_id["mem_1"].cosine  # type: ignore[operator]
    # Keyword search alone without a vector, and only the kinds asked for.
    keyword = await idx.search(query, None, [MemoryKind.LESSON], limit=10)
    assert [m.memory.id for m in keyword] == ["mem_0"] and keyword[0].cosine is None
    assert await idx.search(query, q, [MemoryKind.DECISION], limit=10) == []


async def test_the_same_query_gives_the_same_matches(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    for i in range(6):
        t = f"lesson {i}: tests need httpx"  # same words, different ids: ties everywhere
        await idx.save(mem(t, id=f"mem_{i}", run=f"run_{i}"), await vec(t), merge_at=1.01)
    q = (await EMBED.embed(["tests httpx"], query=True))[0]
    first = await idx.search("tests httpx", q, list(MemoryKind), limit=3)
    for _ in range(3):
        assert await idx.search("tests httpx", q, list(MemoryKind), limit=3) == first


async def test_fts_syntax_in_a_query_is_only_words() -> None:
    assert match_query('NEAR(fastapi "tests") AND -httpx* x') == (
        '"near" OR "fastapi" OR "tests" OR "httpx"'
    )
    assert match_query("the a of") is None
    assert normalize("  FastAPI, tests!! ") == "fastapi tests"


async def test_memories_saved_without_a_vector_can_get_one_later(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    saved, _ = await idx.save(mem("tests need httpx"), None, merge_at=0.92)
    assert [m.id for m in await idx.without_vectors(5)] == [saved.id]
    await idx.set_vector(saved.id, await vec("tests need httpx"))
    assert await idx.without_vectors(5) == []
    q = (await EMBED.embed(["tests need httpx"], query=True))[0]
    [match] = await idx.search("unrelated words", q, [MemoryKind.LESSON], limit=5)
    assert match.cosine is not None and match.cosine > 0.99


async def test_credit_counts_once_per_task_and_only_the_first_attempt(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    a, _ = await idx.save(mem("a lesson", id="mem_a"), None, merge_at=0.92)
    b, _ = await idx.save(mem("another one", id="mem_b"), None, merge_at=0.92)
    await idx.record_fetch("run_1", "task_1", 1, "q", [a.id])
    await idx.record_fetch("run_1", "task_1", 1, "q", [a.id, b.id])  # a resumed attempt
    await idx.record_fetch("run_1", "task_2", 2, "q", [a.id])
    assert sorted(await idx.credit("task_1")) == [a.id, b.id]
    assert await idx.credit("task_1") == []  # already credited
    assert await idx.credit("task_2") == []  # its fetch was on a retry
    assert {m.id: m.usefulness for m in await idx.memories()} == {a.id: 1, b.id: 1}


async def test_a_tiny_store_still_matches_by_keywords(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    await idx.save(mem("the books API needs a test run", id="mem_a"), None, merge_at=0.92)
    await idx.save(mem("books API on SQLite", id="mem_b"), None, merge_at=0.92)
    found = {
        m.memory.id: m
        for m in await idx.search("test runs for the books API", None, [MemoryKind.LESSON], limit=5)
    }
    assert found["mem_a"].bm25 < 0.001  # every term is in half the memories
    assert found["mem_a"].words > found["mem_b"].words > 0


async def test_forget_removes_a_memory_from_both_searches(tmp_path: Path) -> None:
    idx = await index(tmp_path)
    bad, _ = await idx.save(mem("rule S1 caught a credential in web_fetch"), None, merge_at=0.92)
    keep, _ = await idx.save(mem("pytest needs httpx"), None, merge_at=0.92)
    gone = await idx.forget(bad.id)
    assert gone is not None and gone.text == bad.text
    assert [m.id for m in await idx.memories()] == [keep.id]
    found = await idx.search("credential web_fetch", None, list(MemoryKind), limit=10)
    assert found == []
    assert await idx.forget(bad.id) is None
