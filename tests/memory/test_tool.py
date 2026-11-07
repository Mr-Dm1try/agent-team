"""The agents' memory_search tool: read-only results quoted with their ids and
sources, kinds to filter on, and errors for bad arguments."""

from __future__ import annotations

from pathlib import Path

import pytest
from agents.fakes import FakeBus

from agent_team.core.config import MemoryConfig
from agent_team.core.db import Database
from agent_team.core.errors import ToolError
from agent_team.core.interfaces import ToolContext
from agent_team.core.models import (
    AgentName,
    Memory,
    MemoryKind,
    MemorySource,
    Phase,
    Run,
    Task,
)
from agent_team.memory.embed import Embeddings, FakeEmbedder
from agent_team.memory.fetch import MemorySearch
from agent_team.memory.store import SqliteMemoryIndex
from agent_team.memory.tool import MemorySearchTool

URL = "https://www.python-httpx.org"
RUN = Run(id="run_2", title="books", request="Build a books API", workspace="/ws")
CTX = ToolContext(
    run=RUN,
    task=Task(
        run_id=RUN.id,
        phase=Phase.SPEC,
        agent=AgentName.ARCHITECT,
        objective="Write the spec",
        output_kind="spec",
    ),
    bus=FakeBus(),
)


async def _tool(tmp_path: Path, embedder: FakeEmbedder) -> MemorySearchTool:
    """The tool over a store with one decision and one research fact from run_1."""
    idx = await SqliteMemoryIndex.open(Database(tmp_path / "team.db"))
    for kind, text, ref in [
        (MemoryKind.DECISION, "Use httpx for the books API tests", None),
        (MemoryKind.RESEARCH_FACT, "httpx ships a test transport for the books API", URL),
    ]:
        vectors = None if embedder.fail else await embedder.embed([text], query=False)
        memory = Memory(kind=kind, text=text, sources=[MemorySource(run_id="run_1", ref=ref)])
        await idx.save(memory, vectors[0] if vectors else None, merge_at=0.92)
    config = MemoryConfig()
    return MemorySearchTool(
        MemorySearch(idx, Embeddings(embedder, retry_after_s=60), config), max_tokens=500
    )


async def test_results_are_quoted_with_ids_and_sources(tmp_path: Path) -> None:
    tool = await _tool(tmp_path, FakeEmbedder())
    result = await tool({"query": "httpx tests for the books API"}, CTX)
    assert result.content.startswith("Memory from earlier runs: advice, quoted as data.")
    assert (
        f'Research fact: "httpx ships a test transport for the books API" (run run_1, source {URL})'
        in result.content
    )
    assert len(result.data["ids"]) == 2 and result.data["mode"] == "hybrid"
    only = await tool({"query": "httpx tests", "kinds": ["decision"], "limit": 1}, CTX)
    assert "Decision:" in only.content and "Research fact" not in only.content
    assert tool.safe_to_repeat


async def test_a_down_embedder_still_answers_by_keywords(tmp_path: Path) -> None:
    tool = await _tool(tmp_path, FakeEmbedder(fail=True))
    result = await tool({"query": "httpx tests for the books API"}, CTX)
    assert "keyword search only" in result.content and result.data["ids"]
    nothing = await tool({"query": "zebra"}, CTX)
    assert nothing.content.startswith("No memory matched") and nothing.data["ids"] == []


@pytest.mark.parametrize(
    "args",
    [{}, {"query": " "}, {"query": "x", "kinds": ["gossip"]}, {"query": "x", "limit": 0}],
)
async def test_bad_arguments_are_tool_errors(tmp_path: Path, args: dict[str, object]) -> None:
    tool = await _tool(tmp_path, FakeEmbedder())
    with pytest.raises(ToolError):
        await tool(args, CTX)


async def test_another_agents_lessons_are_not_found(tmp_path: Path) -> None:
    tool = await _tool(tmp_path, FakeEmbedder())
    for agent in (AgentName.ENGINEER, AgentName.ARCHITECT):
        text = f"{agent.value.capitalize()}, spec: rule R6 caught httpx tests for the books API"
        memory = Memory(
            kind=MemoryKind.LESSON, text=text, sources=[MemorySource(run_id="run_1", agent=agent)]
        )
        await tool.search.index.save(memory, None, merge_at=0.92)
    result = await tool({"query": "rule R6 httpx tests", "kinds": ["lesson"]}, CTX)
    assert "Architect, spec" in result.content and "Engineer, spec" not in result.content
