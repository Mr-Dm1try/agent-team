"""Ranking memories for one task (docs/DESIGN.md, "Memory", steps 1, 2, 4, 5 and 6).

Everything here is a pure function of its inputs, so the same store, query and clock
give the same memories in the same order.

A memory's text match (``relevance``, 0 to 1) mixes its cosine with the query and its
keyword score; with no vectors (the embedder is down, or the memory was saved while it
was) it is the keyword score alone. The keyword score is BM25, squashed to 0 to 1 as
``s / (1 + s)``, or the share of words in common when that is higher: BM25 gives a
term no weight when it is in half the memories or more, which with a handful of
memories is nearly every term. A decision or research fact from another run must
pass ``strong_match`` on a stricter match (``strong_relevance``) that takes the words
in common, not BM25: in a small store one rare shared word, such as "API", gives BM25
a squashed score above 0.6 on its own. Its ``score``
adds recency, which halves every ``recency_half_life_days``, and usefulness, which
grows as ``u / (u + 2)``. The weights come from the ``memory:`` config.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from agent_team.core.artifacts import PRD, Artifact
from agent_team.core.config import MemoryConfig
from agent_team.core.models import SESSION, AgentName, Memory, MemoryKind, MemoryMatch, Task
from agent_team.llm.tokens import estimate_tokens

#: Kinds that hold for one project, so a match from another run must be strong.
RUN_BOUND = frozenset({MemoryKind.DECISION, MemoryKind.RESEARCH_FACT})
LABELS = {
    MemoryKind.RUN_SUMMARY: "Run summary",
    MemoryKind.DECISION: "Decision",
    MemoryKind.LESSON: "Lesson",
    MemoryKind.PREFERENCE: "Preference",
    MemoryKind.RESEARCH_FACT: "Research fact",
}
#: How a search matched: by meaning and keywords, or by keywords alone.
Mode = Literal["hybrid", "bm25"]
#: Sources named on one memory's line; a memory merged from many runs names the first.
MAX_REFS = 2
#: Longest input title used in a query.
TITLE_CHARS = 80
_SENTENCE = re.compile(r"(?<=[.!?])\s")


@dataclass(frozen=True)
class Ranked:
    """A memory with its text match and final score."""

    memory: Memory
    relevance: float
    score: float


def title_of(artifact: Artifact) -> str:
    """An input's title for the query: the PRD's title, else the first sentence of
    the artifact's summary."""
    if isinstance(artifact, PRD):
        return artifact.title
    first = _SENTENCE.split(artifact.summary.strip(), maxsplit=1)[0]
    return first[:TITLE_CHARS].rstrip(" .")


def build_query(task: Task, inputs: list[Artifact]) -> str:
    """The fixed query for a task, with no model call: its agent, phase and objective
    and the titles of its inputs, e.g. "engineer, implementation, Build the books
    API, Books API"."""
    parts = [task.agent.value, task.phase.value, task.objective]
    parts += [title_of(a) for a in inputs]
    return ", ".join(dict.fromkeys(p.strip() for p in parts if p.strip()))


def keyword_score(match: MemoryMatch) -> float:
    """The keyword match, 0 to 1: BM25 squashed, or the words in common if higher."""
    s = max(match.bm25, 0.0)
    return max(s / (1.0 + s), match.words)


def relevance(match: MemoryMatch, config: MemoryConfig) -> float:
    """How well a memory's text matches the query, 0 to 1."""
    w = config.weights
    keyword = keyword_score(match)
    if match.cosine is None:
        return keyword
    cosine = min(max(match.cosine, 0.0), 1.0)
    return (w.cosine * cosine + w.bm25 * keyword) / (w.cosine + w.bm25)


def strong_relevance(match: MemoryMatch, config: MemoryConfig) -> float:
    """The text match a memory from another run is held to, 0 to 1: like
    ``relevance``, but the keyword part is the share of words in common, since BM25
    over a few memories rates one shared word as a strong match."""
    w = config.weights
    if match.cosine is None:
        return match.words
    cosine = min(max(match.cosine, 0.0), 1.0)
    return (w.cosine * cosine + w.bm25 * match.words) / (w.cosine + w.bm25)


def recency(created_at: datetime, now: datetime, half_life_days: float) -> float:
    """1 for a memory saved now, halving every ``half_life_days``."""
    age_days = max((now - created_at).total_seconds(), 0.0) / 86_400
    return float(0.5 ** (age_days / half_life_days))


def score(match: MemoryMatch, rel: float, config: MemoryConfig, now: datetime) -> float:
    """Text match, recency and usefulness, weighted by the config."""
    w = config.weights
    used = match.memory.usefulness
    return (
        (w.cosine + w.bm25) * rel
        + w.recency * recency(match.memory.created_at, now, config.recency_half_life_days)
        + w.usefulness * (used / (used + 2))
    )


def lesson_agents(memory: Memory) -> set[AgentName]:
    """The agents a lesson came from: every source's agent (a merged lesson can have
    several), or for a lesson saved before sources named one, the agent its text
    starts with ("Engineer, implementation: ..."). Empty when nothing says."""
    named = {s.agent for s in memory.sources if s.agent is not None}
    if named:
        return named
    head = memory.text.split(",", 1)[0].strip().lower()
    return {AgentName(head)} if head in {a.value for a in AgentName} else set()


def rank(
    matches: list[MemoryMatch],
    config: MemoryConfig,
    *,
    now: datetime,
    run_id: str | None,
    agent: AgentName | None = None,
) -> list[Ranked]:
    """Matches that are strong enough, best first. A weak text match is dropped, and a
    decision or research fact from another run than ``run_id`` needs a strong one
    (None keeps every run's alike). A lesson goes only to the ``agent`` whose task
    taught it (None gives every agent's). Ties go to the newer memory, then the lower
    id."""
    out: list[Ranked] = []
    for match in matches:
        if agent is not None and match.memory.kind is MemoryKind.LESSON:
            taught = lesson_agents(match.memory)
            if taught and agent not in taught:
                continue
        rel = relevance(match, config)
        if rel < config.min_relevance:
            continue
        other_run = run_id is not None and match.memory.origin.run_id != run_id
        if (
            other_run
            and match.memory.kind in RUN_BOUND
            and strong_relevance(match, config) < config.strong_match
        ):
            continue
        out.append(Ranked(match.memory, rel, score(match, rel, config, now)))
    out.sort(key=lambda r: (-r.score, -r.memory.created_at.timestamp(), r.memory.id))
    return out


def line(memory: Memory, run_id: str | None) -> str:
    """One memory as a prompt line: its kind, its text quoted as data, and where it
    came from."""
    origin = memory.origin
    if origin.run_id == run_id:
        where = "this run"
    elif origin.run_id == SESSION:
        where = "your messages"
    else:
        where = f"run {origin.run_id}"
    refs = list(dict.fromkeys(s.ref for s in memory.sources if s.ref))
    if refs:
        more = f" and {len(refs) - MAX_REFS} more" if len(refs) > MAX_REFS else ""
        where += ", source " + ", ".join(refs[:MAX_REFS]) + more
    text = json.dumps(" ".join(memory.text.split()), ensure_ascii=False)
    return f"{LABELS[memory.kind]}: {text} ({where})"


def fit(
    ranked: list[Ranked], *, run_id: str | None, max_items: int, max_tokens: int
) -> list[tuple[Ranked, str]]:
    """The best memories and their lines, in rank order, up to ``max_items`` and about
    ``max_tokens``; a line too long for the room left is skipped for the next one."""
    out: list[tuple[Ranked, str]] = []
    used = 0
    for r in ranked:
        if len(out) >= max_items:
            break
        text = line(r.memory, run_id)
        cost = estimate_tokens(text)
        if used + cost <= max_tokens:
            out.append((r, text))
            used += cost
    return out
