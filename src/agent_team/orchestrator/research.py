"""Limits on research that agents ask for mid-run: a request that repeats one already
asked in the run is answered by that task instead of a new one, a task created from a
request can't ask for more past ``max_research_depth``, and a run makes at most
``max_research_tasks`` of them (docs/DESIGN.md, "Research requests")."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from agent_team.core.models import AgentName, Task, TaskOrigin, TaskStatus

_WORD = re.compile(r"[a-z0-9]+")
#: Words that don't tell two research questions apart.
_FILLER = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "the",
        "this",
        "to",
        "use",
        "used",
        "using",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "docs",
        "documentation",
        "official",
        "latest",
        "current",
    }
)
#: Share of content words two questions must have in common to count as the same.
SAME_QUESTION = 0.7


def question_words(text: str) -> frozenset[str]:
    """The content words of a research question, lower-cased, without filler and with
    a plural ``s`` dropped, so rephrasings of one question compare equal."""
    words = set()
    for word in _WORD.findall(text.casefold()):
        if word in _FILLER:
            continue
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return frozenset(words)


def same_question(a: frozenset[str], b: frozenset[str]) -> bool:
    """Whether two questions' content words overlap enough (Jaccard) to be one."""
    if not a or not b:
        return a == b
    return len(a & b) / len(a | b) >= SAME_QUESTION


def research_depth(task: Task, by_id: dict[str, Task]) -> int:
    """How many research requests led to ``task``: 0 for planned work, 1 for a task
    created from a request, and so on along its parents."""
    depth, seen = 0, set()
    current: Task | None = task
    while current is not None and current.id not in seen:
        seen.add(current.id)
        if current.origin is TaskOrigin.RESEARCH_REQUEST:
            depth += 1
        current = by_id.get(current.parent_task_id) if current.parent_task_id else None
    return depth


def requested_count(tasks: Iterable[Task]) -> int:
    """Research tasks the run has made from agents' requests."""
    return sum(1 for t in tasks if t.origin is TaskOrigin.RESEARCH_REQUEST)


@dataclass(frozen=True)
class ResearchLimits:
    """How much research agents may ask for in one run."""

    max_tasks: int = 6
    max_depth: int = 1

    def refusal(self, task: Task, tasks: list[Task]) -> str | None:
        """Why ``task`` may not ask for (more) research now, or None if it may."""
        if not task.may_request_research:
            return "a research request of this task was already refused"
        if research_depth(task, {t.id: t for t in tasks}) >= self.max_depth:
            return f"research requests may chain only {self.max_depth} deep"
        if requested_count(tasks) >= self.max_tasks:
            return f"the run has made its {self.max_tasks} requested research tasks"
        return None


@dataclass(frozen=True)
class Match:
    """An earlier research task that already covers a request."""

    task: Task

    @property
    def answered(self) -> bool:
        """It has finished with a brief that can be used now."""
        return self.task.status is TaskStatus.DONE and self.task.output is not None


def find_same(question: str, tasks: Iterable[Task]) -> Match | None:
    """The research task in ``tasks`` that already asks ``question``, if any; finished
    ones first, since their answer is ready. Replaced (superseded) ones don't count."""
    words = question_words(question)
    found: list[Task] = []
    for t in tasks:
        if t.agent is not AgentName.RESEARCHER or t.status is TaskStatus.SUPERSEDED:
            continue
        if same_question(words, question_words(t.objective)):
            found.append(t)
    if not found:
        return None
    found.sort(key=lambda t: (t.status is not TaskStatus.DONE, t.created_at))
    return Match(found[0])
