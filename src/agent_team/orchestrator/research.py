"""Limits on research that agents ask for mid-run: a request that repeats one already
asked in the run is answered by that task instead of a new one, a task created from a
request can't ask for more past ``max_research_depth``, and a run makes at most
``max_research_tasks`` of them (docs/DESIGN.md, "Research requests")."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agent_team.core.models import AgentName, Task, TaskOrigin, TaskStatus
from agent_team.core.words import terms

#: Words beyond the usual stopwords that don't tell two research questions apart.
_FILLER = frozenset(
    {
        "can",
        "do",
        "does",
        "how",
        "should",
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
        "you",
        "your",
        "doc",
        "docs",
        "documentation",
        "official",
        "latest",
        "current",
    }
)
#: Share of content words (Jaccard) two questions must have in common to count as the
#: same. High, since a false match hands an agent the wrong facts while a missed one
#: only costs a task: "GitHub's rate limit" and "GitLab's rate limit" share 75%.
SAME_QUESTION = 0.85


def question_words(text: str) -> frozenset[str]:
    """The content words of a research question, as memory search reads them, without
    filler, so rephrasings of one question compare equal."""
    return frozenset(terms(text) - _FILLER)


def same_question(a: frozenset[str], b: frozenset[str]) -> bool:
    """Whether two questions' content words overlap enough (Jaccard) to be one. A
    question with no content words left (all filler, or a script the word pattern
    doesn't read) matches nothing, so it is researched rather than wrongly reused."""
    if not a or not b:
        return False
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


def is_requested(task: Task, by_id: dict[str, Task]) -> bool:
    """Whether ``task`` is research an agent asked for, or a retry or revision of it."""
    return task.agent is AgentName.RESEARCHER and research_depth(task, by_id) > 0


def waits_on(task: Task, target: Task, by_id: dict[str, Task]) -> bool:
    """Whether ``target`` already waits, directly or not, on ``task``; then ``task``
    waiting on ``target`` would never end."""
    seen: set[str] = set()
    frontier = [target]
    while frontier:
        current = frontier.pop()
        if current.id == task.id:
            return True
        if current.id in seen:
            continue
        seen.add(current.id)
        frontier += [by_id[d] for d in current.depends_on if d in by_id]
    return False


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
