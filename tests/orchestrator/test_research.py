"""Research request limits: matching repeated questions, depth and the run cap, and
how a task that can't ask turns its requests into open gaps."""

from __future__ import annotations

from agent_team.agents.base import NOT_RESEARCHED, close_research
from agent_team.core.artifacts import (
    AnsweredQuestion,
    ArtifactRef,
    Question,
    ResearchBrief,
    ResearchRequest,
    Spec,
)
from agent_team.core.models import AgentName, Phase, Task, TaskOrigin, TaskStatus
from agent_team.orchestrator.research import (
    ResearchLimits,
    find_same,
    question_words,
    research_depth,
    same_question,
)


def researcher(
    origin: TaskOrigin = TaskOrigin.PLAN, parent: Task | None = None, **kw: object
) -> Task:
    """A research task in run ``r``."""
    return Task(
        run_id="r",
        phase=Phase.PLANNING,
        agent=AgentName.RESEARCHER,
        output_kind="research_brief",
        origin=origin,
        parent_task_id=parent.id if parent else None,
        **{"objective": "q", **kw},  # type: ignore[arg-type]
    )


def test_rephrasings_of_one_question_match_and_different_ones_do_not() -> None:
    a = question_words("Where are the PyYAML docs?")
    assert same_question(a, question_words("where are the official PyYAML docs"))
    assert same_question(
        question_words("What is the mcp SDK import path?"),
        question_words("mcp SDK import paths"),
    )
    assert not same_question(a, question_words("Where are the python-frontmatter docs?"))
    assert not same_question(a, frozenset())


def test_depth_counts_research_requests_along_the_parents() -> None:
    planned = researcher()
    child = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    retry = researcher(TaskOrigin.RETRY, child)
    grandchild = researcher(TaskOrigin.RESEARCH_REQUEST, retry)
    by_id = {t.id: t for t in (planned, child, retry, grandchild)}
    assert [research_depth(t, by_id) for t in by_id.values()] == [0, 1, 1, 2]


def test_limits_refuse_past_the_depth_the_cap_or_an_earlier_refusal() -> None:
    limits = ResearchLimits(max_tasks=2, max_depth=1)
    planned = researcher()
    child = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    assert limits.refusal(planned, [planned, child]) is None
    assert "1 deep" in (limits.refusal(child, [planned, child]) or "")
    other = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    assert "its 2 requested" in (limits.refusal(planned, [planned, child, other]) or "")
    refused = planned.model_copy(update={"may_request_research": False})
    assert "already refused" in (limits.refusal(refused, [refused]) or "")
    assert ResearchLimits(max_tasks=0).refusal(planned, [planned]) is not None


def test_find_same_prefers_a_finished_answer_and_skips_replaced_work() -> None:
    ref = ArtifactRef(kind="research_brief", version=1)
    open_one = researcher(objective="mcp SDK import path?", status=TaskStatus.RUNNING)
    done = researcher(
        objective="What is the mcp SDK import path", status=TaskStatus.DONE, output=ref
    )
    old = researcher(objective="mcp SDK import path", status=TaskStatus.SUPERSEDED, output=ref)
    match = find_same("The mcp SDK import path?", [open_one, done, old])
    assert match is not None and match.task.id == done.id and match.answered
    match = find_same("The mcp SDK import path?", [open_one, old])
    assert match is not None and match.task.id == open_one.id and not match.answered
    assert find_same("Obsidian wikilink syntax", [open_one, done]) is None


def test_a_brief_that_may_not_ask_lists_its_requests_as_gaps() -> None:
    brief = ResearchBrief(
        summary="s",
        answers=[AnsweredQuestion(question="a?", answer="yes")],
        research_requests=[ResearchRequest(question="b?", why="needed for c")],
    )
    closed = close_research(brief)
    assert isinstance(closed, ResearchBrief) and closed.research_requests == []
    assert closed.answers[-1] == AnsweredQuestion(
        question="b?", answer=None, gap="not researched (needed for c)"
    )


def test_other_work_that_may_not_ask_records_an_assumption() -> None:
    spec = Spec.model_validate(
        {
            "summary": "s",
            "components": [],
            "checks": [],
            "questions": [Question(id="Q-2", text="x", why="y").model_dump()],
            "research_requests": [{"question": "Which ORM?", "why": "deps"}],
        }
    )
    closed = close_research(spec)
    assert closed.research_requests == []
    assert closed.questions[-1] == Question(
        id="Q-3", text="Which ORM?", why="deps", assumption=NOT_RESEARCHED
    )
