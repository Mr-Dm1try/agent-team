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
    is_requested,
    question_words,
    research_depth,
    same_question,
    waits_on,
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
    # Questions that differ in their key word are different questions.
    github = "What is the rate limit of the GitHub REST API for authenticated users?"
    assert not same_question(
        question_words(github), question_words(github.replace("GitHub", "GitLab"))
    )
    # Nothing left to compare (all filler, or a script the words pattern can't read)
    # matches nothing, so the question is researched.
    assert not same_question(question_words("Как работает кэш?"), question_words("Какая лицензия?"))
    assert not same_question(
        question_words("What should we use?"), question_words("Which is latest?")
    )


def test_requested_research_and_its_revisions_can_be_waited_on_but_not_in_a_cycle() -> None:
    planned = researcher()
    asked = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    revised = researcher(TaskOrigin.REVISION, asked)
    by_id = {t.id: t for t in (planned, asked, revised)}
    assert not is_requested(planned, by_id)
    assert is_requested(asked, by_id) and is_requested(revised, by_id)
    other = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    by_id[other.id] = other
    assert not waits_on(other, revised, by_id)
    revised.depends_on.append(other.id)  # the revision now waits on ``other``
    assert waits_on(other, revised, by_id)


def test_depth_counts_research_requests_along_the_parents() -> None:
    planned = researcher()
    child = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    retry = researcher(TaskOrigin.RETRY, child)
    grandchild = researcher(TaskOrigin.RESEARCH_REQUEST, retry)
    by_id = {t.id: t for t in (planned, child, retry, grandchild)}
    assert [research_depth(t, by_id) for t in by_id.values()] == [0, 1, 1, 2]


def test_limits_refuse_past_the_depth_the_cap_an_earlier_round_or_the_last_attempt() -> None:
    limits = ResearchLimits(max_tasks=2, max_depth=1)
    planned = researcher()
    child = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    assert limits.refusal(planned, [planned, child]) is None
    assert "1 deep" in (limits.refusal(child, [planned, child]) or "")
    other = researcher(TaskOrigin.RESEARCH_REQUEST, planned)
    assert "its 2 requested" in (limits.refusal(planned, [planned, child, other]) or "")
    asked = planned.model_copy(update={"may_request_research": False})
    assert "already asked" in (limits.refusal(asked, [asked]) or "")
    last = planned.model_copy(update={"attempt": 3, "max_attempts": 3})
    assert "last attempt" in (limits.refusal(last, [last]) or "")
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


def test_a_new_plan_does_not_take_requested_research_as_its_own_line() -> None:
    """A re-planned question that an agent once asked as a request starts its own
    planned line (depth 0, free to ask), while a planned question is still revised."""
    from agent_team.core.artifacts import Plan
    from agent_team.core.models import Run
    from agent_team.orchestrator.planning import expand_plan

    run = Run(title="t", request="r", workspace="/ws")
    planner = Task(
        run_id=run.id,
        phase=Phase.PLANNING,
        agent=AgentName.ORCHESTRATOR,
        objective="plan",
        output_kind="plan",
    )
    old_planned = researcher(objective="Which DB?", status=TaskStatus.SUPERSEDED)
    asked = researcher(
        TaskOrigin.RESEARCH_REQUEST, old_planned, objective="mcp SDK?", status=TaskStatus.SUPERSEDED
    )
    plan = Plan(
        summary="s",
        request_kind="build",
        asks=[{"id": "ASK-1", "text": "x"}],  # type: ignore[list-item]
        research_questions=[
            {"question": "Which DB?", "ask": "ASK-1"},  # type: ignore[list-item]
            {"question": "mcp SDK?", "ask": "ASK-1"},  # type: ignore[list-item]
        ],
    )
    ref = ArtifactRef(kind="plan", version=1)
    tasks = expand_plan(plan, ref, run, planner, previous=[old_planned, asked])
    by_question = {t.objective: t for t in tasks if t.agent is AgentName.RESEARCHER}
    assert by_question["Which DB?"].origin is TaskOrigin.REVISION
    assert by_question["Which DB?"].parent_task_id == old_planned.id
    assert by_question["mcp SDK?"].origin is TaskOrigin.PLAN
