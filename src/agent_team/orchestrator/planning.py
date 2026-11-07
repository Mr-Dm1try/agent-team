"""The fixed pipeline: a passed Plan becomes task rows by code, not by the model.

- Build: one Researcher task (research_brief) per research question in Planning, then
  Product (PRD) and Architect (Spec). The Engineer and QA tasks are created when the
  spec task is done, from the spec's work packages: one Engineer task per package (or
  one for the whole spec), run one after another, and QA after all of them.
- Research question: one Researcher task per research question, then one Researcher
  task that writes the research_answer.
"""

from __future__ import annotations

from agent_team.core.artifacts import ArtifactRef, Plan, Spec
from agent_team.core.models import (
    AgentName,
    Phase,
    RequestKind,
    Run,
    Task,
    TaskOrigin,
    new_id,
)
from agent_team.orchestrator.lineage import ancestry
from agent_team.orchestrator.research import is_requested

RULE = "pipeline"


def _asks(plan: Plan) -> str:
    """The plan's asks as one line, for task objectives."""
    return "; ".join(f"{a.id} {a.text}" for a in plan.asks)


def expand_plan(
    plan: Plan,
    plan_ref: ArtifactRef,
    run: Run,
    planning_task: Task,
    *,
    previous: list[Task] | None = None,
    max_attempts: int = 3,
) -> list[Task]:
    """Tasks from the plan up to the Spec (build) or the answer (research question).

    ``previous`` are superseded tasks from before the plan was reopened: a new task
    for the same work (the same research question, or the PRD, spec or answer) points
    back at the old one, so it runs as a revision with that work's output and
    feedback instead of from scratch."""
    earlier = {t.id: t for t in previous or []}
    old_work = {
        _work_key(t): t
        for t in sorted(earlier.values(), key=lambda t: t.created_at)
        # Research an agent asked for is not the plan's work, even with the same question.
        if not is_requested(t, earlier)
    }

    def task(
        phase: Phase,
        agent: AgentName,
        kind: str,
        objective: str,
        deps: list[str],
        reason: str,
    ) -> Task:
        """A new ready-to-schedule task that depends on the planning task and ``deps``,
        revising the superseded task that did the same work, if any."""
        old = old_work.get((agent, kind, objective if kind == "research_brief" else None))
        return Task(
            id=new_id("task"),
            run_id=run.id,
            phase=phase,
            agent=agent,
            objective=objective,
            inputs=[plan_ref],
            output_kind=kind,
            # Everything depends on the planner, so re-opening the plan supersedes it.
            depends_on=[planning_task.id, *deps],
            max_attempts=max_attempts,
            origin=TaskOrigin.REVISION if old else TaskOrigin.PLAN,
            parent_task_id=old.id if old else planning_task.id,
            reason=reason,
        )

    research = [
        task(
            Phase.PLANNING,
            AgentName.RESEARCHER,
            "research_brief",
            rq.question,
            [],
            f"research question for {rq.ask}",
        )
        for rq in plan.research_questions
    ]
    research_ids = [t.id for t in research]
    if RequestKind(plan.request_kind) is RequestKind.RESEARCH_QUESTION:
        answer = task(
            Phase.PLANNING,
            AgentName.RESEARCHER,
            "research_answer",
            f"Answer the user's question with sources, written for the user: {run.request}",
            research_ids,
            f"final answer covering {', '.join(a.id for a in plan.asks)}",
        )
        return [*research, answer]

    prd = task(
        Phase.PRD,
        AgentName.PRODUCT,
        "prd",
        f"Write the PRD for: {_asks(plan)}",
        research_ids,
        "pipeline: PRD after research",
    )
    spec = task(
        Phase.SPEC,
        AgentName.ARCHITECT,
        "spec",
        "Write the spec for the PRD",
        [prd.id],
        "pipeline: spec after the PRD",
    )
    return [*research, prd, spec]


def _work_key(task: Task) -> tuple[AgentName, str, str | None]:
    """What a pipeline task is for: its agent and output, and for research its
    question, since a plan has one research task per question."""
    question = task.objective if task.output_kind == "research_brief" else None
    return (task.agent, task.output_kind, question)


def package_of(task: Task) -> str | None:
    """The work package an Engineer task was created for (from its reason)."""
    prefix = "work package "
    if task.reason and task.reason.startswith(prefix):
        return task.reason[len(prefix) :]
    return None


def implementation_tasks(
    spec: Spec,
    spec_task: Task,
    run: Run,
    *,
    previous: list[Task] | None = None,
    max_attempts: int = 3,
    change_request: str | None = None,
) -> list[Task]:
    """Engineer tasks per work package, chained one after another, then QA.

    ``previous`` are superseded Implementation tasks from before a spec revision: a
    new task for the same package points back at the newest task in that package's
    line (a fix included), so it runs as a revision of the latest code instead of a
    rebuild. ``change_request`` is the user's change still being built, added to
    every objective so the Engineer and QA know what is new."""
    packages = [(p.name, p.components) for p in spec.work_packages] or [
        ("all", [c.name for c in spec.components])
    ]
    earlier = sorted(previous or [], key=lambda t: t.created_at)
    old_by_package: dict[str | None, Task] = {}
    for t in earlier:
        if t.agent is AgentName.ENGINEER:
            old_by_package[package_of(ancestry(t, earlier)[0])] = t
    old_qa = next((t for t in reversed(earlier) if t.agent is AgentName.QA), None)
    change = f". Change request: {change_request}" if change_request else ""

    tasks: list[Task] = []
    prev_id = spec_task.id
    for name, components in packages:
        old = old_by_package.get(name)
        what = "the whole spec" if name == "all" else f"work package {name!r}"
        tasks.append(
            Task(
                id=new_id("task"),
                run_id=run.id,
                phase=Phase.IMPLEMENTATION,
                agent=AgentName.ENGINEER,
                objective=f"Implement {what}: components {', '.join(components)}{change}",
                output_kind="code_change",
                depends_on=[prev_id],
                max_attempts=max_attempts,
                origin=TaskOrigin.REVISION if old else TaskOrigin.PLAN,
                parent_task_id=old.id if old else spec_task.id,
                reason=f"work package {name}",
            )
        )
        prev_id = tasks[-1].id
    tasks.append(
        Task(
            id=new_id("task"),
            run_id=run.id,
            phase=Phase.IMPLEMENTATION,
            agent=AgentName.QA,
            objective="Review the whole solution against the PRD and run it end to end" + change,
            output_kind="qa_report",
            depends_on=[t.id for t in tasks],
            max_attempts=max_attempts,
            origin=TaskOrigin.REVISION if old_qa else TaskOrigin.PLAN,
            parent_task_id=old_qa.id if old_qa else spec_task.id,
            reason="pipeline: QA after every Engineer task",
        )
    )
    return tasks
