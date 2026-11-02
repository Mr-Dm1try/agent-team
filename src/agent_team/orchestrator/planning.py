"""The fixed pipeline: a passed Plan becomes task rows by code, not by the model.

- Build: one Researcher task (research_brief) per research question in Planning, then
  Product (PRD) and Architect (Spec). The Engineer and QA tasks are created when the
  Spec gate passes, from the spec's work packages: one Engineer task per package (or
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

RULE = "pipeline"


def _asks(plan: Plan) -> str:
    return "; ".join(f"{a.id} {a.text}" for a in plan.asks)


def expand_plan(
    plan: Plan,
    plan_ref: ArtifactRef,
    run: Run,
    planning_task: Task,
    *,
    max_attempts: int = 3,
) -> list[Task]:
    """Tasks from the plan up to the Spec (build) or the answer (research question)."""

    def task(
        phase: Phase,
        agent: AgentName,
        kind: str,
        objective: str,
        deps: list[str],
        reason: str,
    ) -> Task:
        return Task(
            id=new_id("task"),
            run_id=run.id,
            phase=phase,
            agent=agent,
            objective=objective,
            inputs=[plan_ref],
            output_kind=kind,
            depends_on=deps,
            max_attempts=max_attempts,
            origin=TaskOrigin.PLAN,
            parent_task_id=planning_task.id,
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
            f"Answer the user's question with sources: {run.request}",
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
) -> list[Task]:
    """Engineer tasks per work package, chained one after another, then QA.

    ``previous`` are superseded Implementation tasks from before a spec revision: a
    new task for the same package points back at the old one, so it runs as a
    revision of that code instead of a rebuild."""
    packages = [(p.name, p.components) for p in spec.work_packages] or [
        ("all", [c.name for c in spec.components])
    ]
    old_by_package = {package_of(t): t for t in previous or [] if t.agent is AgentName.ENGINEER}
    old_qa = next((t for t in reversed(previous or []) if t.agent is AgentName.QA), None)

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
                objective=f"Implement {what}: components {', '.join(components)}",
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
            objective="Review the whole solution against the PRD and run it end to end",
            output_kind="qa_report",
            depends_on=[t.id for t in tasks],
            max_attempts=max_attempts,
            origin=TaskOrigin.REVISION if old_qa else TaskOrigin.PLAN,
            parent_task_id=old_qa.id if old_qa else spec_task.id,
            reason="pipeline: QA after every Engineer task",
        )
    )
    return tasks
