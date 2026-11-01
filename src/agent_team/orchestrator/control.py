"""The orchestrator's control tools. They are the only way its model touches the run:
read run state (tasks, artifacts, verdicts, decisions, the trace), create tasks and ask
the user. Code validates every call; a bad call is an error result, never an action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_team.agents.planner import AGENT_PLAN
from agent_team.core.artifacts import ArtifactRef
from agent_team.core.errors import ToolError
from agent_team.core.interfaces import (
    ArtifactStore,
    DecisionLog,
    TaskStore,
    Tool,
    ToolContext,
    ToolResult,
    VerdictStore,
)
from agent_team.core.models import (
    AgentName,
    Phase,
    Run,
    Task,
    TaskOrigin,
    TaskStatus,
)
from agent_team.llm.types import ToolSpec

USER_OUTPUT = "user_answer"


async def create_user_task(
    tasks: TaskStore,
    run: Run,
    *,
    phase: Phase,
    question: str,
    origin: TaskOrigin,
    parent: Task | None = None,
    reason: str,
) -> Task:
    """A question or review assigned to the user. It waits until they answer."""
    task = await tasks.create_task(
        Task(
            run_id=run.id,
            phase=phase,
            agent=AgentName.USER,
            objective=question,
            output_kind=USER_OUTPUT,
            origin=origin,
            parent_task_id=parent.id if parent else None,
            reason=reason,
            max_attempts=1,
        )
    )
    for status, why in (
        (TaskStatus.READY, "assigned to the user"),
        (TaskStatus.RUNNING, "shown to the user"),
        (TaskStatus.WAITING, "waiting on the user's answer"),
    ):
        task = await tasks.transition(task.id, status, reason=why)
    return task


def _str(args: dict[str, Any], key: str, tool: str, *, required: bool = True) -> str | None:
    value = args.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"argument {key!r} must be a non-empty string", tool=tool)
    return value.strip()


def _task_line(t: Task) -> str:
    out = f" -> {t.output}" if t.output else ""
    err = f" error={t.last_error}" if t.last_error else ""
    return (
        f"{t.id} {t.phase.value} {t.agent.value} {t.status.value} "
        f"attempt {t.attempt}/{t.max_attempts} origin={t.origin.value}{out}{err}: {t.objective}"
    )


class ListTasks:
    def __init__(self, tasks: TaskStore) -> None:
        self.tasks = tasks
        self.safe_to_repeat = True
        self.spec = ToolSpec(
            name="list_tasks",
            description="List the run's tasks with status, attempts and outputs.",
            input_schema={
                "type": "object",
                "properties": {
                    "phase": {"type": "string", "enum": [p.value for p in Phase]},
                    "status": {"type": "string", "enum": [s.value for s in TaskStatus]},
                },
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            phase = Phase(args["phase"]) if args.get("phase") else None
            status = TaskStatus(args["status"]) if args.get("status") else None
        except ValueError as exc:
            raise ToolError(str(exc), tool=self.spec.name) from exc
        found = await self.tasks.list_tasks(ctx.run.id, status=status, phase=phase)
        body = "\n".join(_task_line(t) for t in found) or "no tasks"
        return ToolResult(content=body, data={"count": len(found)})


class GetTask:
    def __init__(self, tasks: TaskStore) -> None:
        self.tasks = tasks
        self.safe_to_repeat = True
        self.spec = ToolSpec(
            name="get_task",
            description="Full record of one task, including errors and tracebacks.",
            input_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task_id = _str(args, "task_id", self.spec.name)
        try:
            task = await self.tasks.get_task(task_id or "")
        except Exception as exc:
            raise ToolError(f"no task {task_id!r}", tool=self.spec.name) from exc
        if task.run_id != ctx.run.id:
            raise ToolError(f"task {task_id!r} belongs to another run", tool=self.spec.name)
        return ToolResult(content=task.model_dump_json(indent=1))


class GetArtifact:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts
        self.safe_to_repeat = True
        self.spec = ToolSpec(
            name="get_artifact",
            description="Load an artifact by ref such as prd@v2, or the latest by kind (prd).",
            input_schema={
                "type": "object",
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = _str(args, "ref", self.spec.name) or ""
        try:
            if "@" in raw:
                art = await self.artifacts.load(ctx.run.id, ArtifactRef.parse(raw))
            else:
                latest = await self.artifacts.latest(ctx.run.id, raw)
                if latest is None:
                    raise ToolError(f"no {raw} artifact in this run", tool=self.spec.name)
                art = latest
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"can't load {raw!r}: {exc}", tool=self.spec.name) from exc
        return ToolResult(content=art.model_dump_json(indent=1))


class GetVerdicts:
    def __init__(self, verdicts: VerdictStore) -> None:
        self.verdicts = verdicts
        self.safe_to_repeat = True
        self.spec = ToolSpec(
            name="get_verdicts",
            description="Every audit verdict for a task, oldest first, with findings.",
            input_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task_id = _str(args, "task_id", self.spec.name) or ""
        found = await self.verdicts.for_task(task_id)
        if not found:
            return ToolResult(content=f"no verdicts for {task_id}")
        return ToolResult(content="\n".join(v.model_dump_json() for v in found))


class ListDecisions:
    def __init__(self, decisions: DecisionLog) -> None:
        self.decisions = decisions
        self.safe_to_repeat = True
        self.spec = ToolSpec(
            name="list_decisions",
            description="The decision log: every choice made in the run, with its reason.",
            input_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task_id = args.get("task_id")
        found = [
            d for d in await self.decisions.list(ctx.run.id) if not task_id or d.task_id == task_id
        ]
        lines = [
            f"{d.id} {d.kind.value} task={d.task_id or '-'} by {d.made_by}: {d.choice} "
            f"because {d.reason}" + (f" [rule {d.rule}]" if d.rule else "")
            for d in found
        ]
        return ToolResult(content="\n".join(lines) or "no decisions yet")


class ReadTrace:
    """Reads runs/<run_id>/logs/events.jsonl, written by the tracer hook."""

    def __init__(self) -> None:
        self.safe_to_repeat = True
        self.spec = ToolSpec(
            name="read_trace",
            description="Recent events from the run's trace, optionally for one task.",
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "limit": {"type": "integer", "description": "Default 50"},
                },
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = Path(ctx.run.workspace).parent / "logs" / "events.jsonl"
        if not path.is_file():
            return ToolResult(content="no trace file yet", is_error=True)
        limit = args.get("limit", 50)
        if not isinstance(limit, int) or limit <= 0:
            limit = 50
        task_id = args.get("task_id")
        lines = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if task_id and event.get("task_id") != task_id:
                continue
            lines.append(raw)
        return ToolResult(content="\n".join(lines[-limit:]) or "no matching events")


class CreateTask:
    def __init__(self, tasks: TaskStore, max_attempts: int = 3) -> None:
        self.tasks = tasks
        self.max_attempts = max_attempts
        self.safe_to_repeat = False
        self.spec = ToolSpec(
            name="create_task",
            description="Create a task for an expert. Validated against the phase rules.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "enum": [a.value for a in AGENT_PLAN]},
                    "phase": {"type": "string", "enum": [p.value for p in Phase]},
                    "objective": {"type": "string"},
                    "output_kind": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                    "origin": {
                        "type": "string",
                        "enum": ["change_request", "user_rerun", "research_request"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["agent", "phase", "objective", "output_kind", "reason"],
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = self.spec.name
        try:
            agent = AgentName(str(args.get("agent")))
            phase = Phase(str(args.get("phase")))
            origin = TaskOrigin(args.get("origin", "change_request"))
        except ValueError as exc:
            raise ToolError(str(exc), tool=name) from exc
        if agent not in AGENT_PLAN:
            raise ToolError(f"{agent.value} can't be given tasks", tool=name)
        allowed_phase, kinds = AGENT_PLAN[agent]
        output_kind = _str(args, "output_kind", name) or ""
        if output_kind not in kinds:
            raise ToolError(f"{agent.value} produces one of {list(kinds)}", tool=name)
        if agent is not AgentName.RESEARCHER and phase is not allowed_phase:
            raise ToolError(f"{agent.value} works in phase {allowed_phase.value}", tool=name)
        if origin not in (
            TaskOrigin.CHANGE_REQUEST,
            TaskOrigin.USER_RERUN,
            TaskOrigin.RESEARCH_REQUEST,
        ):
            raise ToolError(
                "origin must be change_request, user_rerun or research_request", tool=name
            )
        deps = args.get("depends_on", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise ToolError("depends_on must be a list of task ids", tool=name)
        known = {t.id for t in await self.tasks.list_tasks(ctx.run.id)}
        missing = [d for d in deps if d not in known]
        if missing:
            raise ToolError(f"unknown task ids in depends_on: {missing}", tool=name)
        task = await self.tasks.create_task(
            Task(
                run_id=ctx.run.id,
                phase=phase,
                agent=agent,
                objective=_str(args, "objective", name) or "",
                output_kind=output_kind,
                depends_on=deps,
                origin=origin,
                parent_task_id=ctx.task.id,
                reason=_str(args, "reason", name),
                max_attempts=self.max_attempts,
            )
        )
        return ToolResult(content=f"created {task.id}", data={"task_id": task.id})


class AskUser:
    def __init__(self, tasks: TaskStore) -> None:
        self.tasks = tasks
        self.safe_to_repeat = False
        self.spec = ToolSpec(
            name="ask_user",
            description="Ask the user a question. The run waits for the answer.",
            input_schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "phase": {"type": "string", "enum": [p.value for p in Phase]},
                },
                "required": ["question"],
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        question = _str(args, "question", self.spec.name) or ""
        try:
            phase = Phase(args["phase"]) if args.get("phase") else ctx.task.phase
        except ValueError as exc:
            raise ToolError(str(exc), tool=self.spec.name) from exc
        task = await create_user_task(
            self.tasks,
            ctx.run,
            phase=phase,
            question=question,
            origin=TaskOrigin.QUESTION,
            reason="asked by the orchestrator",
        )
        return ToolResult(content=f"asked; waiting on {task.id}", data={"task_id": task.id})


def read_tools(
    tasks: TaskStore,
    artifacts: ArtifactStore,
    decisions: DecisionLog,
    verdicts: VerdictStore | None = None,
) -> dict[str, Tool]:
    tools: list[Tool] = [
        ListTasks(tasks),
        GetTask(tasks),
        GetArtifact(artifacts),
        ListDecisions(decisions),
        ReadTrace(),
    ]
    if verdicts is not None:
        tools.append(GetVerdicts(verdicts))
    return {t.spec.name: t for t in tools}


def control_tools(
    tasks: TaskStore,
    artifacts: ArtifactStore,
    decisions: DecisionLog,
    verdicts: VerdictStore | None = None,
    *,
    max_attempts: int = 3,
) -> dict[str, Tool]:
    tools = read_tools(tasks, artifacts, decisions, verdicts)
    tools["create_task"] = CreateTask(tasks, max_attempts)
    tools["ask_user"] = AskUser(tasks)
    return tools
