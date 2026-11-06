"""The orchestrator's control tools. They are the only way its model touches the run:
read run state (tasks, artifacts, verdicts, decisions, the trace), create tasks and ask
the user. Code validates every call; a bad call is an error result, never an action.
"""

from __future__ import annotations

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
    VerdictStore,
)
from agent_team.core.models import (
    AgentName,
    Event,
    EventName,
    Phase,
    Run,
    Task,
    TaskOrigin,
    TaskStatus,
    ToolResult,
)
from agent_team.core.tracing import cut_strings, format_event
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


def str_arg(args: dict[str, Any], key: str, tool: str, *, required: bool = True) -> str | None:
    """A required (or optional) non-empty string argument, stripped."""
    value = args.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"argument {key!r} must be a non-empty string", tool=tool)
    return value.strip()


def required_str(args: dict[str, Any], key: str, tool: str) -> str:
    """A required non-empty string argument, stripped."""
    value = str_arg(args, key, tool)
    assert value is not None  # str_arg raises for a missing required argument
    return value


def _task_line(t: Task) -> str:
    """One task as a single line for list_tasks."""
    out = f" -> {t.output}" if t.output else ""
    err = f" error={t.last_error}" if t.last_error else ""
    return (
        f"{t.id} {t.phase.value} {t.agent.value} {t.status.value} "
        f"attempt {t.attempt}/{t.max_attempts} origin={t.origin.value}{out}{err}: {t.objective}"
    )


class ListTasks:
    """Tool: the run's tasks, one line each."""

    def __init__(self, tasks: TaskStore) -> None:
        """Read-only, so it is safe to repeat on resume."""
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
        """The run's tasks, filtered by phase and status."""
        try:
            phase = Phase(args["phase"]) if args.get("phase") else None
            status = TaskStatus(args["status"]) if args.get("status") else None
        except ValueError as exc:
            raise ToolError(str(exc), tool=self.spec.name) from exc
        found = await self.tasks.list_tasks(ctx.run.id, status=status, phase=phase)
        body = "\n".join(_task_line(t) for t in found) or "no tasks"
        return ToolResult(content=body, data={"count": len(found)})


class GetTask:
    """Tool: one task in full."""

    def __init__(self, tasks: TaskStore) -> None:
        """Read-only, so it is safe to repeat on resume."""
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
        """The full task record, refusing tasks of other runs."""
        task_id = required_str(args, "task_id", self.spec.name)
        try:
            task = await self.tasks.get_task(task_id)
        except Exception as exc:
            raise ToolError(f"no task {task_id!r}", tool=self.spec.name) from exc
        if task.run_id != ctx.run.id:
            raise ToolError(f"task {task_id!r} belongs to another run", tool=self.spec.name)
        return ToolResult(content=task.model_dump_json(indent=1))


class GetArtifact:
    """Tool: one artifact version as JSON."""

    def __init__(self, artifacts: ArtifactStore) -> None:
        """Read-only, so it is safe to repeat on resume."""
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
        """Load ``kind@vN`` exactly, or the latest version of a bare kind."""
        raw = required_str(args, "ref", self.spec.name)
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
    """Tool: the audit verdicts on a task."""

    def __init__(self, verdicts: VerdictStore) -> None:
        """Read-only, so it is safe to repeat on resume."""
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
        """Every verdict on the task as JSON lines."""
        task_id = required_str(args, "task_id", self.spec.name)
        found = await self.verdicts.for_task(task_id)
        if not found:
            return ToolResult(content=f"no verdicts for {task_id}")
        return ToolResult(content="\n".join(v.model_dump_json() for v in found))


class ListDecisions:
    """Tool: the run's decision log."""

    def __init__(self, decisions: DecisionLog) -> None:
        """Read-only, so it is safe to repeat on resume."""
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
        """The decision log, optionally for one task, one line each."""
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


#: Events read_trace leaves out unless asked for by name: bookkeeping, not what happened.
TRACE_NOISE = frozenset(
    {EventName.CHECKPOINT_SAVED, EventName.SPAN_STARTED, EventName.SPAN_FINISHED}
)
#: The most characters read_trace returns; older lines go first.
TRACE_CHARS = 6_000
#: Longest string kept from an event's data in a summary line, and in a full line.
TRACE_SUMMARY_CHARS = 120
TRACE_FULL_CHARS = 1_000


class ReadTrace:
    """Reads runs/<run_id>/logs/events.jsonl, written by the tracer hook. By default
    each event is one short line and bookkeeping events are left out, so a few calls
    can't fill the context; ``full`` gives the events' JSON instead."""

    def __init__(self) -> None:
        """Read-only, so it is safe to repeat on resume."""
        self.safe_to_repeat = True
        self.spec = ToolSpec(
            name="read_trace",
            description=(
                "Recent events from the run's trace, one short line each, newest last. "
                "Narrow it with task_id, agent, event names or errors_only; pass full "
                "for an event's JSON. Prefer list_tasks, get_task and get_verdicts, "
                "which already say how each task went."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "agent": {"type": "string", "enum": [a.value for a in AgentName]},
                    "events": {
                        "type": "array",
                        "items": {"type": "string", "enum": [e.value for e in EventName]},
                    },
                    "errors_only": {"type": "boolean", "description": "Warnings and errors"},
                    "full": {"type": "boolean", "description": "Each event's JSON"},
                    "limit": {"type": "integer", "description": "Default 30"},
                },
            },
        )

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """The run's last matching events, as short lines (or JSON), within
        ``TRACE_CHARS``."""
        name = self.spec.name
        path = Path(ctx.run.workspace).parent / "logs" / "events.jsonl"
        if not path.is_file():
            return ToolResult(content="no trace file yet", is_error=True)
        limit = args.get("limit", 30)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            limit = 30
        task_id = str_arg(args, "task_id", name, required=False)
        agent = str_arg(args, "agent", name, required=False)
        wanted = args.get("events") or []
        if not isinstance(wanted, list):
            raise ToolError("events must be a list of event names", tool=name)
        try:
            names = {EventName(str(w)) for w in wanted}
        except ValueError as exc:
            raise ToolError(str(exc), tool=name) from exc
        errors_only, full = bool(args.get("errors_only")), bool(args.get("full"))
        found: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                event = Event.model_validate_json(raw)
            except ValueError:
                continue
            if task_id and event.task_id != task_id:
                continue
            if agent and (event.agent is None or event.agent.value != agent):
                continue
            if names and event.name not in names:
                continue
            if not names and event.name in TRACE_NOISE:
                continue
            if errors_only and event.level not in ("warning", "error"):
                continue
            found.append(_trace_line(event, full=full))
        return ToolResult(content=_fit(found[-limit:]), data={"matched": len(found)})


def _trace_line(event: Event, *, full: bool) -> str:
    """One event for read_trace: a short summary line, or its JSON with long strings
    cut."""
    if full:
        data = cut_strings(event.data, TRACE_FULL_CHARS, one_line=False)
        cut = event.model_copy(update={"data": data})
        return cut.model_dump_json(exclude_none=True)
    return format_event(event, TRACE_SUMMARY_CHARS)


def _fit(lines: list[str]) -> str:
    """The newest lines that fit in ``TRACE_CHARS``, saying how many older ones were
    left out."""
    kept: list[str] = []
    size = 0
    for line in reversed(lines):
        if kept and size + len(line) + 1 > TRACE_CHARS:
            break
        kept.append(line[:TRACE_CHARS])
        size += len(line) + 1
    kept.reverse()
    if not kept:
        return "no matching events"
    dropped = len(lines) - len(kept)
    head = [f"[{dropped} older events left out; narrow the filters to see them]"]
    return "\n".join((head if dropped else []) + kept)


class CreateTask:
    """Tool: add a task, checked against the agent's phase and output kinds."""

    def __init__(self, tasks: TaskStore, max_attempts: int = 3) -> None:
        """Creates rows, so it is not repeated on resume."""
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
        """Validate the agent, phase and output kind, then create the task."""
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
        output_kind = required_str(args, "output_kind", name)
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
                objective=required_str(args, "objective", name),
                output_kind=output_kind,
                depends_on=deps,
                origin=origin,
                parent_task_id=ctx.task.id,
                reason=required_str(args, "reason", name),
                max_attempts=self.max_attempts,
            )
        )
        return ToolResult(content=f"created {task.id}", data={"task_id": task.id})


class AskUser:
    """Tool: put a question to the user as a waiting user task."""

    def __init__(self, tasks: TaskStore) -> None:
        """Creates a user task, so it is not repeated on resume."""
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
        """Create a waiting user task with the question, in the given or current phase."""
        question = required_str(args, "question", self.spec.name)
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
    """The read-only tools for answering questions about a run."""
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
    """The read tools plus the tools that create tasks and ask the user."""
    tools = read_tools(tasks, artifacts, decisions, verdicts)
    tools["create_task"] = CreateTask(tasks, max_attempts)
    tools["ask_user"] = AskUser(tasks)
    return tools
