"""In-memory fakes for the stores, event bus, LLM gateway and auditor, so the agents and
orchestrator tests run without them."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from agent_team.agents.compaction import sent_json
from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.errors import InvalidTransition
from agent_team.core.interfaces import AfterHook, BeforeHook
from agent_team.core.models import (
    AgentName,
    Decision,
    Event,
    EventName,
    Feedback,
    HookOutcome,
    Phase,
    PrecheckDecision,
    PrecheckResult,
    RuleScope,
    Run,
    Task,
    TaskStatus,
    Usage,
    Verdict,
    can_transition,
    utcnow,
)
from agent_team.llm.types import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    Message,
    StreamChunk,
    TextBlock,
    ToolUseBlock,
)


class FakeBus:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self.before: dict[EventName, list[BeforeHook]] = defaultdict(list)

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    async def check(self, event: Event) -> HookOutcome:
        for hook in self.before[event.name]:
            outcome = await hook(event)
            if not outcome.allow:
                return outcome
        return HookOutcome()

    def on_before(self, name: EventName, hook: BeforeHook) -> None:
        self.before[name].append(hook)

    def on_after(self, name: EventName | None, hook: AfterHook) -> None:
        pass

    def names(self) -> list[EventName]:
        return [e.name for e in self.events]


class FakeTaskStore:
    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.tasks: dict[str, Task] = {}
        self.transitions: list[tuple[str, TaskStatus, TaskStatus, str]] = []

    async def create_run(self, run: Run) -> Run:
        self.runs[run.id] = run
        return run

    async def get_run(self, run_id: str) -> Run:
        return self.runs[run_id].model_copy(deep=True)

    async def update_run(self, run: Run) -> Run:
        self.runs[run.id] = run.model_copy(update={"updated_at": utcnow()})
        return self.runs[run.id]

    async def list_runs(self) -> list[Run]:
        return list(self.runs.values())

    async def create_task(self, task: Task) -> Task:
        self.tasks[task.id] = task.model_copy(deep=True)
        return task

    async def get_task(self, task_id: str) -> Task:
        return self.tasks[task_id].model_copy(deep=True)

    async def update_task(self, task: Task) -> Task:
        stored = self.tasks[task.id]
        self.tasks[task.id] = task.model_copy(update={"status": stored.status}, deep=True)
        return self.tasks[task.id].model_copy(deep=True)

    async def transition(self, task_id: str, to: TaskStatus, *, reason: str) -> Task:
        task = self.tasks[task_id]
        if not can_transition(task.status, to):
            raise InvalidTransition(f"{task_id}: {task.status} -> {to} ({reason})")
        self.transitions.append((task_id, task.status, to, reason))
        update: dict[str, Any] = {"status": to}
        if to is TaskStatus.RUNNING:
            update["started_at"] = utcnow()
        if to in (TaskStatus.DONE, TaskStatus.ESCALATED, TaskStatus.SUPERSEDED):
            update["finished_at"] = utcnow()
        self.tasks[task_id] = task.model_copy(update=update)
        return self.tasks[task_id].model_copy(deep=True)

    async def list_tasks(
        self, run_id: str, *, status: TaskStatus | None = None, phase: Phase | None = None
    ) -> list[Task]:
        return [
            t.model_copy(deep=True)
            for t in self.tasks.values()
            if t.run_id == run_id
            and (status is None or t.status is status)
            and (phase is None or t.phase is phase)
        ]

    async def ready_tasks(self, run_id: str) -> list[Task]:
        out = []
        for t in list(self.tasks.values()):
            if t.run_id != run_id:
                continue
            if t.status is TaskStatus.PENDING and all(
                self.tasks[d].status is TaskStatus.DONE for d in t.depends_on
            ):
                await self.transition(t.id, TaskStatus.READY, reason="deps done")
            if self.tasks[t.id].status is TaskStatus.READY:
                out.append(self.tasks[t.id].model_copy(deep=True))
        return out


class FakeArtifactStore:
    def __init__(self) -> None:
        self.items: dict[tuple[str, str, int], Artifact] = {}

    async def save(self, artifact: Artifact) -> ArtifactRef:
        versions = [v for (r, k, v) in self.items if r == artifact.run_id and k == artifact.kind]
        version = max(versions, default=0) + 1
        stored = artifact.model_copy(update={"version": version})
        self.items[(artifact.run_id, artifact.kind, version)] = stored
        return ArtifactRef(kind=artifact.kind, version=version)

    async def load(self, run_id: str, ref: ArtifactRef) -> Artifact:
        return self.items[(run_id, ref.kind, ref.version)]

    async def latest(self, run_id: str, kind: str) -> Artifact | None:
        versions = [v for (r, k, v) in self.items if r == run_id and k == kind]
        return self.items[(run_id, kind, max(versions))] if versions else None

    async def dependents(self, run_id: str, ref: ArtifactRef) -> list[ArtifactRef]:
        out: list[ArtifactRef] = []
        frontier = [ref]
        while frontier:
            cur = frontier.pop()
            for (r, k, v), art in self.items.items():
                if r == run_id and cur in art.built_from:
                    child = ArtifactRef(kind=k, version=v)
                    if child not in out:
                        out.append(child)
                        frontier.append(child)
        return out

    def kinds(self, run_id: str) -> list[str]:
        return sorted({k for (r, k, _) in self.items if r == run_id})


class FakeCheckpoints:
    def __init__(self) -> None:
        self.steps: dict[tuple[str, int], list[tuple[int, list[Message]]]] = defaultdict(list)

    async def save_step(
        self, task_id: str, attempt: int, step: int, messages: list[Message]
    ) -> None:
        self.steps[(task_id, attempt)].append((step, [m.model_copy(deep=True) for m in messages]))

    async def load_latest(self, task_id: str, attempt: int) -> tuple[int, list[Message]] | None:
        saved = self.steps.get((task_id, attempt))
        return saved[-1] if saved else None


class FakeDecisionLog:
    def __init__(self) -> None:
        self.items: list[Decision] = []

    async def record(self, decision: Decision) -> None:
        self.items.append(decision)

    async def list(self, run_id: str) -> list[Decision]:
        return [d for d in self.items if d.run_id == run_id]


class FakeFeedback:
    def __init__(self) -> None:
        self.items: list[tuple[str, Feedback]] = []

    async def add(self, run_id: str, feedback: Feedback) -> None:
        self.items.append((run_id, feedback))

    async def for_work(
        self, run_id: str, *, refs: Sequence[ArtifactRef], task_ids: Sequence[str]
    ) -> list[Feedback]:
        return [
            f
            for r, f in self.items
            if r == run_id
            and (f.artifact in refs or (f.task_id is not None and f.task_id in task_ids))
        ]


class FakeVerdicts:
    def __init__(self) -> None:
        self.items: list[Verdict] = []

    async def save(self, verdict: Verdict) -> None:
        self.items.append(verdict)

    async def latest_for_task(self, task_id: str) -> Verdict | None:
        found = [v for v in self.items if v.task_id == task_id]
        return found[-1] if found else None

    async def for_task(self, task_id: str) -> list[Verdict]:
        return [v for v in self.items if v.task_id == task_id]


class PassAuditor:
    """Always passes, unless a test sets ``fail_tasks`` to fail some task sweeps."""

    def __init__(self) -> None:
        self.fail_next: dict[str, list[Verdict]] = defaultdict(list)  # by output kind
        self.task_calls: list[str] = []
        self.runs = 0

    async def audit_task(self, task: Task, artifact: Artifact | None, run: Run) -> Verdict:
        self.task_calls.append(task.id)
        queued = self.fail_next.get(task.output_kind)
        if queued:
            return queued.pop(0).model_copy(update={"task_id": task.id, "run_id": run.id})
        return Verdict(run_id=run.id, scope=RuleScope.TASK, task_id=task.id, passed=True)

    async def audit_run(self, run: Run) -> Verdict:
        self.runs += 1
        return Verdict(run_id=run.id, scope=RuleScope.RUN, passed=True)

    async def precheck(
        self, tool: str, args: dict[str, Any], task: Task, run: Run
    ) -> PrecheckResult:
        return PrecheckResult(decision=PrecheckDecision.ALLOW)


Responder = Callable[[LLMRequest], list[ContentBlock] | str]


def reply(text: str) -> list[ContentBlock]:
    return [TextBlock(text=text)]


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> list[ContentBlock]:
    return [ToolUseBlock(id=call_id, name=name, arguments=args)]


def prompt_tokens(request: LLMRequest) -> int:
    """A provider-like prompt count: four characters of the sent JSON per token."""
    return max(1, len(sent_json(request.system, request.messages, request.tools)) // 4)


class FakeLLM:
    """Scripted model. ``script`` maps an agent to a queue of replies, or a responder
    function that gets the request. Records every request."""

    def __init__(
        self,
        script: dict[AgentName, list[list[ContentBlock] | str] | Responder] | None = None,
        *,
        budget: int = 100_000,
    ) -> None:
        self.script = script or {}
        self.requests: list[LLMRequest] = []
        self.budget = budget
        self.overrides: dict[str, Any] = {}
        #: The route the next call may take, as the gateway reports it.
        self.providers = ["fake"]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        entry = self.script.get(request.agent)
        if entry is None:
            raise AssertionError(f"no script for {request.agent}")
        if callable(entry):
            out = entry(request)
        else:
            if not entry:
                raise AssertionError(f"script for {request.agent} ran out")
            out = entry.pop(0)
        blocks = reply(out) if isinstance(out, str) else out
        stop = "tool_use" if any(isinstance(b, ToolUseBlock) for b in blocks) else "end_turn"
        return LLMResponse(
            content=blocks,
            stop_reason=stop,
            usage=Usage(input_tokens=prompt_tokens(request), output_tokens=5),
            provider="fake",
            model="fake-1",
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        response = await self.complete(request)
        yield StreamChunk(text=response.text, done=True, response=response)

    def context_budget(self, agent: AgentName, task_id: str | None = None) -> int:
        return self.budget

    def providers_for(self, agent: AgentName, task_id: str | None) -> list[str]:
        return self.providers

    def use_stronger_model(self, task_id: str) -> object:
        self.overrides[task_id] = "stronger"
        return "stronger"

    def set_override(self, task_id: str, route: Any) -> None:
        if route is None:
            self.overrides.pop(task_id, None)
        else:
            self.overrides[task_id] = route


def as_json(data: dict[str, Any]) -> str:
    return json.dumps(data)
