"""Working out what a user message means, from the run's state (docs/DESIGN.md,
"Questions as well as builds").

Users don't know the team's internals. They say "make it paginate" or "yes, Postgres",
so one orchestrator tool loop reads the run (the read-only control tools) and turns the
message into actions: answer a pending item, propose a change, start or stop a run,
keep a preference the user stated, and a reply in plain words. Code validates every
action call, and a bad call goes back to the model as a tool error. Actions are only
collected during the loop; the orchestrator carries them out after it ends, so a loop
that fails does nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from agent_team.agents.loop import LoopSettings, ToolLoop
from agent_team.agents.prompts import MESSAGES
from agent_team.audit.patterns import injection_markers
from agent_team.core.artifacts import PLAIN_KINDS, plain_kind
from agent_team.core.base import Contract
from agent_team.core.errors import ToolError
from agent_team.core.interfaces import (
    ArtifactStore,
    CheckpointStore,
    DecisionLog,
    EventBus,
    TaskStore,
    Tool,
    ToolContext,
    VerdictStore,
)
from agent_team.core.models import (
    SESSION,
    AgentName,
    Phase,
    RequestKind,
    Run,
    Task,
    TaskOrigin,
    ToolResult,
    new_id,
)
from agent_team.llm.types import LLMClient, Message, ToolSpec
from agent_team.orchestrator.base import ESCALATION, RUN_QUESTION
from agent_team.orchestrator.control import read_tools, required_str
from agent_team.orchestrator.replies import plain_agent

#: The reason on a user task that confirms a proposed change before anything reopens.
CONFIRM_CHANGE = "confirm_change"
#: Artifacts a change can start from, in pipeline order.
CHANGE_TARGETS = ("plan", "prd", "spec", "code_change")
#: Preferences one message may state, and the longest one kept.
MAX_PREFERENCES = 3
MAX_PREFERENCE_CHARS = 300
#: About six tool turns: each turn is a model step and a tool step, plus the reply.
SETTINGS = LoopSettings(max_steps=13, max_repairs=1)

#: A typed confirmation that agrees ("yes", "yes, go ahead", "sure, do it").
YES = re.compile(
    r"^\s*(yes|yep|yeah|sure|ok(ay)?|go ahead|do it|please do|approved?|lgtm|sounds good)\b",
    re.I,
)
#: A typed confirmation that declines ("no", "don't", "cancel that").
NO = re.compile(r"^\s*(no|nope|don'?t|do not|cancel|never ?mind|skip( it)?)\b", re.I)

#: The shape of every stored id (``task_3f9a1c2e4b5d6a7f``), whatever its prefix.
_ID = r"\b[a-z]+_[0-9a-f]{12,}\b"
#: An id, with the space before it, or wrapped alone in parentheses.
_IDS = re.compile(rf"[ \t]?\({_ID}\)|[ \t]?{_ID}")
_REFS = re.compile(rf"\b({'|'.join(PLAIN_KINDS)})@v\d+\b")


@dataclass
class Action:
    """One validated action call, carried out after the loop ends."""

    tool: Literal[
        "answer_pending", "request_change", "start_run", "stop_run", "remember_preference"
    ]
    args: dict[str, str]


@dataclass
class Interpretation:
    """What the loop made of a message: its actions in call order and its reply."""

    actions: list[Action] = field(default_factory=list)
    reply: str = ""


class MessageReply(Contract):
    """The loop's final output: the reply to show the user."""

    reply: str = Field(min_length=1)


def describe_pending(task: Task, parent: Task | None) -> str:
    """Who is asking, in plain words, for one item waiting on the user."""
    if task.reason == CONFIRM_CHANGE:
        return "confirming your change"
    if task.origin is TaskOrigin.REVIEW:
        return f"a review of the {task.phase.value} phase"
    if task.reason == RUN_QUESTION:
        return "the auditor"
    if task.reason == ESCALATION:
        return "a stuck step"
    if parent is not None:
        return plain_agent(parent.agent)
    return "the team"


def scrub(text: str, message: str) -> str:
    """User-facing text without internal names: ids are removed (with the space before
    them, or their parentheses), and artifact refs such as ``prd@v2`` become plain
    words unless the user named that kind themselves."""
    said = message.lower()

    def ref(m: re.Match[str]) -> str:
        """A ref to a kind the user named stays; any other becomes its plain name."""
        named = re.search(rf"\b{m.group(1)}\b", said)
        return m.group(0) if named else plain_kind(m.group(1))

    return _REFS.sub(ref, _IDS.sub("", text)).strip()


def _spec(name: str, description: str, props: dict[str, Any]) -> ToolSpec:
    """A tool spec whose every property is a required string."""
    return ToolSpec(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": props, "required": list(props)},
    )


class _Actions:
    """The action tools for one message. Each validates its call against the run as it
    stands and queues the action; nothing changes until the loop has ended."""

    def __init__(self, run: Run | None, pending: list[Task], targets: set[str]) -> None:
        """``pending`` are the user's waiting items; ``targets`` the artifact kinds a
        change could start from (produced, or being produced now)."""
        self.run = run
        self.pending = {t.id: t for t in pending}
        self.targets = targets
        self.actions: list[Action] = []

    def _queue(self, tool: str, **args: str) -> ToolResult:
        """Queue one validated action and confirm it to the model."""
        self.actions.append(Action(tool=tool, args=args))  # type: ignore[arg-type]
        return ToolResult(content=f"ok: {tool} will be carried out after your reply")

    def _done(self, tool: str) -> set[str]:
        """Task ids already answered by an earlier call in this message."""
        return {a.args["task_id"] for a in self.actions if a.tool == tool}

    def tools(self) -> dict[str, Tool]:
        """The action tools, keyed by name."""
        made = [
            _Tool(
                _spec(
                    "answer_pending",
                    "Answer one item waiting on the user, with the user's words for it. "
                    'For a review pass "approve" or the user\'s comment; for a change '
                    'confirmation pass "yes" or "no".',
                    {"task_id": {"type": "string"}, "text": {"type": "string"}},
                ),
                self.answer_pending,
                safe=False,
            ),
            _Tool(
                _spec(
                    "request_change",
                    "Propose a change to the run, starting from the first artifact it "
                    f"touches: one of {', '.join(CHANGE_TARGETS)}. Read them first. "
                    "Nothing reopens until the user confirms.",
                    {"text": {"type": "string"}, "target": {"type": "string"}},
                ),
                self.request_change,
                safe=False,
            ),
            _Tool(
                _spec(
                    "start_run",
                    "Start a new run: kind is build (software to make) or "
                    "research_question (a question to research, no code).",
                    {"request": {"type": "string"}, "kind": {"type": "string"}},
                ),
                self.start_run,
                safe=False,
            ),
            _Tool(
                _spec(
                    "stop_run",
                    "Stop the run in progress; it can be resumed later.",
                    {"reason": {"type": "string"}},
                ),
                self.stop_run,
                safe=False,
            ),
            _Tool(
                _spec(
                    "remember_preference",
                    "Keep a lasting preference the user stated (how they want the team "
                    "to work) for later runs, in one short sentence.",
                    {"text": {"type": "string"}},
                ),
                self.remember_preference,
                safe=False,
            ),
        ]
        return {t.spec.name: t for t in made}

    def answer_pending(self, args: dict[str, Any]) -> ToolResult:
        """The task must be waiting on the user and not answered yet in this message; a
        change confirmation takes exactly "yes" or "no", so the user's meaning is the
        model's call, not a regex's."""
        name = "answer_pending"
        task_id, text = required_str(args, "task_id", name), required_str(args, "text", name)
        task = self.pending.get(task_id)
        if task is None:
            raise ToolError(f"{task_id!r} is not waiting on the user", tool=name)
        if task_id in self._done(name):
            raise ToolError(f"{task_id!r} is already answered in this message", tool=name)
        if task.reason == CONFIRM_CHANGE and text.lower().rstrip(".!") not in ("yes", "no"):
            raise ToolError('pass "yes" or "no" for a change confirmation', tool=name)
        return self._queue(name, task_id=task_id, text=text)

    def request_change(self, args: dict[str, Any]) -> ToolResult:
        """There must be a run, and the target must be an artifact it has produced."""
        name = "request_change"
        text, target = required_str(args, "text", name), required_str(args, "target", name)
        if self.run is None:
            raise ToolError("no run is in progress; use start_run for new work", tool=name)
        if target not in self.targets:
            have = ", ".join(k for k in CHANGE_TARGETS if k in self.targets) or "none yet"
            raise ToolError(f"target must be an artifact this run has: {have}", tool=name)
        return self._queue(name, text=text, target=target)

    def start_run(self, args: dict[str, Any]) -> ToolResult:
        """One new run per message, of a kind a run can be."""
        name = "start_run"
        request, kind = required_str(args, "request", name), required_str(args, "kind", name)
        if kind not in (RequestKind.BUILD, RequestKind.RESEARCH_QUESTION):
            raise ToolError("kind must be build or research_question", tool=name)
        if any(a.tool == name for a in self.actions):
            raise ToolError("only one new run per message", tool=name)
        return self._queue(name, request=request, kind=kind)

    def stop_run(self, args: dict[str, Any]) -> ToolResult:
        """There must be a run to stop."""
        name = "stop_run"
        reason = required_str(args, "reason", name)
        if self.run is None:
            raise ToolError("no run is in progress", tool=name)
        return self._queue(name, reason=reason)

    def remember_preference(self, args: dict[str, Any]) -> ToolResult:
        """One short sentence, not already kept in this message, with nothing in it
        that reads like instructions to an agent, and a few per message at most."""
        name = "remember_preference"
        text = " ".join(required_str(args, "text", name).split())
        if len(text) > MAX_PREFERENCE_CHARS:
            raise ToolError(f"keep it under {MAX_PREFERENCE_CHARS} characters", tool=name)
        if injection_markers(text):
            raise ToolError("a preference can't hold instructions for the agents", tool=name)
        kept = [a.args["text"].lower() for a in self.actions if a.tool == name]
        if text.lower() in kept:
            raise ToolError("that preference is already kept in this message", tool=name)
        if len(kept) >= MAX_PREFERENCES:
            raise ToolError(f"at most {MAX_PREFERENCES} preferences per message", tool=name)
        return self._queue(name, text=text)


class _Tool:
    """Adapts one validating method to the Tool protocol."""

    def __init__(self, spec: ToolSpec, call: Any, *, safe: bool) -> None:
        """``safe`` says whether a resumed loop may repeat the call."""
        self.spec = spec
        self._call = call
        self.safe_to_repeat = safe

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Validate and queue; a bad call raises ToolError for the model to read."""
        result: ToolResult = self._call(args)
        return result


def prompt(message: str, run: Run | None, pending: list[tuple[Task, str]]) -> str:
    """The loop's first message: the run's status, what waits on the user, the message."""
    if run is None:
        state = "No run is in progress."
    else:
        state = f"Run in progress: {run.title!r}, status {run.status.value}. Request: {run.request}"
    if pending:
        items = "\n".join(f"- task_id={t.id}, asked by {who}: {t.objective}" for t, who in pending)
        waiting = f"Waiting on the user:\n{items}"
    else:
        waiting = "Nothing is waiting on the user."
    return f"{state}\n\n{waiting}\n\nThe user's message:\n{message}"


async def interpret(
    message: str,
    run: Run | None,
    pending: list[tuple[Task, str]],
    targets: set[str],
    *,
    tasks: TaskStore,
    artifacts: ArtifactStore,
    decisions: DecisionLog,
    verdicts: VerdictStore,
    checkpoints: CheckpointStore,
    bus: EventBus,
    llm: LLMClient,
    workspace: str,
) -> Interpretation | None:
    """Run the message loop. Returns the actions and reply, or None if the loop failed,
    in which case nothing may be acted on."""
    actions = _Actions(run, [t for t, _ in pending], targets)
    context_run = run or Run(id=SESSION, title=SESSION, request="", workspace=workspace)
    probe = Task(
        id=new_id("msg"),
        run_id=context_run.id,
        phase=Phase.PLANNING,
        agent=AgentName.ORCHESTRATOR,
        objective=message[:200],
        output_kind="message_reply",
    )
    tools = {**read_tools(tasks, artifacts, decisions, verdicts), **actions.tools()}
    loop: ToolLoop[MessageReply] = ToolLoop(
        agent=AgentName.ORCHESTRATOR,
        system=MESSAGES,
        output_model=MessageReply,
        llm=llm,
        bus=bus,
        checkpoints=checkpoints,
        tools=tools,
        tool_ctx=ToolContext(run=context_run, task=probe, bus=bus),
        run=context_run,
        task_id=probe.id,
        attempt=1,
        settings=SETTINGS,
    )
    outcome = await loop.run_loop(Message.user(prompt(message, run, pending)))
    if outcome.output is None:
        return None
    return Interpretation(actions=actions.actions, reply=outcome.output.reply)
