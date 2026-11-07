"""Working out what a user message means, from the run's state (docs/DESIGN.md,
"Questions as well as builds").

Users don't know the team's internals. They say "make it paginate" or "yes, Postgres",
so one orchestrator tool loop reads the run (the read-only control tools) and turns the
message into actions: answer a pending item, propose a change, start or stop a run,
keep a preference the user stated, and a reply in plain words. The reply also sorts
the message (a build, a research question, a question about the run, ...), and code
holds the actions to that sort: a research question gets a run, never an answer from
the model's memory. Code validates every action call, and a bad call goes back to the
model as a tool error. Actions are only collected during the loop; the orchestrator
carries them out after it ends, so a loop that fails does nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from agent_team.agents.loop import Check, LoopSettings, ToolLoop
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
    RunStatus,
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
#: The reason on a user task that confirms a stop the user's message didn't ask for.
CONFIRM_STOP = "confirm_stop"
#: User tasks answered with exactly "yes" or "no".
CONFIRMATIONS = (CONFIRM_CHANGE, CONFIRM_STOP)
#: Artifacts a change can start from, in pipeline order.
CHANGE_TARGETS = ("plan", "prd", "spec", "code_change")
#: Preferences one message may state, and the longest one kept.
MAX_PREFERENCES = 3
MAX_PREFERENCE_CHARS = 300
#: Five tool turns (each a model step and a tool step), then a turn that can only
#: reply, and room to repair that reply once.
SETTINGS = LoopSettings(max_steps=15, max_repairs=1, final_reply=True)

#: A typed confirmation that agrees ("yes", "yes, go ahead", "sure, do it").
_YES_WORDS = r"(yes|yep|yeah|sure|ok(ay)?|go ahead|do it|please do|approved?|lgtm|sounds good)"
_NO_WORDS = r"(no|nope|nah|don'?t|do not|cancel|never ?mind|skip( it)?)"
YES = re.compile(rf"^\s*{_YES_WORDS}\b", re.I)
#: A typed confirmation that declines ("no", "don't", "cancel that").
NO = re.compile(rf"^\s*{_NO_WORDS}\b", re.I)
#: A message that is only a yes or a no ("yes", "Nope."), with nothing else said.
BARE_YES_NO = re.compile(rf"^\s*(y|n|{_YES_WORDS}|{_NO_WORDS})[\s.!]*$", re.I)
#: Verbs that stop a run when named with it ("cancel the run", "end this run").
_STOP_VERB = r"(stop|halt|abort|cancel|quit|end|kill)"
#: A whole message that only says stop: "stop", "Stop.", "please stop it now". Not
#: "cancel that", which declines a confirmation.
_STOP_PHRASE = re.compile(
    r"^\W*(please\s+)?(stop|halt|abort|quit)(\s+(it|this|that|now|please|everything|"
    r"(the|this|that)\s+(whole\s+)?(run|work)))*\W*$",
    re.I,
)
#: A stop of the run inside a longer message: "stop the run and build a todo API".
_STOP_RUN = re.compile(rf"\b{_STOP_VERB}\s+(the|this|that)\s+(whole\s+)?run\b", re.I)
#: A message that says not to stop ("don't stop", "no need to halt").
_NOT_STOP = re.compile(rf"\b(don'?t|do not|never|no need to|not)\s+{_STOP_VERB}\b", re.I)
#: An argument that holds no real value: a placeholder or filler the model left in.
_PLACEHOLDER = re.compile(
    r"^\W*(placeholder|tbd|todo|fixme|n/?a|none|null|nil|undefined|unknown|dummy|"
    r"example|lorem ipsum|reason|text|x+|\.+|-+|_+|<[^>]*>|\{[^}]*\}|\[[^\]]*\])\W*$",
    re.I,
)

#: How the reply sorts a message. A message can be several at once.
MessageKind = Literal[
    "build",  # new software to build
    "research_question",  # a question to research and answer with sources
    "run_question",  # a question about a run: its tasks, sources, retries
    "change",  # a change to the run's result
    "answer",  # answers or approves something waiting on the user
    "stop",  # asks to stop the run
    "preference",  # states a lasting preference
    "unclear",  # can't tell; the reply asks one clarifying question
]
#: Kinds that are new work, each a kind of run.
NEW_RUN_KINDS: tuple[str, ...] = ("build", "research_question")
#: Kinds that need a run to act on.
RUN_KINDS = frozenset({"run_question", "change", "answer", "stop"})
#: The reply when a message is about a run but none is open.
NO_RUN_REPLY = (
    "No run is open in this session, so there is nothing to accept, change or ask about "
    "yet. Describe what to build, or ask a question and I'll research it with sources. "
    "To look at an earlier run, resume it: agent-team resume <run id>."
)
#: The reply when code starts the run a message needs, by kind.
STARTING = {
    "build": "Starting a run to build that.",
    "research_question": "I'll research that and answer with sources.",
}

#: The shape of every stored id (``task_3f9a1c2e4b5d6a7f``), whatever its prefix.
_ID = r"\b[a-z]+_[0-9a-f]{12,}\b"
#: An id, with the space before it, or wrapped alone in parentheses.
_IDS = re.compile(rf"[ \t]?\({_ID}\)|[ \t]?{_ID}")
_REFS = re.compile(rf"\b({'|'.join(PLAIN_KINDS)})@v\d+\b")


@dataclass
class Action:
    """One validated action call, carried out after the loop ends. ``by`` is "code"
    for an action code added because the message's sort needs it."""

    tool: Literal[
        "answer_pending", "request_change", "start_run", "stop_run", "remember_preference"
    ]
    args: dict[str, str]
    by: Literal["model", "code"] = "model"


@dataclass
class Interpretation:
    """What the loop made of a message: its actions in call order and its reply."""

    actions: list[Action] = field(default_factory=list)
    reply: str = ""


class MessageReply(Contract):
    """The loop's final output: what the message is, and the reply to show the user."""

    kinds: list[MessageKind] = Field(min_length=1)
    reply: str = Field(min_length=1)


def says_stop(text: str) -> bool:
    """Whether the user's own words ask to stop the run: only "stop" (or the like),
    or a stop of the run by name. A question ("why did it stop?"), a negation, or a
    stop of something else ("stop using SQLite") is not one."""
    text = text.strip()
    if text.endswith("?") or _NOT_STOP.search(text):
        return False
    return bool(_STOP_PHRASE.match(text) or _STOP_RUN.search(text))


def describe_pending(task: Task, parent: Task | None) -> str:
    """Who is asking, in plain words, for one item waiting on the user."""
    if task.reason == CONFIRM_CHANGE:
        return "confirming your change"
    if task.reason == CONFIRM_STOP:
        return "confirming a stop"
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

    return renumber(_REFS.sub(ref, _IDS.sub("", text)).strip())


#: A numbered list item: its indent, number, and the rest from the marker's end.
_ITEM = re.compile(r"^(\s*)(\d+)([.)]\s.*)$")


def renumber(text: str) -> str:
    """Each numbered list numbered by its items' places, from the first item's number,
    so a model's "1, 2, 2, 3" reads 1, 2, 3, 4. A list goes on across blank lines and
    deeper lines (an item's own text or a nested list), and ends at a line as shallow
    as its items that isn't one. Fenced code blocks are left as they are."""
    counters: dict[int, int] = {}  # indent -> the number the next item there gets
    fenced = False
    out: list[str] = []
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            counters = {}
        m = None if fenced else _ITEM.match(line)
        indent = len(line) - len(line.lstrip())
        if fenced or not line.strip():
            out.append(line)
            continue
        counters = {i: n for i, n in counters.items() if i < indent or (m and i == indent)}
        if m is None:
            out.append(line)
            continue
        number = counters.get(indent, int(m.group(2)))
        counters[indent] = number + 1
        out.append(f"{m.group(1)}{number}{m.group(3)}")
    return "\n".join(out)


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

    def __init__(
        self, message: str, run: Run | None, pending: list[Task], targets: set[str]
    ) -> None:
        """``message`` is the user's; ``pending`` are the user's waiting items;
        ``targets`` the artifact kinds a change could start from (produced, or being
        produced now)."""
        self.message = message
        self.run = run
        self.pending = {t.id: t for t in pending}
        self.targets = targets
        self.actions: list[Action] = []

    def _arg(self, args: dict[str, Any], key: str, tool: str) -> str:
        """A required string argument with a real value: a placeholder such as
        "placeholder", "TBD" or "<reason>" is refused, unless the user said it."""
        value = required_str(args, key, tool)
        if _PLACEHOLDER.match(value) and value.lower() not in self.message.lower():
            raise ToolError(
                f"argument {key!r} is a placeholder ({value!r}); pass the real value, or "
                "don't make this call",
                tool=tool,
            )
        return value

    def check(self, out: MessageReply) -> list[str]:
        """Hold the actions to the reply's sort: new work with a run open needs
        ``start_run`` (with none open, code starts it after the loop), something
        waiting that the message answers needs ``answer_pending``, and an unclear
        message gets a clarifying question. Returns the errors for the model."""
        errors: list[str] = []
        tools = {a.tool for a in self.actions}
        kinds = set(out.kinds)
        new = [k for k in NEW_RUN_KINDS if k in kinds]
        if self.run is not None and new and "start_run" not in tools:
            errors.append(
                f"you sorted the message as {new[0]}: call start_run with kind {new[0]}, "
                "or sort it as what it is (a question about the run is run_question)"
            )
        if "answer" in kinds and self.pending and "answer_pending" not in tools:
            errors.append("you sorted the message as an answer: call answer_pending for it")
        if kinds == {"unclear"} and not self.actions and "?" not in out.reply:
            errors.append("for an unclear message, reply with one short clarifying question")
        return errors

    def settle(self, out: MessageReply) -> str:
        """The reply once code has acted on the sort. With no run open, new work gets
        its run here (the model may have answered from memory instead), and a message
        about a run gets a plain "no run" reply."""
        if self.run is not None or any(a.tool == "start_run" for a in self.actions):
            return out.reply
        if "unclear" in out.kinds:
            return out.reply  # the model is asking what the user means
        new = next((k for k in NEW_RUN_KINDS if k in out.kinds), None)
        if new is not None:
            self.actions.append(
                Action(tool="start_run", args={"request": self.message, "kind": new}, by="code")
            )
            return STARTING[new]
        if RUN_KINDS & set(out.kinds):
            return NO_RUN_REPLY
        return out.reply

    def _queue(self, tool: str, **args: str) -> ToolResult:
        """Queue one validated action and confirm it to the model."""
        self.actions.append(Action(tool=tool, args=args))  # type: ignore[arg-type]
        return ToolResult(content=f"ok: {tool} will be carried out after your reply")

    def _done(self, tool: str) -> set[str]:
        """Task ids already answered by an earlier call in this message."""
        return {a.args["task_id"] for a in self.actions if a.tool == tool}

    def tools(self) -> dict[str, Tool]:
        """The action tools, keyed by name. answer_pending is offered only while
        something waits on the user, so an old, answered question can't be answered
        again."""
        made = [
            _Tool(
                _spec(
                    "answer_pending",
                    "Answer one item waiting on the user, with the user's words for it. "
                    'For a review pass "approve" or the user\'s comment; for a change or '
                    'stop confirmation pass "yes" or "no".',
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
                    "Stop the run in progress, only when the user asks to stop; it can "
                    "be resumed later. A finished run can't be stopped. Unless the "
                    "user's message says stop, the user is asked to confirm first.",
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
        return {t.spec.name: t for t in made if self.pending or t.spec.name != "answer_pending"}

    def answer_pending(self, args: dict[str, Any]) -> ToolResult:
        """The task must be waiting on the user and not answered yet in this message; a
        change confirmation takes exactly "yes" or "no", so the user's meaning is the
        model's call, not a regex's."""
        name = "answer_pending"
        task_id, text = required_str(args, "task_id", name), self._arg(args, "text", name)
        task = self.pending.get(task_id)
        if task is None:
            raise ToolError(f"{task_id!r} is not waiting on the user", tool=name)
        if task_id in self._done(name):
            raise ToolError(f"{task_id!r} is already answered in this message", tool=name)
        if task.reason in CONFIRMATIONS and text.lower().rstrip(".!") not in ("yes", "no"):
            raise ToolError('pass "yes" or "no" for a confirmation', tool=name)
        if says_stop(text) and not says_stop(self.message):
            raise ToolError("the user didn't ask to stop; pass their own words", tool=name)
        return self._queue(name, task_id=task_id, text=text)

    def request_change(self, args: dict[str, Any]) -> ToolResult:
        """There must be a run, and the target must be an artifact it has produced."""
        name = "request_change"
        text, target = self._arg(args, "text", name), required_str(args, "target", name)
        if self.run is None:
            raise ToolError("no run is in progress; use start_run for new work", tool=name)
        if target not in self.targets:
            have = ", ".join(k for k in CHANGE_TARGETS if k in self.targets) or "none yet"
            raise ToolError(f"target must be an artifact this run has: {have}", tool=name)
        return self._queue(name, text=text, target=target)

    def start_run(self, args: dict[str, Any]) -> ToolResult:
        """One new run per message, of a kind a run can be."""
        name = "start_run"
        request, kind = self._arg(args, "request", name), required_str(args, "kind", name)
        if kind not in (RequestKind.BUILD, RequestKind.RESEARCH_QUESTION):
            raise ToolError("kind must be build or research_question", tool=name)
        if any(a.tool == name for a in self.actions):
            raise ToolError("only one new run per message", tool=name)
        return self._queue(name, request=request, kind=kind)

    def stop_run(self, args: dict[str, Any]) -> ToolResult:
        """There must be a run that is still going, and a real reason; one stop per
        message. Whether the user confirms first is decided when it is carried out."""
        name = "stop_run"
        reason = self._arg(args, "reason", name)
        if self.run is None:
            raise ToolError("no run is in progress", tool=name)
        if self.run.status is RunStatus.DONE:
            raise ToolError("the run is finished, so there is nothing to stop", tool=name)
        if self.run.status is RunStatus.STOPPED:
            raise ToolError("the run is already stopped", tool=name)
        if any(a.tool == name for a in self.actions):
            raise ToolError("the stop is already queued in this message", tool=name)
        return self._queue(name, reason=reason)

    def remember_preference(self, args: dict[str, Any]) -> ToolResult:
        """One short sentence, not already kept in this message, with nothing in it
        that reads like instructions to an agent, and a few per message at most."""
        name = "remember_preference"
        text = " ".join(self._arg(args, "text", name).split())
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
        waiting = (
            "Nothing is waiting on the user, so there is no answer_pending tool: questions "
            "and escalations in the run's history are already answered."
        )
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
    actions = _Actions(message, run, [t for t, _ in pending], targets)
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
        check=_reply_check(actions),
    )
    outcome = await loop.run_loop(Message.user(prompt(message, run, pending)))
    if outcome.output is None:
        return None
    reply = actions.settle(outcome.output)
    return Interpretation(actions=actions.actions, reply=reply)


def _reply_check(actions: _Actions) -> Check:
    """The loop's check of its final reply against the actions it queued."""

    async def check(out: Any) -> list[str]:
        """Errors for the model when the reply's sort and its actions disagree."""
        assert isinstance(out, MessageReply)
        return actions.check(out)

    return check
