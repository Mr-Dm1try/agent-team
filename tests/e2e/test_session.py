"""A terminal session over the real team with a scripted model: the review of a
finished run, a change request that re-runs it and shows what changed, and the drive
lock the session holds while it works."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agents.scenario import (
    PRD,
    _last_is_tool_result,
    message_reply,
    orchestrator_responder,
    read_message,
    tool_call,
)
from rich.console import Console

from agent_team import cli
from agent_team.app import App
from agent_team.core.control import DriveLock
from agent_team.core.models import AgentName, DecisionKind, Event, EventName, RunStatus
from agent_team.core.tracing import read_events
from agent_team.llm.types import ContentBlock, LLMRequest, ToolUseBlock
from agent_team.orchestrator.conversation import HELD_REPLY
from agent_team.ui.session import STOPPING, Session, StdinLines
from e2e.team import asking_script, e2e_app, e2e_config, e2e_script, engineer_with_tests

PAGED_PRD = dict(
    PRD,
    requirements=[dict(PRD["requirements"][0], text="List books, 20 per page")],
)


def _lines(*lines: str | None) -> Iterator[str | None]:
    """The user's lines, then the end of input."""
    yield from lines
    while True:
        yield None


class Typist:
    """Types the scripted lines into the session, and notes whether the run was locked
    at each prompt and at each model call."""

    def __init__(self, app: App, *lines: str | None) -> None:
        """Type ``lines`` into a session over ``app``."""
        self.app = app
        self.lines = _lines(*lines)
        self.at_prompt: list[bool] = []
        self.at_call: list[bool] = []
        app.bus.on_after(EventName.LLM_CALL_AFTER, self._on_call)

    def _driven(self, run_id: str) -> bool:
        """Whether some session holds the run's lock."""
        return DriveLock(self.app.runs_dir).is_driven(run_id)

    async def _on_call(self, event: Event) -> None:
        """Note the lock at a model call of the run."""
        if event.run_id != "session":
            self.at_call.append(self._driven(event.run_id))

    async def read_line(self) -> str | None:
        """The next line; also note the lock while the session waits on it."""
        runs = await self.app.stores.tasks.list_runs()
        if runs:
            self.at_prompt.append(self._driven(runs[0].id))
        return next(self.lines)


def _paging_script() -> dict[AgentName, Any]:
    """The team, where the orchestrator turns "paginate" into a change to the PRD, and
    Product and the Engineer write a new version when asked again."""
    script = e2e_script()
    default = orchestrator_responder()

    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """Propose the change for "paginate"; otherwise answer like the default."""
        said, _ = read_message(request)
        if said == "paginate the list":
            if _last_is_tool_result(request):
                return message_reply("I can add pagination.", "change")
            return tool_call("request_change", {"text": "paginate the list", "target": "prd"})
        return default(request)

    script[AgentName.ORCHESTRATOR] = orchestrator_responder(message=message)
    prds: list[int] = []

    def product(request: LLMRequest) -> str:
        """The plain PRD first, the paged one after."""
        prds.append(1)
        return json.dumps(PRD if len(prds) == 1 else PAGED_PRD)

    script[AgentName.PRODUCT] = product
    writes: list[int] = []

    def engineer(request: LLMRequest) -> Any:
        """The e2e Engineer, writing the paged app.py on its second version."""
        out = engineer_with_tests(request)
        if isinstance(out, list) and out[0].name == "write_file":
            writes.append(1)
            if len(writes) > 1:
                args = {"path": "app.py", "content": "print('books, 20 per page')\n"}
                return [ToolUseBlock(id="call_write", name="write_file", arguments=args)]
        return out

    script[AgentName.ENGINEER] = engineer
    return script


def _console() -> tuple[Console, io.StringIO]:
    """A plain console that writes to a buffer."""
    out = io.StringIO()
    return Console(file=out, width=160, color_system=None), out


async def test_a_change_request_shows_the_section_and_code_diffs(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path, script=_paging_script())
    console, out = _console()
    typist = Typist(app, "paginate the list", "yes", "accept")
    with app:
        session = Session(app, console, plain=True, read_line=typist.read_line, listen=False)
        code = await session.start("Build a books API")
        (run,) = await app.stores.tasks.list_runs()
    text = out.getvalue()
    assert code == 0, text
    # The first review: the summary, then the prompt.
    assert "Audit report:\n  passed" in text
    assert "Acceptance criteria (QA):\n  ✓ AC-1.1 pass: app.py:1" in text
    assert "Tests:\n" in text and "`pytest -q` 1 passed, 0 failed, exit 0" in text
    # The change was confirmed, re-run, and shown as a diff.
    assert "Go ahead? (yes or no)" in text
    changed = text.split("What changed:\n", 1)[1]
    assert "prd v1 → v2:" in changed
    assert '-  "text": "List books"' in changed
    assert '+  "text": "List books, 20 per page"' in changed
    assert "code: code_change v1 → v2:" in changed
    assert "-print('books')" in changed and "+print('books, 20 per page')" in changed
    assert text.rstrip().endswith(
        "is done.\nOutside any run: 2 model calls · fake 1.9k in / 19 out · $0.0000"
    )
    assert "Accepted. Run" in text
    # The session held the run's lock while the team worked, and let go at prompts.
    assert typist.at_call and all(typist.at_call)
    assert typist.at_prompt and not any(typist.at_prompt)
    assert not DriveLock(app.runs_dir).is_driven(run.id)


async def test_accepting_right_away_shows_no_diff(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path)
    console, out = _console()
    typist = Typist(app, "lgtm")
    with app:
        code = await Session(
            app, console, plain=True, read_line=typist.read_line, listen=False
        ).start("Build a books API")
    text = out.getvalue()
    assert code == 0, text
    assert "What changed:" not in text
    assert "Type 'accept' to finish, or describe a change." in text
    assert "Accepted. Run" in text


async def test_a_question_at_the_review_prompt_brings_the_prompt_back(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path)
    console, out = _console()
    typist = Typist(app, "what did you build?", "accept")
    with app:
        code = await Session(
            app, console, plain=True, read_line=typist.read_line, listen=False
        ).start("Build a books API")
    text = out.getvalue()
    assert code == 0, text
    after = text.split("Noted.", 1)[1]
    assert "Type 'accept' to finish, or describe a change." in after
    assert "Accepted. Run" in after


async def test_a_stop_while_the_session_waits_on_the_user_stops_the_run(
    tmp_path: Path,
) -> None:
    app = await e2e_app(tmp_path, script=asking_script(), review="questions_only")
    console, out = _console()
    typist = Typist(app, "20")
    config = e2e_config(tmp_path, review="questions_only")

    async def stop_then_answer() -> str | None:
        """Run ``agent-team stop`` from another session, then type the answer."""
        (run,) = await app.stores.tasks.list_runs()
        assert await cli._stop(config, run.id, "lunch") == 0
        return await typist.read_line()

    with app:
        session = Session(app, console, plain=True, read_line=stop_then_answer, listen=False)
        code = await session.start("Build a books API")
        (run,) = await app.stores.tasks.list_runs()
        stops = [d for d in await app.stores.decisions.list(run.id) if d.kind is DecisionKind.STOP]
    text = out.getvalue()
    assert code == 1, text
    assert "The run was stopped before I could read that." in text
    assert run.status is RunStatus.STOPPED
    assert [d.reason for d in stops] == ["lunch"]


async def test_a_stop_while_the_team_works_is_logged_with_its_reason(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path)
    console, out = _console()

    async def stop_at_the_code(event: Event) -> None:
        """Run ``agent-team stop`` from another session once the Engineer starts."""
        if event.agent is AgentName.ENGINEER and event.data.get("to") == "running":
            config = e2e_config(tmp_path)
            assert DriveLock(app.runs_dir).is_driven(event.run_id)
            assert await cli._stop(config, event.run_id, "lunch") == 0

    app.bus.on_after(EventName.TASK_STATE_CHANGED, stop_at_the_code)
    with app:
        code = await Session(
            app, console, plain=True, read_line=Typist(app).read_line, listen=False
        ).start("Build a books API")
        (run,) = await app.stores.tasks.list_runs()
        stops = [d for d in await app.stores.decisions.list(run.id) if d.kind is DecisionKind.STOP]
    assert code == 1, out.getvalue()
    assert run.status is RunStatus.STOPPED
    assert len(stops) == 1 and "lunch" in stops[0].reason


class MidRunTypist:
    """Types ``mid`` while the team works, once the Engineer starts, and then each of
    ``lines`` only once the session is at a prompt (its output ends with "> ")."""

    def __init__(self, app: App, out: io.StringIO, mid: str, *lines: str | None) -> None:
        """Watch ``app``'s bus for the Engineer and ``out`` for the prompt."""
        self.out = out
        self.mid: str | None = mid
        self.lines = _lines(*lines)
        self.seen = 0  # how much output there was when the last line was typed
        self.engineer = asyncio.Event()
        app.bus.on_after(EventName.TASK_STATE_CHANGED, self._on_state)

    async def _on_state(self, event: Event) -> None:
        """Let the typed line through when the Engineer first starts, and give the
        session time to read it before the Engineer's next model call, as a real
        model call would."""
        engineer = event.agent is AgentName.ENGINEER and event.data.get("to") == "running"
        if engineer and not self.engineer.is_set():
            self.engineer.set()
            await asyncio.sleep(0.05)

    async def read_line(self) -> str | None:
        """The mid-run line first, then the prompt lines."""
        async with asyncio.timeout(10):
            if self.mid is not None:
                await self.engineer.wait()
                line, self.mid = self.mid, None
                return line
            while not self._at_prompt():  # noqa: ASYNC110 (the console has no event)
                await asyncio.sleep(0.005)
            self.seen = len(self.out.getvalue())
            return next(self.lines)

    def _at_prompt(self) -> bool:
        """Whether the session printed a new prompt since the last line was typed."""
        text = self.out.getvalue()
        return len(text) > self.seen and text.endswith("> ")


async def test_typing_stop_while_the_team_works_stops_the_run(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path)
    console, out = _console()
    calls: list[str] = []

    async def count(event: Event) -> None:
        """Note each agent's model calls."""
        calls.append(event.agent.value if event.agent else "-")

    app.bus.on_after(EventName.LLM_CALL_AFTER, count)
    typist = MidRunTypist(app, out, "stop the run")
    with app:
        code = await Session(app, console, plain=True, read_line=typist.read_line).start(
            "Build a books API"
        )
        (run,) = await app.stores.tasks.list_runs()
        stops = [d for d in await app.stores.decisions.list(run.id) if d.kind is DecisionKind.STOP]
        qa = [t for t in await app.stores.tasks.list_tasks(run.id) if t.agent is AgentName.QA]
    text = out.getvalue()
    assert code == 1, text
    assert STOPPING in text
    assert run.status is RunStatus.STOPPED
    assert len(stops) == 1 and "typed stop" in stops[0].reason
    # Stopped at the Engineer's first model call: it never answered, and QA never ran.
    assert "engineer" not in calls
    assert qa and all(t.started_at is None for t in qa)


async def test_a_question_typed_while_the_team_works_is_answered_under_the_view(
    tmp_path: Path,
) -> None:
    script = e2e_script()
    default = orchestrator_responder()

    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """Answer the progress question; otherwise answer like the default."""
        said, _ = read_message(request)
        if said == "how is it going?":
            return message_reply("The engineer is writing the code now.", "run_question")
        return default(request)

    script[AgentName.ORCHESTRATOR] = orchestrator_responder(message=message)
    app = await e2e_app(tmp_path, script=script)
    console, out = _console()
    typist = MidRunTypist(app, out, "how is it going?", "accept")
    with app:
        code = await Session(app, console, plain=True, read_line=typist.read_line).start(
            "Build a books API"
        )
        (run,) = await app.stores.tasks.list_runs()
    text = out.getvalue()
    assert code == 0, text
    assert run.status is RunStatus.DONE
    # The answer printed while the team worked, before the run finished.
    answered = text.index("The engineer is writing the code now.")
    assert answered < text.index(f"run {run.id} finished")
    said = [e for e in read_events(app.runs_dir, run.id) if e.name is EventName.USER_MESSAGE]
    assert said[0].data == {"text": "how is it going?", "while_working": True}
    assert "Accepted. Run" in text


async def test_a_change_typed_while_the_team_works_waits_until_it_stops(
    tmp_path: Path,
) -> None:
    app = await e2e_app(tmp_path, script=_paging_script())
    console, out = _console()
    typist = MidRunTypist(app, out, "paginate the list", "yes", "accept")
    with app:
        code = await Session(app, console, plain=True, read_line=typist.read_line).start(
            "Build a books API"
        )
    text = out.getvalue()
    assert code == 0, text
    held = text.index(HELD_REPLY)
    # Held while the team worked, then sent once the run was done and shown.
    assert held < text.index("Audit report:") < text.index("> paginate the list")
    assert "Go ahead? (yes or no)" in text
    assert "prd v1 → v2:" in text.split("What changed:\n", 1)[1]
    assert "Accepted. Run" in text


async def test_a_reply_while_the_team_works_prints_under_the_live_tree(tmp_path: Path) -> None:
    script = e2e_script()
    default = orchestrator_responder()

    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """Answer the progress question; otherwise answer like the default."""
        said, _ = read_message(request)
        if said == "how is it going?":
            return message_reply("The engineer is writing the code now.", "run_question")
        return default(request)

    script[AgentName.ORCHESTRATOR] = orchestrator_responder(message=message)
    app = await e2e_app(tmp_path, script=script)
    out = io.StringIO()
    console = Console(file=out, width=120, color_system=None, force_terminal=True)
    typist = MidRunTypist(app, out, "how is it going?", "accept")
    with app:
        code = await Session(app, console, read_line=typist.read_line).start("Build a books API")
    text = out.getvalue()
    assert code == 0, text
    # The tree printed once when the view stopped for the reply, then the reply.
    reply = text.index("The engineer is writing the code now.")
    assert text.rfind("run run_", 0, reply) >= 0 and "engineer" in text[:reply]
    assert "Accepted. Run" in text[reply:]


async def test_stdin_lines_are_read_on_a_thread(monkeypatch: Any) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("first\n  second  \n"))
    read = StdinLines()
    assert [await read(), await read(), await read()] == ["first", "second", None]
