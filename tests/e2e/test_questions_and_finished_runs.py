"""A research question asked with no run open, and a finished run looked at again: the
question becomes a research run whose answer shows its sources, and resuming a done
run re-runs nothing."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agents.scenario import (
    BRIEF,
    _last_is_tool_result,
    message_reply,
    orchestrator_responder,
    read_message,
)
from rich.console import Console

from agent_team.agents.base import render_envelope
from agent_team.agents.planner import kind_from_objective
from agent_team.app import App
from agent_team.core.models import (
    AgentName,
    DecisionKind,
    EventName,
    Feedback,
    FeedbackSource,
    RunStatus,
    TaskStatus,
)
from agent_team.core.tracing import read_events
from agent_team.llm.types import ContentBlock, LLMRequest, ToolUseBlock
from agent_team.orchestrator.messages import NO_RUN_REPLY, STARTING
from agent_team.ui.session import Session
from e2e.team import e2e_app, e2e_script

QUESTION = "How do you run a FastMCP server over stdio? Answer with sources."
FROM_MEMORY = "You call mcp.run(transport='stdio'). I wrote this from memory."
SOURCE = {"ref": "notes.md", "title": "Project notes"}
ANSWER = {
    "summary": "Run it with the stdio transport.",
    "question": QUESTION,
    "answer": "Call mcp.run(transport='stdio') on the FastMCP server.",
    "sources": [SOURCE],
}
PLAN = {
    "summary": "One question.",
    "request_kind": "research_question",
    "asks": [{"id": "ASK-1", "text": "Run FastMCP over stdio"}],
    "research_questions": [{"question": "How does FastMCP run over stdio?", "ask": "ASK-1"}],
}


def _researcher(request: LLMRequest) -> Any:
    """Open the notes, then write the brief or the answer, each citing the notes."""
    if not _last_is_tool_result(request):
        return [ToolUseBlock(id="call_notes", name="read_file", arguments={"path": "notes.md"})]
    text = getattr(request.messages[0].content[0], "text", "")
    if "Produce: research_answer" in text:
        return json.dumps(ANSWER)
    return json.dumps(BRIEF)


def _from_memory(request: LLMRequest) -> list[ContentBlock] | str:
    """A careless message loop: it sorts the question right but answers it itself, and
    sorts the other no-run messages as questions about a run."""
    said, _ = read_message(request)
    if said == QUESTION:
        return message_reply(FROM_MEMORY, "research_question")
    if said == "accept":
        return message_reply("Accepted.", "answer")
    return message_reply("It opened two pages.", "run_question")


def _script(message: Any) -> dict[AgentName, Any]:
    """The team for a research question, with ``message`` as the message loop."""
    script = e2e_script()
    script[AgentName.ORCHESTRATOR] = orchestrator_responder(
        kind="research_question", plan=PLAN, message=message
    )
    script[AgentName.RESEARCHER] = _researcher
    return script


def _console() -> tuple[Console, io.StringIO]:
    """A plain console that writes to a buffer."""
    out = io.StringIO()
    return Console(file=out, width=160, color_system=None), out


def _lines(*lines: str | None) -> Iterator[str | None]:
    """The user's lines, then the end of input."""
    yield from lines
    while True:
        yield None


async def _typed(*lines: str | None) -> Any:
    """A line reader that types ``lines``."""
    feed = _lines(*lines)

    async def read() -> str | None:
        """The next scripted line."""
        return next(feed)

    return read


async def test_a_research_question_with_no_run_starts_a_research_run(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path, script=_script(_from_memory))
    with app:
        reply = await app.orchestrator.handle(QUESTION)
        (run,) = await app.stores.tasks.list_runs()
        tasks = await app.stores.tasks.list_tasks(run.id)
        decisions = await app.stores.decisions.list(run.id)
    assert reply.kind == "done", reply.text
    assert FROM_MEMORY not in reply.text and reply.text.startswith(STARTING["research_question"])
    assert "Call mcp.run(transport='stdio')" in reply.text
    planner = next(t for t in tasks if t.output_kind == "plan")
    assert kind_from_objective(planner.objective) == "research_question"
    assert any(t.output_kind == "research_answer" and t.output for t in tasks)
    started = [d for d in decisions if d.kind is DecisionKind.CLASSIFY_REQUEST]
    assert started[0].made_by == "code" and started[0].choice == "start_run"


async def test_with_no_run_a_question_about_a_run_gets_a_short_reply(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path, script=_script(_from_memory))
    with app:
        asked = await app.orchestrator.handle("Which pages did the researcher open?")
        accepted = await app.orchestrator.handle("accept")
        runs = await app.stores.tasks.list_runs()
    assert runs == []
    for reply in (asked, accepted):
        assert reply.kind == "answer" and reply.text == NO_RUN_REPLY


async def _finished(app: App) -> str:
    """Ask the question and return the finished research run's id."""
    reply = await app.orchestrator.handle(QUESTION)
    assert reply.kind == "done" and reply.run_id, reply.text
    return reply.run_id


async def test_the_review_shows_the_answers_sources(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path, script=_script(_from_memory))
    console, out = _console()
    with app:
        session = Session(app, console, plain=True, read_line=await _typed("accept"))
        assert await session.start(QUESTION) == 0
    text = out.getvalue()
    assert "Sources:\n  - Project notes: notes.md\n" in text
    assert text.index("Sources:") < text.index("Audit report:")


async def test_resuming_a_done_run_reruns_nothing(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path, script=_script(_from_memory))
    with app:
        run_id = await _finished(app)
    before = read_events(app.runs_dir, run_id)
    # A new process: a fresh app over the same database and runs folder.
    again = await e2e_app(tmp_path, script=_script(_from_memory))
    console, out = _console()
    calls: list[str] = []

    async def count(event: Any) -> None:
        """Note every model call."""
        calls.append(event.run_id)

    again.bus.on_after(EventName.LLM_CALL_AFTER, count)
    with again:
        session = Session(again, console, plain=True, read_line=await _typed(None))
        assert await session.resume(run_id) == 0
        run = await again.stores.tasks.get_run(run_id)
    after = read_events(again.runs_dir, run_id)
    assert calls == []
    assert [e.name for e in after[len(before) :]] == []
    assert run.status is RunStatus.DONE
    assert "Sources:\n  - Project notes: notes.md" in out.getvalue()
    assert f"[run {run_id}: done]" in out.getvalue()


async def test_the_answer_task_gets_the_users_feedback_on_its_briefs(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path, script=_script(_from_memory))
    with app:
        run_id = await _finished(app)
        run = await app.stores.tasks.get_run(run_id)
        tasks = await app.stores.tasks.list_tasks(run_id)
        brief = next(t for t in tasks if t.output_kind == "research_brief")
        answer = next(t for t in tasks if t.output_kind == "research_answer")
        guidance = "Answer for mcp 1.x, not 2.x."
        await app.stores.feedback.add(
            run_id,
            Feedback(
                artifact=None,
                task_id=brief.id,
                source=FeedbackSource.USER_ANSWER,
                text=guidance,
            ),
        )
        await app.stores.feedback.add(
            run_id,
            Feedback(
                artifact=brief.output,
                task_id=brief.id,
                source=FeedbackSource.AUDITOR,
                text="an auditor note",
            ),
        )
        envelope = await app.dispatcher.build_envelope(answer, run)
    assert [fb.text for fb in envelope.input_feedback] == [guidance]
    assert envelope.input_feedback[0].artifact == brief.output
    shown = render_envelope(envelope, {})
    assert "## The user's guidance on your inputs" in shown
    assert f"[user_answer on {brief.output}] {guidance}" in shown
    assert answer.status is TaskStatus.DONE
