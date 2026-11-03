"""The message loop: the orchestrator works out what a message means from the run's
state, and code carries out only the actions it validated."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from agents.fakes import tool_call
from agents.scenario import (
    PRD,
    Responder,
    _last_is_tool_result,
    orchestrator_responder,
    team_script,
)

from agent_team.core.errors import BudgetExceeded
from agent_team.core.models import (
    AgentName,
    DecisionKind,
    EventName,
    ReviewMode,
    RunStatus,
    Task,
    TaskStatus,
)
from agent_team.llm.types import ContentBlock, LLMRequest, ToolResultBlock, ToolUseBlock
from agent_team.orchestrator.messages import scrub
from orchestrator.harness import Team, make_team

QUESTIONS = [
    {"id": "Q-1", "text": "Which database?", "why": "storage"},
    {"id": "Q-2", "text": "Which auth?", "why": "security"},
]


def _prompt(request: LLMRequest) -> str:
    """The message loop's first message."""
    return getattr(request.messages[0].content[0], "text", "")


def _item(request: LLMRequest, question: str) -> str:
    """The task_id the prompt lists for ``question``."""
    found = re.search(rf"task_id=(\S+), asked by [^:]+: {re.escape(question)}", _prompt(request))
    assert found, _prompt(request)
    return found.group(1)


def _team(
    tmp_path: Path, message: Responder, questions: list[dict[str, str]] | None = None
) -> Team:
    """A build whose PRD asks ``questions`` once; ``message`` scripts the message loop
    once a run exists (the first message still starts the run)."""
    script = team_script()
    first = orchestrator_responder()

    def orchestrator(request: LLMRequest) -> Any:
        """Starting the run uses the default script; later messages use ``message``."""
        if request.system.startswith("You are the orchestrator") and "No run" not in _prompt(
            request
        ):
            return message(request)
        return first(request)

    script[AgentName.ORCHESTRATOR] = orchestrator
    if questions:
        calls: list[int] = []

        def product(request: LLMRequest) -> str:
            """Ask on the first attempt, then write a plain PRD."""
            calls.append(1)
            return json.dumps(dict(PRD, questions=questions) if len(calls) == 1 else PRD)

        script[AgentName.PRODUCT] = product
    mode = ReviewMode.QUESTIONS_ONLY if questions else ReviewMode.AUTO_ASSUMPTIONS
    return make_team(tmp_path, script=script, review_mode=mode)


def _then_reply(text: str, first: list[ContentBlock]) -> Responder:
    """Make ``first`` the loop's opening turn, then reply ``text``."""

    def respond(request: LLMRequest) -> list[ContentBlock] | str:
        """Act once, then reply."""
        return json.dumps({"reply": text}) if _last_is_tool_result(request) else first

    return respond


async def _users(team: Team, run_id: str) -> list[Task]:
    """The run's user tasks, oldest first."""
    return [t for t in await team.tasks.list_tasks(run_id) if t.agent is AgentName.USER]


async def _kind(team: Team, run_id: str, kind: str) -> list[Task]:
    """The run's tasks producing ``kind``, oldest first."""
    return [t for t in await team.tasks.list_tasks(run_id) if t.output_kind == kind]


async def test_the_right_item_is_answered_when_two_are_pending(tmp_path: Path) -> None:
    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """The user named an auth scheme, so it answers the auth question."""
        if _last_is_tool_result(request):
            return json.dumps({"reply": "Thanks, OAuth it is."})
        return tool_call(
            "answer_pending", {"task_id": _item(request, "Which auth?"), "text": "OAuth"}
        )

    team = _team(tmp_path, message, QUESTIONS)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "waiting" and reply.run_id and len(reply.pending) == 2
    reply = await team.orch.handle("OAuth, please", run_id=reply.run_id)
    database, auth = await _users(team, reply.run_id or "")
    assert auth.status is TaskStatus.DONE and database.status is TaskStatus.WAITING
    assert [p.task_id for p in reply.pending] == [database.id]
    assert any(f.text == "OAuth" for _, f in team.feedback.items)
    logged = [d for d in team.decisions.items if d.choice == f"answer_pending {auth.id}"]
    assert (
        logged and logged[0].made_by == "model" and logged[0].kind is DecisionKind.CLASSIFY_REQUEST
    )


async def test_run_question_leaves_the_item_pending(tmp_path: Path) -> None:
    team = _team(
        tmp_path,
        _then_reply("The requirements were redone once.", tool_call("list_tasks", {})),
        QUESTIONS[:1],
    )
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    run_id = reply.run_id
    reply = await team.orch.handle("Why was it redone?", run_id=run_id)
    assert reply.kind == "answer" and "redone once" in reply.text
    (question,) = await _users(team, run_id)
    assert question.status is TaskStatus.WAITING
    assert [p.task_id for p in reply.pending] == [question.id]


async def test_an_answer_and_a_change_in_one_message(tmp_path: Path) -> None:
    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """Postgres answers the database question; /health is a change."""
        if _last_is_tool_result(request):
            return json.dumps({"reply": "Postgres noted. Adding /health needs your OK."})
        return [
            ToolUseBlock(
                id="a",
                name="answer_pending",
                arguments={"task_id": _item(request, "Which database?"), "text": "Postgres"},
            ),
            ToolUseBlock(
                id="b",
                name="request_change",
                arguments={"text": "add a /health endpoint", "target": "prd"},
            ),
        ]

    team = _team(tmp_path, message, QUESTIONS[:1])
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    run_id = reply.run_id
    reply = await team.orch.handle("Postgres, and add /health", run_id=run_id)
    assert reply.kind == "waiting", reply.text
    question, confirm = await _users(team, run_id)
    assert question.status is TaskStatus.DONE
    assert confirm.status is TaskStatus.WAITING and "add a /health endpoint" in confirm.objective
    assert "Go ahead? (yes or no)" in reply.text
    # Nothing was reopened for the change yet.
    assert all(t.status is not TaskStatus.SUPERSEDED for t in await _kind(team, run_id, "prd"))


async def _proposed(tmp_path: Path) -> tuple[Team, str]:
    """A finished build with a pagination change proposed and waiting on a yes or no."""
    change = tool_call("request_change", {"text": "paginate the book list", "target": "prd"})
    proposing = {"done": False}
    default = orchestrator_responder()

    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """Propose the change once; afterwards answer like the default script."""
        if not proposing["done"]:
            if _last_is_tool_result(request):
                proposing["done"] = True
                return json.dumps({"reply": "I can add pagination."})
            return change
        return default(request)

    team = _team(tmp_path, message)
    reply = await team.orch.handle("Build a books API")
    assert reply.kind == "done" and reply.run_id
    reply = await team.orch.handle("make it paginate", run_id=reply.run_id)
    assert reply.kind == "waiting", reply.text
    (confirm,) = await _users(team, reply.run_id or "")
    assert (
        "redoing the requirements, then re-running the design, the code, the QA check"
        in confirm.objective
    )
    return team, reply.run_id or ""


async def test_confirming_a_change_with_yes_reopens(tmp_path: Path) -> None:
    team, run_id = await _proposed(tmp_path)
    reply = await team.orch.handle("yes", run_id=run_id)
    assert reply.kind == "done", reply.text
    prds = await _kind(team, run_id, "prd")
    assert [t.status for t in prds] == [TaskStatus.SUPERSEDED, TaskStatus.DONE]
    assert any(f.text == "paginate the book list" for _, f in team.feedback.items)


async def test_declining_a_change_reopens_nothing(tmp_path: Path) -> None:
    team, run_id = await _proposed(tmp_path)
    audits = team.auditor.runs
    reply = await team.orch.handle("no", run_id=run_id)
    assert reply.kind == "done", reply.text
    assert len(await _kind(team, run_id, "prd")) == 1
    assert (await team.tasks.get_run(run_id)).status is RunStatus.DONE
    assert team.auditor.runs == audits  # the finished run isn't audited again


async def test_a_task_id_that_is_not_pending_is_rejected(tmp_path: Path) -> None:
    errors: list[str] = []

    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """Try a made-up id, then ask instead."""
        if _last_is_tool_result(request):
            result = request.messages[-1].content[0]
            assert isinstance(result, ToolResultBlock)
            errors.append(result.content if result.is_error else "")
            return json.dumps({"reply": "Which question is that for?"})
        return tool_call("answer_pending", {"task_id": "task_0000000000000000", "text": "yes"})

    team = _team(tmp_path, message, QUESTIONS)
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.handle("yes", run_id=reply.run_id)
    assert errors and "is not waiting on the user" in errors[0]
    assert all(t.status is TaskStatus.WAITING for t in await _users(team, reply.run_id or ""))
    assert "Which question" in reply.text


async def test_a_change_confirmation_takes_yes_or_no(tmp_path: Path) -> None:
    team, run_id = await _proposed(tmp_path)
    (confirm,) = await _users(team, run_id)
    errors: list[str] = []

    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """Answer the confirmation with something that is neither yes nor no."""
        if _last_is_tool_result(request):
            result = request.messages[-1].content[0]
            errors.append(getattr(result, "content", ""))
            return json.dumps({"reply": "Should I go ahead, yes or no?"})
        return tool_call("answer_pending", {"task_id": confirm.id, "text": "maybe later"})

    team.llm.script[AgentName.ORCHESTRATOR] = orchestrator_responder(message=message)
    await team.orch.handle("maybe later", run_id=run_id)
    assert errors and 'pass "yes" or "no"' in errors[0]
    assert (await team.tasks.get_task(confirm.id)).status is TaskStatus.WAITING


async def test_fallback_acts_on_nothing(tmp_path: Path) -> None:
    team = _team(tmp_path, lambda request: "not json at all", QUESTIONS[:1])
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.handle("Postgres", run_id=reply.run_id)
    assert reply.kind == "waiting"
    assert reply.text.startswith("I didn't catch that.") and "- Which database?" in reply.text
    (question,) = await _users(team, reply.run_id or "")
    assert question.status is TaskStatus.WAITING


async def test_the_reply_shows_no_internal_ids(tmp_path: Path) -> None:
    def message(request: LLMRequest) -> list[ContentBlock] | str:
        """A careless reply that names ids and artifact refs."""
        task_id = _item(request, "Which database?")
        if _last_is_tool_result(request):
            return json.dumps({"reply": f"Answered {task_id} (see prd@v1, run_0123456789abcdef)."})
        return tool_call("answer_pending", {"task_id": task_id, "text": "Postgres"})

    team = _team(tmp_path, message, QUESTIONS[:1])
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.handle("Postgres", run_id=reply.run_id)
    assert not re.search(r"\b(task|run)_[0-9a-f]{8,}", reply.text), reply.text
    assert "@v" not in reply.text and "the requirements" in reply.text


async def test_a_change_to_work_an_answer_just_reopened_is_folded_in(tmp_path: Path) -> None:
    team, run_id = await _proposed(tmp_path)
    (confirm,) = await _users(team, run_id)
    team.llm.script[AgentName.ORCHESTRATOR] = orchestrator_responder(
        message=_then_reply(
            "Going ahead, with sorting too.",
            [
                ToolUseBlock(
                    id="a", name="answer_pending", arguments={"task_id": confirm.id, "text": "yes"}
                ),
                ToolUseBlock(
                    id="b",
                    name="request_change",
                    arguments={"text": "sort by title", "target": "prd"},
                ),
            ],
        )
    )
    reply = await team.orch.handle("yes, and sort it by title", run_id=run_id)
    assert reply.kind == "done", reply.text
    assert "added to the redo of the requirements" in reply.text
    prds = await _kind(team, run_id, "prd")
    assert [t.status for t in prds] == [TaskStatus.SUPERSEDED, TaskStatus.DONE]
    texts = {f.text for _, f in team.feedback.items}
    assert {"paginate the book list", "sort by title"} <= texts
    assert len(await _users(team, run_id)) == 1  # no second confirmation


async def test_stopping_one_run_and_starting_another_drives_the_new_one(tmp_path: Path) -> None:
    team = _team(
        tmp_path,
        _then_reply(
            "Stopped that one and started the new one.",
            [
                ToolUseBlock(id="a", name="stop_run", arguments={"reason": "the user moved on"}),
                ToolUseBlock(
                    id="b",
                    name="start_run",
                    arguments={"request": "Build a todo API", "kind": "build"},
                ),
            ],
        ),
        QUESTIONS[:1],
    )
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    old = reply.run_id
    reply = await team.orch.handle("Forget that, build a todo API instead", run_id=old)
    assert reply.run_id and reply.run_id != old
    assert (await team.tasks.get_run(old)).status is RunStatus.STOPPED
    assert reply.kind == "done", reply.text
    assert reply.text.startswith("Stopped that one")


async def test_running_out_of_budget_while_reading_pauses_and_acts_on_nothing(
    tmp_path: Path,
) -> None:
    def message(request: LLMRequest) -> str:
        """The gateway refuses the call: the run is over budget."""
        raise BudgetExceeded("token cap reached", limit="tokens")

    team = _team(tmp_path, message, QUESTIONS[:1])
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.handle("Postgres", run_id=reply.run_id)
    assert reply.kind == "stopped" and "budget ran out" in reply.text
    assert (await team.tasks.get_run(reply.run_id or "")).status is RunStatus.PAUSED
    (question,) = await _users(team, reply.run_id or "")
    assert question.status is TaskStatus.WAITING


async def test_a_crash_while_reading_is_logged_and_acts_on_nothing(tmp_path: Path) -> None:
    def message(request: LLMRequest) -> str:
        """A bug in the loop."""
        raise RuntimeError("boom")

    team = _team(tmp_path, message, QUESTIONS[:1])
    reply = await team.orch.handle("Build a books API")
    assert reply.run_id
    reply = await team.orch.handle("Postgres", run_id=reply.run_id)
    assert reply.text.startswith("I didn't catch that.")
    errors = [e for e in team.bus.events if e.name is EventName.ERROR]
    assert errors and errors[-1].data["error_type"] == "RuntimeError"
    assert "boom" in str(errors[-1].data["traceback"])


def test_scrub_keeps_layout_and_unknown_refs() -> None:
    text = (
        "Answered (task_0123456789abcdef).\n- the prd@v2 is done\n"
        "- see foo@v1 and task_0123456789abcdef"
    )
    assert scrub(text, "ok") == "Answered.\n- the the requirements is done\n- see foo@v1 and"
    assert "prd@v2" in scrub(text, "what does the PRD say?")
    assert "prd@v2" not in scrub(text, "is it prdx?")


@pytest.mark.parametrize(
    ("typed", "reopened"),
    [("Yes, go ahead!", True), ("sure, do it", True), ("no thanks", False), ("don't", False)],
)
async def test_a_typed_confirmation_is_read_leniently(
    tmp_path: Path, typed: str, reopened: bool
) -> None:
    team, run_id = await _proposed(tmp_path)
    await team.orch.answer(run_id, typed)
    assert len(await _kind(team, run_id, "prd")) == (2 if reopened else 1)


async def test_a_change_while_a_redo_is_under_way_joins_it(tmp_path: Path) -> None:
    team, run_id = await _proposed(tmp_path)
    (confirm,) = await _users(team, run_id)
    # Say yes without driving the run, so the requirements stay reopened and unrun.
    await team.orch._record_answer(await team.tasks.get_run(run_id), confirm, "yes", [])
    team.llm.script[AgentName.ORCHESTRATOR] = orchestrator_responder(
        message=_then_reply(
            "I'll add sorting to it.",
            tool_call("request_change", {"text": "sort by title", "target": "prd"}),
        )
    )
    reply = await team.orch.handle("also sort it by title", run_id=run_id)
    assert "added to the redo of the requirements" in reply.text
    assert {"paginate the book list", "sort by title"} <= {f.text for _, f in team.feedback.items}
    assert len(await _users(team, run_id)) == 1
