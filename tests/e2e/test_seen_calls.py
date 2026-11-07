"""A change request whose QA check stops on a dangerous action, through the real team
with only the model scripted: the user's guidance retries QA, and the next check
judges the retry on its own calls instead of stopping again on the one the user saw.
A new dangerous call still stops the run, and resuming a run whose check stopped on
calls the user had already seen checks it again."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from agents.scenario import QA

from agent_team.app import App
from agent_team.core.models import SEEN_CALLS, AgentName, Task, TaskStatus
from agent_team.llm.types import LLMRequest, ToolResultBlock, ToolUseBlock
from e2e.team import PASS, e2e_app
from e2e.test_change_request import CHANGE, script

DANGER = "rm -rf .git"
GUIDANCE = "Retry the QA check, and this time leave the git history alone."


def _envelope(request: LLMRequest) -> str:
    """The first message's text: the task envelope."""
    return getattr(request.messages[0].content[0], "text", "")


def _results(request: LLMRequest) -> int:
    """How many tool results the conversation holds so far."""
    return sum(isinstance(b, ToolResultBlock) for m in request.messages for b in m.content)


def change_qa(dangerous: set[int]) -> Any:
    """QA for the change request: each attempt runs one command, the dangerous one on
    the attempts in ``dangerous`` (counted from 1), then reports."""
    attempts: list[int] = []

    def qa(request: LLMRequest) -> Any:
        """One step of QA's conversation."""
        if "AC-2.1" not in _envelope(request):
            return json.dumps(QA)  # the first build's QA
        if _results(request) == 0:
            attempts.append(1)
            danger = len(attempts) in dangerous
            argv = ["sh", "-c", DANGER] if danger else ["python", "-m", "pytest", "-q"]
            return [ToolUseBlock(id="call_0", name="run_command", arguments={"argv": argv})]
        more = {"criterion": "AC-2.1", "result": "pass", "evidence": "recent.py:1"}
        return json.dumps(dict(QA, results=[*QA["results"], more]))

    return qa


def judge(prompt: str) -> Any:
    """An auditor model that reports S5 on every dangerous call it sees that isn't
    marked as seen by the user."""
    if not prompt.startswith("# Sweep: task"):
        return PASS
    flagged = [line for line in prompt.splitlines() if DANGER in line and "[seen]" not in line]
    if not flagged:
        return PASS
    finding = {
        "rule_id": "S5",
        "severity": "blocker",
        "item": "run_command",
        "message": "QA deleted the workspace's git history",
    }
    return {"summary": "a dangerous action", "findings": [finding]}


async def _team(tmp_path: Path, dangerous: set[int]) -> App:
    """The e2e team whose change-request QA runs the dangerous command on ``dangerous``."""
    team = script()
    team[AgentName.QA] = change_qa(dangerous)
    return await e2e_app(tmp_path, script=team, judge=judge)


async def _stopped_change(app: App) -> str:
    """Build, then ask for the change; QA's check stops the run. Returns its id."""
    reply = await app.orchestrator.handle("Build a notes server")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    reply = await app.orchestrator.request_change(reply.run_id, CHANGE, target="prd")
    assert reply.kind == "stopped", reply.text
    assert "deleted the workspace's git history" in await _asked(app, reply.run_id)
    return reply.run_id


async def _asked(app: App, run_id: str) -> str:
    """The one question waiting on the user."""
    [waiting] = [
        t
        for t in await app.stores.tasks.list_tasks(run_id, status=TaskStatus.WAITING)
        if t.agent is AgentName.USER
    ]
    return waiting.objective


async def _change_qa(app: App, run_id: str) -> Task:
    """The change request's QA task, the newest one."""
    tasks = await app.stores.tasks.list_tasks(run_id)
    return [t for t in tasks if t.agent is AgentName.QA and t.status is not TaskStatus.SUPERSEDED][
        -1
    ]


async def test_a_change_request_reaches_done_after_the_user_retries_a_stopped_qa(
    tmp_path: Path,
) -> None:
    app = await _team(tmp_path, dangerous={1})
    run_id = await _stopped_change(app)
    reply = await app.orchestrator.answer(run_id, GUIDANCE)
    assert reply.kind == "done", reply.text
    qa = await _change_qa(app, run_id)
    assert qa.status is TaskStatus.DONE and qa.attempt == 2
    seen = [d for d in await app.stores.decisions.list(run_id) if d.rule == SEEN_CALLS]
    assert len(seen) == 1 and seen[0].task_id == qa.id
    latest = await app.stores.verdicts.latest_for_task(qa.id)
    assert latest is not None and latest.passed


async def test_a_new_dangerous_call_on_the_retry_still_stops_the_run(tmp_path: Path) -> None:
    app = await _team(tmp_path, dangerous={1, 2})
    run_id = await _stopped_change(app)
    reply = await app.orchestrator.answer(run_id, GUIDANCE)
    assert reply.kind == "stopped", reply.text
    assert "unsafe" in await _asked(app, run_id)
    qa = await _change_qa(app, run_id)
    assert qa.attempt == 2 and qa.status is TaskStatus.ESCALATED
    assert (await app.stores.tasks.get_run(run_id)).status.value == "stopped"


async def test_resume_checks_again_a_stop_on_calls_the_user_had_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = await _team(tmp_path, dangerous={1})
    run_id = await _stopped_change(app)

    async def forget(*_: Any) -> bool:
        """Record nothing, as the team did before it kept seen calls."""
        return False

    # Without seen calls, the retry's check stops again on the first attempt's call.
    monkeypatch.setattr(app.orchestrator, "_note_seen_calls", forget)
    reply = await app.orchestrator.answer(run_id, GUIDANCE)
    assert reply.kind == "stopped", reply.text
    assert "unsafe" in await _asked(app, run_id)
    monkeypatch.undo()

    reply = await app.orchestrator.resume(run_id)
    assert reply.kind == "done", reply.text
    qa = await _change_qa(app, run_id)
    assert qa.status is TaskStatus.DONE and qa.attempt == 2  # checked again, not retried
    decisions = await app.stores.decisions.list(run_id)
    assert [d.rule for d in decisions if d.rule == "recheck_seen_calls"] == ["recheck_seen_calls"]
    asked = [
        t
        for t in await app.stores.tasks.list_tasks(run_id)
        if t.parent_task_id == qa.id and t.agent is AgentName.USER
    ]
    assert [t.status for t in asked] == [TaskStatus.DONE, TaskStatus.SUPERSEDED]
