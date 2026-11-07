"""A change request that adds a feature, through the real team with only the model
scripted: the Engineer's envelope names the new requirement and the change, so the
first change-request attempt builds it, and the per-version commits leave out the
database the code writes when it runs."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from agents.scenario import PRD, QA, SPEC

from agent_team.core.artifacts import ArtifactRef, CodeChange
from agent_team.core.models import AgentName, DecisionKind, TaskOrigin
from agent_team.llm.types import LLMRequest, ToolResultBlock, ToolUseBlock
from agent_team.sandbox import WorkspaceGit
from e2e.team import TESTS, e2e_app, e2e_script

CHANGE = "Add a tool that lists notes modified in the last N days"
RECENT_REQ = {
    "id": "REQ-2",
    "text": "List notes modified in the last N days",
    "asks": ["ASK-1"],
    "acceptance_criteria": [{"id": "AC-2.1", "text": "list_recent_notes(7) returns them"}],
}
RECENT = {
    "name": "recent",
    "responsibility": "the list_recent_notes tool",
    "requirements": ["REQ-2"],
    "files": ["recent.py"],
}


def _envelope(request: LLMRequest) -> str:
    """The first message's text: the task envelope."""
    return getattr(request.messages[0].content[0], "text", "")


def _results(request: LLMRequest) -> int:
    """How many tool results the conversation holds so far."""
    return sum(isinstance(b, ToolResultBlock) for m in request.messages for b in m.content)


def engineer(request: LLMRequest) -> Any:
    """An Engineer that builds what its envelope names: app.py (which writes a
    database when it runs), plus recent.py when the envelope names REQ-2 as new."""
    files = {"app.py": "import sqlite3\nsqlite3.connect('library.db')\n"}
    if "new: REQ-2" in _envelope(request):
        files["recent.py"] = "def list_recent_notes(days):\n    return []\n"
    steps: list[tuple[str, dict[str, Any]]] = [
        ("write_file", {"path": path, "content": text}) for path, text in files.items()
    ]
    steps += [
        ("install_deps", {"packages": ["pytest==8.3.3"]}),
        ("run_command", {"argv": ["pytest", "-q"]}),
    ]
    done = _results(request)
    if done < len(steps):
        name, args = steps[done]
        return [ToolUseBlock(id=f"call_{done}", name=name, arguments=args)]
    return json.dumps({"summary": "Built it.", "files_written": list(files), "tests": TESTS})


def script() -> dict[AgentName, Any]:
    """The e2e team, whose PRD, spec and QA take up REQ-2 once the change asks for it."""
    team = e2e_script()

    def product(request: LLMRequest) -> str:
        """The PRD, with REQ-2 once the user asked for it."""
        if CHANGE in _envelope(request):
            return json.dumps(dict(PRD, requirements=[*PRD["requirements"], RECENT_REQ]))
        return json.dumps(PRD)

    def architect(request: LLMRequest) -> str:
        """The spec, with the recent component once the PRD has REQ-2."""
        if "REQ-2" in _envelope(request):
            check = {"id": "CHK-2", "criteria": ["AC-2.1"], "kind": "unit_test", "description": "t"}
            spec = dict(SPEC, components=[*SPEC["components"], RECENT])
            return json.dumps(dict(spec, checks=[*SPEC["checks"], check]))
        return json.dumps(SPEC)

    def qa(request: LLMRequest) -> str:
        """QA's report, with the new criterion once the PRD has it."""
        if "AC-2.1" in _envelope(request):
            more = {"criterion": "AC-2.1", "result": "pass", "evidence": "recent.py:1"}
            return json.dumps(dict(QA, results=[*QA["results"], more]))
        return json.dumps(QA)

    team.update({AgentName.PRODUCT: product, AgentName.ARCHITECT: architect, AgentName.QA: qa})
    team[AgentName.ENGINEER] = engineer
    return team


async def test_a_change_request_is_built_on_the_engineers_first_attempt(tmp_path: Path) -> None:
    app = await e2e_app(tmp_path, script=script())
    orch = app.orchestrator
    reply = await orch.handle("Build a notes server")
    assert reply.kind == "done", reply.text
    assert reply.run_id
    run = await app.stores.tasks.get_run(reply.run_id)
    # The database app.py writes when it runs sits in the workspace.
    await asyncio.to_thread(Path(run.workspace, "library.db").write_bytes, b"SQLite format 3\x00")
    before = len(await app.stores.tasks.list_tasks(run.id))

    reply = await orch.request_change(run.id, CHANGE, target="prd")
    assert reply.kind == "done", reply.text
    tasks = await app.stores.tasks.list_tasks(run.id)
    after = tasks[before:]
    [change] = [t for t in after if t.agent is AgentName.ENGINEER]
    assert change.attempt == 1 and CHANGE in change.objective
    assert change.output is not None
    code = await app.stores.artifacts.load(run.id, change.output)
    assert isinstance(code, CodeChange) and "recent.py" in code.files_written
    decisions = await app.stores.decisions.list(run.id)
    assert not [
        d
        for d in decisions
        if d.kind in (DecisionKind.FIX, DecisionKind.RETRY, DecisionKind.UPSTREAM_RERUN)
    ]
    assert not [t for t in after if t.origin is TaskOrigin.FIX]

    # The version commits and their diff leave the database out.
    listed = await asyncio.to_thread(
        subprocess.run,
        ["git", "ls-files"],
        cwd=run.workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    files = listed.stdout.split()
    assert "recent.py" in files and "library.db" not in files
    diff = await WorkspaceGit(Path(run.workspace)).diff(
        ArtifactRef(kind="code_change", version=1), change.output
    )
    assert "recent.py" in diff and "library.db" not in diff
