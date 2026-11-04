"""A build request through the real team on SQLite, with only the model scripted: the
composition root, the gateway with fault injection, the rule catalog, the auditor and
its pre-check, the verdict policy and the JSONL trace."""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import Any

from agents.scenario import Responder
from sandbox.fakes import FakeRunner

from agent_team.app import App
from agent_team.core.artifacts import ArtifactRef
from agent_team.core.models import (
    AgentName,
    DecisionKind,
    EventName,
    Phase,
    RunStatus,
    TaskStatus,
)
from agent_team.core.tracing import SpanNode, load_spans, read_events, render_tree, span_tree
from agent_team.llm.types import LLMRequest
from agent_team.orchestrator.replies import Reply
from agent_team.sandbox import WorkspaceGit
from e2e.team import PASS, e2e_app, e2e_script

SPEC_GAP = {
    "summary": "the spec misses a requirement",
    "findings": [
        {
            "rule_id": "R7",
            "severity": "blocker",
            "artifact": "spec@v1",
            "item": "api",
            "message": "the spec's only component leaves REQ-1 without a route",
        }
    ],
}


def _walk(node: SpanNode) -> list[SpanNode]:
    """The node and everything under it."""
    return [node, *(n for child in node.children for n in _walk(child))]


async def test_a_build_goes_through_every_phase_to_done(tmp_path: Path) -> None:
    sweeps: list[tuple[str | None, str | None]] = []

    def auditor(agent: str | None, phase: str | None) -> Any:
        """Crash once on the PRD's check, and blame the spec once from QA's."""
        sweeps.append((agent, phase))
        if agent == "product" and sweeps.count(("product", "prd")) == 1:
            raise RuntimeError("auditor model fell over")
        if agent == "qa" and sweeps.count(("qa", "implementation")) == 1:
            return SPEC_GAP
        return PASS

    # The first model call of the run hits a 503 and is retried.
    runner = FakeRunner()
    app = await e2e_app(
        tmp_path, auditor=auditor, runner=runner, chaos={"script": ["server_error"]}
    )
    with app:
        reply = await app.orchestrator.handle("Build a books API")
        await _check_the_build(app, reply)
        await _check_the_sandbox(app, runner, reply)


async def _check_the_sandbox(app: App, runner: FakeRunner, reply: Reply) -> None:
    """The Engineer's commands went through the real tools, each passing version was
    committed, and every task's containers were cleaned up."""
    assert reply.run_id
    run = await app.stores.tasks.get_run(reply.run_id)
    tasks = await app.stores.tasks.list_tasks(run.id)
    engineers = [t for t in tasks if t.agent is AgentName.ENGINEER]
    # Each Engineer attempt installed pytest with the network on, then ran it without.
    installs = [s for s in runner.specs if s.network]
    tests = [s for s in runner.specs if not s.network]
    assert len(installs) == len(tests) == len(engineers) == 2
    assert all(s.argv[-1] == "pytest==8.3.3" for s in installs)
    assert all(s.argv == ["pytest", "-q"] and s.workspace == run.workspace for s in tests)
    assert {s.task_id for s in runner.specs} == {t.id for t in engineers}
    cleaned = {c["task_id"] for c in runner.cleaned}
    assert {t.id for t in engineers} <= cleaned
    assert {"run_id": run.id, "task_id": None} in runner.cleaned

    # Both code versions passed their checks and were committed and tagged.
    def git(*args: str) -> list[str]:
        """Run git in the workspace and return its output lines."""
        out = subprocess.run(["git", *args], cwd=run.workspace, capture_output=True, text=True)
        return out.stdout.splitlines()

    assert git("log", "--format=%s") == [
        "code_change v2: " + engineers[1].objective[:72],
        "code_change v1: " + engineers[0].objective[:72],
    ]
    assert {"app.py", ".gitignore"} <= set(
        git("show", "--name-only", "--format=", "code_change-v1")
    )
    assert ".deps" not in "".join(git("ls-files"))
    assert (
        await WorkspaceGit(run.workspace).diff(
            ArtifactRef(kind="code_change", version=1), ArtifactRef(kind="code_change", version=2)
        )
        == ""
    )
    ignored = set(git("show", "code_change-v1:.gitignore"))
    assert {".deps/", "__pycache__/", ".pytest_cache/"} <= ignored


async def _check_the_build(app: App, reply: Reply) -> None:
    """Everything the build test expects of the finished run."""
    assert reply.kind == "done", reply.text
    assert reply.run_id
    run_id = reply.run_id
    stores = app.stores
    run = await stores.tasks.get_run(run_id)
    tasks = await stores.tasks.list_tasks(run_id)
    work = [t for t in tasks if t.agent not in (AgentName.AUDITOR, AgentName.USER)]
    live = [t for t in work if t.status is not TaskStatus.SUPERSEDED]
    assert run.status is RunStatus.DONE
    assert {t.phase for t in live} == set(Phase)
    assert all(t.status is TaskStatus.DONE for t in live)

    # The injected fault was retried by the gateway and the run carried on.
    errors = [e for e in read_events(app.runs_dir, "session") if e.name is EventName.ERROR]
    assert [(e.data["status"], e.data["retry"]) for e in errors] == [(503, True)]

    # The auditor crash was retried as a check; the PRD kept its first attempt.
    prd = next(t for t in tasks if t.output_kind == "prd")
    [prd_check] = [t for t in tasks if t.parent_task_id == prd.id]
    assert prd_check.agent is AgentName.AUDITOR
    assert prd.attempt == 1 and prd_check.attempt == 2 and prd_check.status is TaskStatus.DONE
    decisions = await stores.decisions.list(run_id)
    assert not any(d.kind is DecisionKind.RETRY and d.task_id == prd.id for d in decisions)

    # QA's check blamed the spec: the spec was reopened and everything after it rebuilt.
    assert any(d.kind is DecisionKind.UPSTREAM_RERUN for d in decisions)
    for kind in ("spec", "code_change", "qa_report"):
        found = [t.status for t in tasks if t.output_kind == kind]
        assert found == [TaskStatus.SUPERSEDED, TaskStatus.DONE], kind
    assert run.upstream_reruns == 1

    # The whole run is one trace: a single tree under the run's span, all closed.
    spans = load_spans(app.runs_dir, run_id)
    [root] = span_tree(spans)
    assert root.span.kind == "run" and root.span.span_id == run.span_id
    assert {s.trace_id for s in spans} == {run.trace_id}
    assert all(s.status != "open" for s in spans), render_tree([root])
    assert {"phase", "task", "attempt", "audit", "tool", "llm"} <= {s.kind for s in spans}
    under = {n.span.span_id for n in _walk(root)}
    assert under == {s.span_id for s in spans}


class Crash(BaseException):
    """Stands in for the process dying: nothing in the team catches it."""


async def test_a_crash_leaves_the_trace_open_and_resume_closes_it(tmp_path: Path) -> None:
    script = e2e_script()
    engineer: Responder = script[AgentName.ENGINEER]
    calls: list[int] = []

    def crashing(request: LLMRequest) -> Any:
        """Die on the Engineer's second model call, after it wrote the module."""
        calls.append(1)
        if len(calls) == 2:
            raise Crash
        return engineer(request)

    script[AgentName.ENGINEER] = crashing
    app = await e2e_app(tmp_path, script=script)
    with app:
        with contextlib.suppress(Crash):
            await app.orchestrator.handle("Build a books API")
        [run] = await app.stores.tasks.list_runs()
    [root] = span_tree(load_spans(app.runs_dir, run.id))
    assert root.span.status == "open"  # the run never finished

    restarted = await e2e_app(tmp_path, script=script)
    with restarted:
        reply = await restarted.orchestrator.resume(run.id)
        assert reply.kind == "done", reply.text
        tasks = await restarted.stores.tasks.list_tasks(run.id)
    engineer_task = next(t for t in tasks if t.agent is AgentName.ENGINEER)
    assert engineer_task.attempt == 1  # continued from its checkpoint
    spans = load_spans(restarted.runs_dir, run.id)
    [root] = span_tree(spans)  # both processes wrote into the same tree
    assert root.span.status == "ok"
    assert {s.trace_id for s in spans} == {run.trace_id}
