"""run_command and install_deps over a fake runner, and the shape of what they log."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sandbox.fakes import FakeRunner

from agent_team.audit.catalog import load_catalog
from agent_team.audit.evidence import HELPERS, AuditContext
from agent_team.audit.toollog import tool_call_from_event
from agent_team.core.artifacts import CodeChange, Spec
from agent_team.core.artifacts import TestRun as UnitTestRun
from agent_team.core.config import SandboxConfig
from agent_team.core.errors import SandboxError, ToolError
from agent_team.core.interfaces import ToolContext
from agent_team.core.models import (
    AgentName,
    Event,
    EventName,
    Phase,
    RuleScope,
    Run,
    Task,
    ToolResult,
)
from agent_team.sandbox import CommandResult, CommandSpec
from agent_team.tools import command_tools, file_tools
from agents.fakes import FakeBus

CONFIG = SandboxConfig(timeout_s=30, max_timeout_s=60, install_timeout_s=90)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    ws = tmp_path / "run" / "workspace"
    (ws / "src").mkdir(parents=True)
    run = Run(title="t", request="r", workspace=str(ws))
    task = Task(
        run_id=run.id,
        phase=Phase.IMPLEMENTATION,
        agent=AgentName.ENGINEER,
        objective="o",
        output_kind="code_change",
    )
    return ToolContext(run=run, task=task, bus=FakeBus())


def workspace(ctx: ToolContext) -> Path:
    """The run's workspace folder."""
    return Path(ctx.run.workspace)


def failing(spec: CommandSpec) -> CommandResult:
    """A test run with one failure."""
    return CommandResult(exit_code=1, stdout="1 failed, 2 passed", stderr="E assert", duration_s=2)


async def test_run_command_runs_argv_offline_and_reports_the_result(ctx: ToolContext) -> None:
    runner = FakeRunner()
    tool = command_tools(runner, CONFIG)["run_command"]
    out = await tool({"argv": ["pytest", "-q"], "cwd": "src"}, ctx)
    [spec] = runner.specs
    assert spec.argv == ["pytest", "-q"] and not spec.network and spec.cwd == "src"
    assert (spec.run_id, spec.task_id, spec.timeout_s) == (ctx.run.id, ctx.task.id, 30)
    assert spec.workspace == ctx.run.workspace
    assert out.content.startswith("$ pytest -q\nexit 0 in 0.01s\n--- stdout ---\n1 passed")
    assert not out.is_error and out.data is not None
    assert out.data["argv"] == ["pytest", "-q"] and out.data["exit_code"] == 0
    assert ".deps/" in workspace(ctx).joinpath(".gitignore").read_text()


async def test_a_failing_command_is_an_error_result(ctx: ToolContext) -> None:
    tool = command_tools(FakeRunner(answer=failing), CONFIG)["run_command"]
    out = await tool({"argv": ["pytest"]}, ctx)
    assert out.is_error and "exit 1" in out.content and "--- stderr ---\nE assert" in out.content


async def test_a_timeout_is_reported(ctx: ToolContext) -> None:
    def slow(spec: CommandSpec) -> CommandResult:
        """A command killed at its timeout."""
        return CommandResult(exit_code=124, stdout="", stderr="", duration_s=60, timed_out=True)

    runner = FakeRunner(answer=slow)
    out = await command_tools(runner, CONFIG)["run_command"](
        {"argv": ["sleep", "999"], "timeout_s": 999}, ctx
    )
    assert runner.specs[0].timeout_s == 60  # capped at max_timeout_s
    assert out.is_error and "timed out after 60s" in out.content
    assert out.data is not None and out.data["timed_out"] is True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"argv": "pytest -q"}, "list of strings"),
        ({"argv": []}, "non-empty list"),
        ({"argv": ["", "x"]}, "empty program"),
        ({"argv": ["ls"], "cwd": "../.."}, "outside the workspace"),
        ({"argv": ["ls"], "cwd": "missing"}, "not a folder"),
        ({"argv": ["ls"], "cwd": ".git"}, "inside .git"),
        ({"argv": ["ls"], "timeout_s": 0}, "positive number"),
    ],
)
async def test_bad_run_command_args_are_refused(
    ctx: ToolContext, args: dict[str, Any], message: str
) -> None:
    workspace(ctx).joinpath(".git").mkdir()
    runner = FakeRunner()
    with pytest.raises(ToolError, match=message):
        await command_tools(runner, CONFIG)["run_command"](args, ctx)
    assert runner.specs == []


async def test_install_deps_installs_into_the_deps_folder_online(ctx: ToolContext) -> None:
    workspace(ctx).joinpath("requirements.txt").write_text(
        "# api\nfastapi==0.115.0  # pinned\n\nuvicorn[standard]>=0.30,<1\n"
    )
    runner = FakeRunner()
    out = await command_tools(runner, CONFIG)["install_deps"](
        {"packages": ["pytest==8.3.3"], "requirements": "requirements.txt"}, ctx
    )
    [spec] = runner.specs
    assert spec.network and spec.timeout_s == 90 and spec.cwd == "/tmp"
    assert spec.argv[:5] == ["python", "-I", "-m", "pip", "install"]  # nothing from the workspace
    assert spec.argv[-5:] == [
        "--target",
        "/workspace/.deps",
        "pytest==8.3.3",
        "fastapi==0.115.0",
        "uvicorn[standard]>=0.30,<1",
    ]
    assert workspace(ctx).joinpath(".deps").is_dir()
    assert out.data is not None and out.data["network"] is True
    assert out.data["packages"][0] == "pytest==8.3.3"


@pytest.mark.parametrize(
    "package",
    [
        "--index-url=https://evil.example/simple",
        "-e .",
        "git+https://github.com/x/y.git",
        "https://files.example/x.whl",
        "./local_pkg",
        "requests; python_version>'3'",
        "requests==2.0 --extra-index-url x",
        "payload.tar.gz",
        "evil.whl",
        "pkg.zip[extra]",
    ],
)
async def test_install_deps_refuses_options_urls_and_paths(ctx: ToolContext, package: str) -> None:
    runner = FakeRunner()
    with pytest.raises(ToolError, match="only package names"):
        await command_tools(runner, CONFIG)["install_deps"]({"packages": [package]}, ctx)
    assert runner.specs == []


async def test_install_deps_checks_requirements_file_lines_too(ctx: ToolContext) -> None:
    workspace(ctx).joinpath("req.txt").write_text("flask==3.0.0\n-r other.txt\n")
    tools = command_tools(FakeRunner(), CONFIG)
    with pytest.raises(ToolError, match="only package names"):
        await tools["install_deps"]({"requirements": "req.txt"}, ctx)
    with pytest.raises(ToolError, match="give packages"):
        await tools["install_deps"]({}, ctx)
    with pytest.raises(ToolError, match="outside the workspace"):
        await tools["install_deps"]({"requirements": "/etc/passwd"}, ctx)


def _logged(result: ToolResult, ctx: ToolContext, args: dict[str, Any]) -> Event:
    """The tool_call_after event the agent loop emits for a finished call."""
    return Event(
        name=EventName.TOOL_CALL_AFTER,
        run_id=ctx.run.id,
        task_id=ctx.task.id,
        agent=AgentName.ENGINEER,
        data={
            "tool": "run_command",
            "call_id": "c1",
            "args": args,
            "result": result.content,
            "is_error": result.is_error,
            "blocked": False,
            "data": result.data,
        },
    )


async def test_r6_reads_the_logged_run_command(ctx: ToolContext) -> None:
    """The exit code a real run_command logs reaches R6 through the tool log."""
    args = {"argv": ["pytest", "-q"]}
    result = await command_tools(FakeRunner(answer=failing), CONFIG)["run_command"](args, ctx)
    call = tool_call_from_event(_logged(result, ctx, args))
    assert call is not None and call.data["exit_code"] == 1
    claimed = UnitTestRun(command="pytest -q", passed=3, failed=0, exit_code=0, output_tail="")
    change = CodeChange(summary="s", files_written=[], tests=claimed)
    audit = AuditContext(
        run=ctx.run,
        scope=RuleScope.TASK,
        phase=Phase.IMPLEMENTATION,
        outputs=[(ctx.task, change)],
        latest={"spec": Spec(summary="s", components=[], checks=[])},
        tool_calls=[call],
    )
    [rule] = [r for r in load_catalog() if r.id == "R6"]
    [finding] = HELPERS["tests_ran"](audit, rule).findings
    assert "exited 1, but the change reports exit 0" in finding.message


async def test_the_deps_folder_stays_out_of_listings(ctx: ToolContext) -> None:
    await command_tools(FakeRunner(), CONFIG)["install_deps"]({"packages": ["httpx"]}, ctx)
    workspace(ctx).joinpath(".deps", "httpx.py").write_text("vendored = True\n")
    tools = file_tools()
    listing = await tools["list_dir"]({}, ctx)
    assert ".deps" not in listing.content
    found = await tools["grep"]({"pattern": "vendored"}, ctx)
    assert found.content == "no matches"


async def test_the_repository_exists_before_the_first_command(ctx: ToolContext) -> None:
    """The container must never see a writable .git it could plant config in."""
    seen: list[bool] = []

    def check(spec: CommandSpec) -> CommandResult:
        """Note whether .git was there when the command started."""
        seen.append(workspace(ctx).joinpath(".git", "HEAD").is_file())
        return CommandResult(exit_code=0, stdout="", stderr="", duration_s=0)

    await command_tools(FakeRunner(answer=check), CONFIG)["run_command"]({"argv": ["ls"]}, ctx)
    assert seen == [True]


async def test_a_planted_git_file_stops_the_command(ctx: ToolContext) -> None:
    workspace(ctx).joinpath(".git").write_text("gitdir: /somewhere/else\n")
    runner = FakeRunner()
    with pytest.raises(ToolError, match="not a folder the harness created"):
        await command_tools(runner, CONFIG)["run_command"]({"argv": ["ls"]}, ctx)
    assert runner.specs == []


async def test_a_sandbox_failure_is_an_error_the_model_reads(ctx: ToolContext) -> None:
    def broken(spec: CommandSpec) -> CommandResult:
        """Docker fell over."""
        raise SandboxError("can't create the sandbox container: no space left")

    tools = command_tools(FakeRunner(answer=broken), CONFIG)
    with pytest.raises(ToolError, match=r"the sandbox failed: .*no space left"):
        await tools["run_command"]({"argv": ["ls"]}, ctx)
    with pytest.raises(ToolError, match="the sandbox failed"):
        await tools["install_deps"]({"packages": ["httpx"]}, ctx)
