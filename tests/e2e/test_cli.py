"""The command line over the real team with a scripted model: a run answered from
stdin, a run left waiting and resumed, and reading the logs back."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner, Result

from agent_team import cli
from agent_team.app import App
from agent_team.core.config import Config
from agent_team.core.control import DriveLock, KillSwitch
from agent_team.core.db import Database
from agent_team.core.errors import SandboxError
from agent_team.core.models import Decision, DecisionKind, Run, RunStatus
from agent_team.core.stores import SqliteDecisionLog, SqliteTaskStore
from e2e.team import asking_script, e2e_config, scripted_app


def _asking_team() -> Callable[[Config], Awaitable[App]]:
    """A team whose Product asks one question before writing the PRD."""
    script = asking_script()

    async def build(config: Config) -> App:
        """The scripted app from the CLI's config."""
        return await scripted_app(config, script=script)

    return build


@pytest.fixture
def config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config file for the scripted team, and the CLI pointed at that team."""
    path = tmp_path / "config.yaml"
    config = e2e_config(tmp_path, review="questions_only")
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setattr(cli, "_factory", _asking_team())
    return path


def _plain(result: Result) -> str:
    """The command's output without colour codes, which CI may force on."""
    return re.sub(r"\x1b\[[0-9;]*m", "", result.output)


def _run_id(output: str) -> str:
    """The run id the CLI printed."""
    found = re.search(r"\[run (run_\w+): ", output)
    assert found, output
    return found.group(1)


def test_run_answers_questions_from_stdin(config_file: Path) -> None:
    result = CliRunner().invoke(
        cli.app, ["run", "Build a books API", "-c", str(config_file)], input="\n\n20 per page\n"
    )
    assert result.exit_code == 0, result.output
    assert "Paginate the list?" in result.output
    assert result.output.count(": waiting]") == 1  # blank lines don't repeat the question
    assert "'Build a books API' is done." in result.output


def test_a_run_left_waiting_resumes_and_its_logs_read_back(config_file: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli.app, ["run", "Build a books API", "-c", str(config_file)], input="")
    assert first.exit_code == 0, first.output
    run_id = _run_id(first.output)
    assert f"agent-team resume {run_id}" in first.output

    resumed = runner.invoke(cli.app, ["resume", run_id, "-c", str(config_file)], input="20\n")
    assert resumed.exit_code == 0, resumed.output
    assert "is done." in resumed.output

    logs = runner.invoke(cli.app, ["logs", run_id, "-c", str(config_file)])
    assert logs.exit_code == 0, logs.output
    assert "run_started" in logs.output and "run_finished" in logs.output
    engineer = runner.invoke(
        cli.app, ["logs", run_id, "--agent", "engineer", "-c", str(config_file)]
    )
    lines = engineer.output.splitlines()
    assert lines and all(" engineer " in line for line in lines)
    spans = runner.invoke(cli.app, ["logs", run_id, "--spans", "-c", str(config_file)])
    assert spans.output.startswith("run:run [ok]")


def test_logs_of_an_unknown_run_fail(config_file: Path) -> None:
    result = CliRunner().invoke(cli.app, ["logs", "run_nope", "-c", str(config_file)])
    assert result.exit_code == 1
    assert "No events for run run_nope" in result.output


def test_no_docker_prints_the_help_not_a_traceback(
    config_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_docker(config: Config) -> App:
        """The sandbox can't start."""
        raise SandboxError("Docker isn't reachable (test). Start Docker.")

    monkeypatch.setattr(cli, "_factory", no_docker)
    result = CliRunner().invoke(cli.app, ["run", "Build it", "-c", str(config_file)])
    assert result.exit_code == 1
    assert "Can't start: Docker isn't reachable (test). Start Docker." in result.output
    assert "Traceback" not in result.output


def _left_waiting(config_file: Path, *extra: str) -> str:
    """Start a run, leave it waiting on its question and return its id."""
    args = ["run", "Build a books API", "-c", str(config_file), "--plain", *extra]
    result = CliRunner().invoke(cli.app, args, input="")
    assert result.exit_code == 0, result.output
    return _run_id(result.output)


async def _run_and_stops(config_file: Path, run_id: str) -> tuple[Run, list[Decision]]:
    """The run as stored, and its stop decisions."""
    db = Database(config_file.parent / "team.db")
    try:
        run = await SqliteTaskStore(db).get_run(run_id)
        decisions = await SqliteDecisionLog(db).list(run_id)
    finally:
        db.close()
    return run, [d for d in decisions if d.kind is DecisionKind.STOP]


def test_stopping_a_run_nobody_drives_marks_it_stopped(config_file: Path) -> None:
    run_id = _left_waiting(config_file)
    runner = CliRunner()
    stopped = runner.invoke(cli.app, ["stop", run_id, "-c", str(config_file)])
    assert stopped.exit_code == 0, stopped.output
    assert f"Stopped run {run_id}. Continue: agent-team resume {run_id}" in stopped.output
    run, stops = asyncio.run(_run_and_stops(config_file, run_id))
    assert run.status is RunStatus.STOPPED and len(stops) == 1
    assert KillSwitch(config_file.parent / "runs").is_stopped(run_id)

    # Resuming lifts the stop: the run carries on to done.
    resumed = runner.invoke(cli.app, ["resume", run_id, "-c", str(config_file)], input="20\n")
    assert resumed.exit_code == 0, resumed.output
    assert "is done." in resumed.output
    assert not KillSwitch(config_file.parent / "runs").is_stopped(run_id)


def test_stopping_a_driven_run_only_sets_the_flag(config_file: Path) -> None:
    run_id = _left_waiting(config_file)
    runs = config_file.parent / "runs"
    session = DriveLock(runs)
    assert session.acquire(run_id)
    try:
        stopped = CliRunner().invoke(cli.app, ["stop", run_id, "-c", str(config_file)])
        assert stopped.exit_code == 0, stopped.output
        assert f"Asked the session driving {run_id} to stop" in stopped.output
        run, stops = asyncio.run(_run_and_stops(config_file, run_id))
        assert run.status is not RunStatus.STOPPED and stops == []
        assert KillSwitch(runs).is_stopped(run_id)
        # A second session can't take the run over.
        resumed = CliRunner().invoke(cli.app, ["resume", run_id, "-c", str(config_file)])
        assert resumed.exit_code == 1
        assert f"Run {run_id} is being driven by another session." in resumed.output
    finally:
        session.release_all()


def test_stop_or_resume_of_an_unknown_or_finished_run(config_file: Path) -> None:
    runner = CliRunner()
    for command in ("stop", "resume"):
        unknown = runner.invoke(cli.app, [command, "run_nope", "-c", str(config_file)])
        assert unknown.exit_code == 1 and "No run run_nope." in unknown.output
    assert not (config_file.parent / "runs" / "run_nope").exists()
    done = runner.invoke(
        cli.app, ["run", "Build a books API", "-c", str(config_file)], input="20\naccept\n"
    )
    run_id = _run_id(done.output)
    assert f"Accepted. Run {run_id} is done." in done.output
    again = runner.invoke(cli.app, ["stop", run_id, "-c", str(config_file)])
    assert again.exit_code == 0 and f"Run {run_id} is already done." in again.output
    assert not KillSwitch(config_file.parent / "runs").is_stopped(run_id)


def test_stats_of_one_run_and_of_all_runs(config_file: Path) -> None:
    runner = CliRunner()
    empty = runner.invoke(cli.app, ["stats", "-c", str(config_file)])
    assert empty.exit_code == 0 and "No runs yet." in empty.output
    run_id = _left_waiting(config_file)
    wide = {"COLUMNS": "140"}
    one = runner.invoke(cli.app, ["stats", run_id, "-c", str(config_file)], env=wide)
    assert one.exit_code == 0, one.output
    lines = _plain(one).splitlines()
    assert lines[0].split() == [
        "calls",
        "tokens",
        "in",
        "tokens",
        "out",
        "cost",
        "time",
        "tasks",
        "retries",
        "first",
        "pass",
    ]
    assert lines[1].startswith(f"run {run_id} (waiting)")
    assert any(line.startswith("  product ") for line in lines), one.output
    every = runner.invoke(cli.app, ["stats", "-c", str(config_file)], env=wide)
    assert any(line.startswith("all runs") for line in _plain(every).splitlines())
    unknown = runner.invoke(cli.app, ["stats", "run_nope", "-c", str(config_file)])
    assert unknown.exit_code == 1 and "No run run_nope." in unknown.output


def test_context_folders_reach_the_run_read_only(config_file: Path, tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    run_id = _left_waiting(config_file, "--context", str(docs))
    run, _ = asyncio.run(_run_and_stops(config_file, run_id))
    assert run.context_dirs == [str(docs.resolve())]
    missing = CliRunner().invoke(
        cli.app, ["run", "x", "-c", str(config_file), "--context", str(tmp_path / "nope")]
    )
    assert missing.exit_code == 2 and "Invalid value for '--context'" in _plain(missing)
