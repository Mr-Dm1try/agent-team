"""The command line over the real team with a scripted model: a run answered from
stdin, a run left waiting and resumed, and reading the logs back."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
import yaml
from agents.scenario import PRD
from typer.testing import CliRunner

from agent_team import cli
from agent_team.app import App
from agent_team.core.config import Config
from agent_team.core.models import AgentName
from agent_team.llm.types import LLMRequest
from e2e.team import e2e_config, e2e_script, scripted_app

QUESTION = [{"id": "Q-1", "text": "Paginate the list?", "why": "list size"}]


def _asking_team() -> Callable[[Config], Awaitable[App]]:
    """A team whose Product asks one question before writing the PRD."""
    script = e2e_script()
    calls: list[int] = []

    def product(request: LLMRequest) -> str:
        """Ask on the first attempt, then write a plain PRD."""
        calls.append(1)
        return json.dumps(dict(PRD, questions=QUESTION) if len(calls) == 1 else PRD)

    script[AgentName.PRODUCT] = product

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


def _run_id(output: str) -> str:
    """The run id the CLI printed."""
    found = re.search(r"\[run (run_\w+): ", output)
    assert found, output
    return found.group(1)


def test_run_answers_questions_from_stdin(config_file: Path) -> None:
    result = CliRunner().invoke(
        cli.app, ["run", "Build a books API", "-c", str(config_file)], input="20 per page\n"
    )
    assert result.exit_code == 0, result.output
    assert "Paginate the list?" in result.output
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
