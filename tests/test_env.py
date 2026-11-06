"""Loading ``.env`` at CLI startup."""

from __future__ import annotations

import os
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from agent_team import cli

KEY = "AGENT_TEAM_TEST_KEY"


@pytest.fixture
def unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start with KEY unset and remove it again afterwards, whoever set it."""
    monkeypatch.setenv(KEY, "placeholder")
    monkeypatch.delenv(KEY)


@pytest.mark.usefixtures("unset")
def test_env_file_is_loaded(tmp_path: Path) -> None:
    """Variables in .env end up in the environment."""
    (tmp_path / ".env").write_text(f"{KEY}=from-file\n")
    assert cli.load_env(tmp_path)
    assert os.environ[KEY] == "from-file"


def test_a_variable_already_set_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A variable set in the shell is not replaced by the file."""
    monkeypatch.setenv(KEY, "from-shell")
    (tmp_path / ".env").write_text(f"{KEY}=from-file\n")
    cli.load_env(tmp_path)
    assert os.environ[KEY] == "from-shell"


@pytest.mark.usefixtures("unset")
def test_a_missing_env_file_is_fine(tmp_path: Path) -> None:
    """No .env loads nothing and raises nothing."""
    assert not cli.load_env(tmp_path)
    assert KEY not in os.environ


@pytest.mark.usefixtures("unset")
def test_entry_loads_env_from_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command loads .env from where it runs before any command sees the config."""
    (tmp_path / ".env").write_text(f"{KEY}=from-cwd\n")
    monkeypatch.chdir(tmp_path)
    seen: list[str | None] = []
    monkeypatch.setattr(cli, "app", lambda: seen.append(os.environ.get(KEY)))
    cli.entry()
    assert seen == ["from-cwd"]


def test_the_installed_command_goes_through_entry() -> None:
    """``agent-team`` starts at ``entry``, so .env is loaded for every command."""
    [script] = entry_points(group="console_scripts", name="agent-team")
    assert script.value == "agent_team.cli:entry"
