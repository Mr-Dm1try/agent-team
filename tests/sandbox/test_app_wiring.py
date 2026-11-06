"""The composition root builds the sandbox and web tools from the config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from e2e.team import e2e_config

from agent_team import app as app_module
from agent_team.app import build_app
from agent_team.core.errors import SandboxError
from agent_team.llm.fake import FakeProvider
from sandbox.fakes import FakeRunner


def _provider() -> dict[str, Any]:
    """A fake provider for the e2e config's routes."""
    return {"fake": FakeProvider({})}


async def test_the_runner_backs_the_command_tools(tmp_path: Path) -> None:
    runner = FakeRunner()
    with await build_app(e2e_config(tmp_path), providers=_provider(), runner=runner) as app:
        tools = app.dispatcher.tools
        assert {"run_command", "install_deps", "web_search", "web_fetch"} <= set(tools)
        assert tools["run_command"].runner is runner  # type: ignore[attr-defined]
    assert runner.sweeps == 1
    assert not runner.closed  # a runner passed in belongs to the caller


async def test_disabled_sandbox_and_web_leave_only_file_tools(tmp_path: Path) -> None:
    config = e2e_config(
        tmp_path, sandbox={"enabled": False}, web={"enabled": False}, memory={"enabled": False}
    )
    with await build_app(config, providers=_provider()) as app:
        assert set(app.dispatcher.tools) == {"read_file", "list_dir", "grep", "write_file"}


async def test_no_docker_fails_fast_and_closes_what_it_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[str] = []
    real_database = app_module.Database

    class TrackedDatabase(real_database):  # type: ignore[misc, valid-type]
        def close(self) -> None:
            closed.append("db")
            super().close()

    async def no_docker(*args: Any) -> Any:
        """Docker isn't running."""
        raise SandboxError("Docker isn't reachable (test). There is no fallback.")

    monkeypatch.setattr(app_module, "Database", TrackedDatabase)
    monkeypatch.setattr(app_module.DockerRunner, "connect", no_docker)
    with pytest.raises(SandboxError, match="no fallback"):
        await build_app(e2e_config(tmp_path), providers=_provider())
    assert closed == ["db"]
