"""The real sandbox: install pytest with the network on, then run it with the network
off. Needs a Docker daemon, so it is skipped where there is none."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from agents.fakes import FakeBus

from agent_team.core.config import SandboxConfig
from agent_team.core.errors import SandboxError
from agent_team.core.interfaces import ToolContext
from agent_team.core.models import AgentName, Phase, Run, Task
from agent_team.sandbox import DockerRunner
from agent_team.tools import command_tools

pytestmark = pytest.mark.docker

CONFIG = SandboxConfig(timeout_s=120, install_timeout_s=300)
TEST = "def test_add():\n    assert 1 + 1 == 2\n"
NETWORK_PROBE = (
    "import socket\n"
    "try:\n"
    "    socket.create_connection(('pypi.org', 443), timeout=5)\n"
    "except OSError:\n"
    "    print('offline')\n"
    "else:\n"
    "    print('online')\n"
)


@pytest.fixture
async def docker_runner(tmp_path: Path) -> AsyncIterator[DockerRunner]:
    try:
        runner = await DockerRunner.connect(CONFIG, tmp_path / "runs")
    except SandboxError as exc:
        pytest.skip(f"no Docker daemon: {exc}")
    try:
        yield runner
    finally:
        runner.close()


async def test_pytest_runs_in_the_container_with_the_network_off(
    docker_runner: DockerRunner, tmp_path: Path
) -> None:
    ws = tmp_path / "runs" / "run_1" / "workspace"
    ws.mkdir(parents=True)
    (ws / "test_add.py").write_text(TEST)
    (ws / "probe.py").write_text(NETWORK_PROBE)
    run = Run(title="t", request="r", workspace=str(ws))
    task = Task(
        run_id=run.id,
        phase=Phase.IMPLEMENTATION,
        agent=AgentName.ENGINEER,
        objective="o",
        output_kind="code_change",
    )
    ctx = ToolContext(run=run, task=task, bus=FakeBus())
    tools = command_tools(docker_runner, CONFIG)

    installed = await tools["install_deps"]({"packages": ["pytest==8.3.3"]}, ctx)
    assert not installed.is_error, installed.content
    assert (ws / ".deps" / "pytest").is_dir()

    tests = await tools["run_command"]({"argv": ["pytest", "-q", "test_add.py"]}, ctx)
    assert not tests.is_error, tests.content
    assert "1 passed" in tests.content and tests.data and tests.data["exit_code"] == 0

    probe = await tools["run_command"]({"argv": ["python", "probe.py"]}, ctx)
    assert "offline" in probe.content, probe.content

    slow = await tools["run_command"](
        {"argv": ["python", "-c", "import time; time.sleep(30)"], "timeout_s": 2}, ctx
    )
    assert slow.is_error and slow.data and slow.data["timed_out"] and slow.data["exit_code"] == 124

    # A crash leaves no core dump in the workspace.
    crash = await tools["run_command"]({"argv": ["python", "-c", "import os; os.abort()"]}, ctx)
    assert crash.data and crash.data["exit_code"] == 134, crash.content
    assert not [p for p in ws.iterdir() if p.name.startswith("core")]

    # Every container was removed when its command ended.
    assert await docker_runner.cleanup(run_id=run.id) == 0
