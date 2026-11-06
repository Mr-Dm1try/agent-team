"""DockerRunner over a fake Docker client: container settings, timeouts, cleanup."""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from agent_team.core.config import SandboxConfig
from agent_team.core.errors import SandboxError
from agent_team.sandbox import CommandSpec, DockerRunner
from agent_team.sandbox.base import COMMAND_ENV
from agent_team.sandbox.docker_runner import OWNER_LABEL, RUN_LABEL, TASK_LABEL, root_label

CONFIG = SandboxConfig(cpus=0.5, memory="512m", pids_limit=64, output_tail_chars=10)


class ReadTimeout(Exception):
    """Named like the HTTP library's timeout, which the SDK raises from wait()."""


class FakeContainer:
    """A container that 'runs' by returning a scripted status."""

    def __init__(self, client: FakeClient, kwargs: dict[str, Any]) -> None:
        self.client = client
        self.kwargs = kwargs
        self.labels: dict[str, str] = kwargs.get("labels", {})
        self.started = self.killed = self.removed = False

    def start(self) -> None:
        self.started = True
        self.client.running.set()

    def wait(self, timeout: float) -> dict[str, int]:
        return self.client.wait(self, timeout)

    def kill(self) -> None:
        self.killed = True

    def logs(self, *, stdout: bool, stderr: bool, tail: int) -> bytes:
        return b"0123456789-the end of stdout" if stdout else b"warn\n"

    def remove(self, *, force: bool) -> None:
        self.removed = True
        self.client.release.set()
        if self in self.client.created:
            self.client.created.remove(self)


class FakeContainers:
    """The ``client.containers`` collection."""

    def __init__(self, client: FakeClient) -> None:
        self.client = client

    def create(self, image: str, **kwargs: Any) -> FakeContainer:
        container = FakeContainer(self.client, {"image": image, **kwargs})
        self.client.created.append(container)
        self.client.all.append(container)
        return container

    def list(self, *, all: bool, filters: dict[str, list[str]]) -> list[FakeContainer]:
        wanted = [f.split("=", 1) for f in filters["label"]]
        return [c for c in self.client.created if all and all_match(c, wanted)]


def all_match(container: FakeContainer, wanted: list[list[str]]) -> bool:
    """Whether the container carries every wanted label value."""
    return all(container.labels.get(k) == v for k, v in wanted)


class FakeClient:
    """Just enough of docker.DockerClient."""

    def __init__(self, status: int = 0) -> None:
        self.status = status
        self.created: list[FakeContainer] = []
        self.all: list[FakeContainer] = []
        self.containers = FakeContainers(self)
        self.release = threading.Event()
        self.running = threading.Event()
        self.hang = False
        self.closed = False

    def wait(self, container: FakeContainer, timeout: float) -> dict[str, int]:
        if self.hang:  # blocks until the container is removed, like a real wait
            self.release.wait(5)
            return {"StatusCode": 137}
        if timeout < 1:
            raise ReadTimeout("read timed out")
        return {"StatusCode": self.status}

    def close(self) -> None:
        self.closed = True


def spec(tmp_path: Path, **kw: Any) -> CommandSpec:
    """A command in a workspace with a .git folder."""
    ws = tmp_path / "ws"
    (ws / ".git").mkdir(parents=True, exist_ok=True)
    (ws / "src").mkdir(exist_ok=True)
    base: dict[str, Any] = {
        "argv": ["pytest", "-q"],
        "workspace": str(ws),
        "run_id": "run_1",
        "task_id": "task_1",
        "timeout_s": 30,
    }
    return CommandSpec(**{**base, **kw})


def runner(client: FakeClient, tmp_path: Path) -> DockerRunner:
    """A runner over the fake client."""
    return DockerRunner(client, CONFIG, root=root_label(tmp_path))


async def test_a_command_runs_locked_down_and_offline(tmp_path: Path) -> None:
    client = FakeClient(status=3)
    result = await runner(client, tmp_path).run(spec(tmp_path, cwd="src"))
    assert result.exit_code == 3 and not result.timed_out
    assert result.stdout.endswith("of stdout") and "chars cut" in result.stdout
    assert result.stderr == "warn\n"
    [container] = client.all
    kw = container.kwargs
    assert kw["image"] == "python:3.12-slim" and kw["command"] == ["pytest", "-q"]
    assert kw["network_mode"] == "none" and kw["working_dir"] == "/workspace/src"
    assert kw["user"] == f"{os.getuid()}:{os.getgid()}"
    assert kw["cap_drop"] == ["ALL"] and kw["security_opt"] == ["no-new-privileges"]
    assert kw["read_only"] and "/tmp" in kw["tmpfs"]
    assert (kw["mem_limit"], kw["nano_cpus"], kw["pids_limit"]) == ("512m", 500_000_000, 64)
    ws = str(tmp_path / "ws")
    assert kw["volumes"][ws] == {"bind": "/workspace", "mode": "rw"}
    assert kw["volumes"][f"{ws}/.git"] == {"bind": "/workspace/.git", "mode": "ro"}
    assert kw["environment"]["PYTHONPATH"] == "/workspace/.deps"
    assert (kw["labels"][RUN_LABEL], kw["labels"][TASK_LABEL]) == ("run_1", "task_1")
    assert container.started and container.removed and client.created == []


async def test_containers_get_none_of_the_host_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API key in the harness's environment never reaches a container."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host-only")
    client = FakeClient()
    await runner(client, tmp_path).run(spec(tmp_path, network=True))
    [container] = client.all
    assert container.kwargs["environment"] == COMMAND_ENV


async def test_only_install_steps_get_the_network(tmp_path: Path) -> None:
    client = FakeClient()
    await runner(client, tmp_path).run(spec(tmp_path, network=True))
    assert client.all[0].kwargs["network_mode"] == "bridge"


async def test_a_timeout_kills_the_container(tmp_path: Path) -> None:
    client = FakeClient()
    result = await runner(client, tmp_path).run(spec(tmp_path, timeout_s=0.001))
    assert result.timed_out and result.exit_code == 124
    assert client.all[0].killed and client.all[0].removed


async def test_a_cancelled_call_still_removes_its_container(tmp_path: Path) -> None:
    client = FakeClient()
    client.hang = True
    call = asyncio.create_task(runner(client, tmp_path).run(spec(tmp_path)))
    assert await asyncio.to_thread(client.running.wait, 5)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    for _ in range(100):  # the shielded removal finishes in its thread
        if client.all[0].removed:
            break
        await asyncio.sleep(0.01)
    assert client.all[0].removed


async def test_a_call_cancelled_while_creating_removes_the_container(tmp_path: Path) -> None:
    client = FakeClient()
    creating = threading.Event()
    go_on = threading.Event()
    real_create = client.containers.create

    def slow_create(image: str, **kwargs: Any) -> FakeContainer:
        """Create the container only after the call was cancelled."""
        creating.set()
        go_on.wait(5)
        return real_create(image, **kwargs)

    client.containers.create = slow_create  # type: ignore[method-assign]
    call = asyncio.create_task(runner(client, tmp_path).run(spec(tmp_path)))
    assert await asyncio.to_thread(creating.wait, 5)
    call.cancel()
    await asyncio.sleep(0)
    go_on.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    [container] = client.all
    assert container.removed and not container.started


async def test_docker_errors_become_sandbox_errors(tmp_path: Path) -> None:
    client = FakeClient()

    def broken_create(image: str, **kwargs: Any) -> FakeContainer:
        """The daemon refuses the container."""
        raise RuntimeError("no space left on device")

    client.containers.create = broken_create  # type: ignore[method-assign]
    with pytest.raises(SandboxError, match="no space left"):
        await runner(client, tmp_path).run(spec(tmp_path))


async def test_cleanup_removes_by_task_and_run(tmp_path: Path) -> None:
    client = FakeClient()
    r = runner(client, tmp_path)
    labels = {RUN_LABEL: "run_1", TASK_LABEL: "task_1"}
    left = [client.containers.create("img", labels={**r_labels(r), **labels}) for _ in range(2)]
    other = client.containers.create("img", labels={**r_labels(r), RUN_LABEL: "run_2"})
    assert await r.cleanup() == 0  # no filter, nothing removed
    assert await r.cleanup(task_id="task_1") == 2
    assert all(c.removed for c in left) and not other.removed
    assert await r.cleanup(run_id="run_2") == 1 and other.removed


async def test_sweep_removes_only_dead_owners(tmp_path: Path) -> None:
    client = FakeClient()
    r = runner(client, tmp_path)
    mine = client.containers.create("img", labels=r_labels(r))
    dead = client.containers.create("img", labels={**r_labels(r), OWNER_LABEL: _dead_owner()})
    elsewhere = client.containers.create("img", labels={**r_labels(r), OWNER_LABEL: "other-host:1"})
    other_root = client.containers.create(
        "img", labels={**r_labels(r), "agent-team.root": "another-checkout", OWNER_LABEL: "x:1"}
    )
    assert await r.sweep() == 1
    assert dead.removed and not mine.removed and not elsewhere.removed
    assert not other_root.removed
    r.close()
    assert mine.removed and client.closed and not elsewhere.removed


async def test_connect_without_docker_fails_fast_with_help(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import docker

    def no_daemon() -> Any:
        """Docker isn't running."""
        raise docker.errors.DockerException("Error while fetching server API version")

    monkeypatch.setattr(docker, "from_env", no_daemon)
    with pytest.raises(SandboxError) as caught:
        await DockerRunner.connect(CONFIG, tmp_path)
    message = str(caught.value)
    assert "Docker isn't reachable" in message and "no fallback" in message
    assert "Start Docker" in message


def r_labels(r: DockerRunner) -> dict[str, str]:
    """The labels every container of this runner carries."""
    return {"agent-team": "1", "agent-team.root": r.root, OWNER_LABEL: r.owner}


def _dead_owner() -> str:
    """An owner label for a pid that no longer exists on this host."""
    import socket
    import subprocess

    proc = subprocess.Popen(["true"])
    proc.wait()
    return f"{socket.gethostname()}:{proc.pid}"
