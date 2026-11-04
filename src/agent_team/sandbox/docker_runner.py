"""Runs commands in Docker containers, one short-lived container per command.

Each container mounts the run's workspace read-write and its ``.git`` read-only, runs
as the host user so files stay theirs, has a read-only root filesystem with a small
``/tmp``, drops every capability and has CPU, memory and process limits. The network
is off unless the command asks for it, which only install_deps does.

Every container carries labels naming this harness, its runs folder, the process that
started it, the run and the task. The container is removed when its command ends,
even if the call is cancelled. The cleanup hooks remove any left at the end of a task
or run, and ``sweep`` removes those of processes that died.

Without Docker there is no fallback: ``connect`` raises SandboxError.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import os
import posixpath
import socket
import time
from pathlib import Path
from typing import Any

from agent_team.core.config import SandboxConfig
from agent_team.core.errors import SandboxError
from agent_team.sandbox.base import (
    COMMAND_ENV,
    CONTAINER_WORKSPACE,
    TIMEOUT_EXIT,
    CommandResult,
    CommandSpec,
    tail,
)

log = logging.getLogger(__name__)

LABEL = "agent-team"
ROOT_LABEL = "agent-team.root"
OWNER_LABEL = "agent-team.owner"
RUN_LABEL = "agent-team.run"
TASK_LABEL = "agent-team.task"

NO_DOCKER = (
    "Docker isn't reachable ({reason}). Generated code only runs inside the Docker "
    "sandbox, and there is no fallback without it. Start Docker (Docker Desktop, or "
    "`sudo systemctl start docker`), check that `docker info` works for this user, "
    "then run again. To run without code execution, set `sandbox: {{enabled: false}}` "
    "in the config; R6 will then fail every build."
)


def root_label(runs_dir: str | Path) -> str:
    """A short id for the runs folder, so two checkouts never sweep each other's
    containers."""
    return hashlib.sha256(str(Path(runs_dir).resolve()).encode()).hexdigest()[:16]


def owner_label() -> str:
    """This process, as host and pid."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _owner_alive(owner: str) -> bool:
    """Whether the process named by an owner label may still be running. One on
    another host counts as alive, since this host can't tell; a missing or malformed
    label counts as dead."""
    host, _, pid = owner.rpartition(":")
    if not host or not pid.isdigit():
        return False
    if host != socket.gethostname():
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, under another user
    return True


class DockerRunner:
    """Implements ``CommandRunner`` with the Docker SDK."""

    def __init__(self, client: Any, config: SandboxConfig, *, root: str) -> None:
        """Wrap a connected Docker client; ``root`` labels this runs folder."""
        self.client = client
        self.config = config
        self.root = root
        self.owner = owner_label()

    @classmethod
    async def connect(cls, config: SandboxConfig, runs_dir: str | Path) -> DockerRunner:
        """Connect to Docker and make sure the image is there; SandboxError if not."""

        def _connect() -> Any:
            """Open the client, ping the daemon and pull the image if it's missing."""
            import docker

            try:
                client = docker.from_env()
                client.ping()
            except Exception as exc:
                raise SandboxError(NO_DOCKER.format(reason=_reason(exc))) from exc
            try:
                client.images.get(config.image)
            except docker.errors.ImageNotFound:
                log.info("pulling %s for the sandbox", config.image)
                try:
                    client.images.pull(config.image)
                except Exception as exc:
                    raise SandboxError(f"can't pull {config.image}: {_reason(exc)}") from exc
            return client

        client = await asyncio.to_thread(_connect)
        return cls(client, config, root=root_label(runs_dir))

    # --- running ---------------------------------------------------------------------

    async def run(self, spec: CommandSpec) -> CommandResult:
        """Run one command in a fresh container and remove the container afterwards."""
        creating = asyncio.ensure_future(asyncio.to_thread(self._create, spec))
        try:
            container = await asyncio.shield(creating)
        except asyncio.CancelledError:
            # Cancelled mid-create: the container still appears, so wait and remove it.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self._remove, await creating)
            raise
        try:
            return await asyncio.to_thread(self._wait, container, spec)
        finally:
            # Shielded so a cancelled call (a task timeout) still kills its container.
            await asyncio.shield(asyncio.to_thread(self._remove, container))

    def _create(self, spec: CommandSpec) -> Any:
        """Create the container for one command, without starting it."""
        workspace = Path(spec.workspace).resolve()
        volumes = {str(workspace): {"bind": CONTAINER_WORKSPACE, "mode": "rw"}}
        if (workspace / ".git").is_dir():
            volumes[str(workspace / ".git")] = {"bind": f"{CONTAINER_WORKSPACE}/.git", "mode": "ro"}
        labels = {
            LABEL: "1",
            ROOT_LABEL: self.root,
            OWNER_LABEL: self.owner,
            RUN_LABEL: spec.run_id,
            TASK_LABEL: spec.task_id or "",
        }
        try:
            return self.client.containers.create(
                self.config.image,
                command=list(spec.argv),
                working_dir=posixpath.normpath(posixpath.join(CONTAINER_WORKSPACE, spec.cwd)),
                user=f"{os.getuid()}:{os.getgid()}",
                environment={**COMMAND_ENV, **spec.env},
                network_mode="bridge" if spec.network else "none",
                mem_limit=self.config.memory,
                memswap_limit=self.config.memory,
                nano_cpus=int(self.config.cpus * 1_000_000_000),
                pids_limit=self.config.pids_limit,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                read_only=True,
                tmpfs={"/tmp": "rw,size=256m"},
                volumes=volumes,
                labels=labels,
                detach=True,
            )
        except Exception as exc:
            raise SandboxError(f"can't create the sandbox container: {_reason(exc)}") from exc

    def _wait(self, container: Any, spec: CommandSpec) -> CommandResult:
        """Start the container, wait up to the timeout and collect the ends of its output."""
        started = time.monotonic()
        timed_out = False
        try:
            container.start()
        except Exception as exc:
            raise SandboxError(f"can't start the sandbox container: {_reason(exc)}") from exc
        try:
            status = container.wait(timeout=spec.timeout_s)
            exit_code = int(status.get("StatusCode", 1))
        except Exception as exc:
            waited = time.monotonic() - started
            if not (_is_timeout(exc) and waited >= spec.timeout_s - 1):
                raise SandboxError(f"lost the sandbox container: {_reason(exc)}") from exc
            timed_out = True
            exit_code = TIMEOUT_EXIT
            try:
                container.kill()
            except Exception:
                log.debug("container already stopped at its timeout", exc_info=True)
        duration = round(time.monotonic() - started, 3)
        limit = self.config.output_tail_chars
        return CommandResult(
            exit_code=exit_code,
            stdout=tail(self._logs(container, stdout=True), limit),
            stderr=tail(self._logs(container, stdout=False), limit),
            duration_s=duration,
            timed_out=timed_out,
        )

    def _logs(self, container: Any, *, stdout: bool) -> str:
        """The end of the container's stdout or stderr, decoded leniently."""
        try:
            raw = container.logs(stdout=stdout, stderr=not stdout, tail=2000)
        except Exception:
            log.debug("no logs from the sandbox container", exc_info=True)
            return ""
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    @staticmethod
    def _remove(container: Any) -> None:
        """Kill and remove one container; gone already is fine."""
        try:
            container.remove(force=True)
        except Exception:
            log.debug("sandbox container already removed", exc_info=True)

    # --- cleanup ---------------------------------------------------------------------

    def _list(self, **labels: str) -> list[Any]:
        """This runs folder's containers that carry every given label value."""
        filters = [f"{ROOT_LABEL}={self.root}", *(f"{k}={v}" for k, v in labels.items())]
        try:
            return list(self.client.containers.list(all=True, filters={"label": filters}))
        except Exception:
            log.warning("can't list sandbox containers", exc_info=True)
            return []

    def _remove_all(self, containers: list[Any]) -> int:
        """Remove each container; return how many there were."""
        for container in containers:
            self._remove(container)
        return len(containers)

    async def cleanup(self, *, run_id: str | None = None, task_id: str | None = None) -> int:
        """Remove the containers left by one run's or one task's commands."""
        labels: dict[str, str] = {}
        if run_id:
            labels[RUN_LABEL] = run_id
        if task_id:
            labels[TASK_LABEL] = task_id
        if not labels:
            return 0

        def _cleanup() -> int:
            """List and remove the matching containers."""
            return self._remove_all(self._list(**labels))

        return await asyncio.to_thread(_cleanup)

    async def sweep(self) -> int:
        """Remove the containers of processes that are no longer running."""

        def _sweep() -> int:
            """Remove each container whose owner process is gone."""
            dead = [
                c
                for c in self._list()
                if not _owner_alive(str((c.labels or {}).get(OWNER_LABEL, "")))
            ]
            return self._remove_all(dead)

        return await asyncio.to_thread(_sweep)

    def close(self) -> None:
        """Remove this process's containers and close the Docker client."""
        self._remove_all(self._list(**{OWNER_LABEL: self.owner}))
        try:
            self.client.close()
        except Exception:
            log.debug("docker client already closed", exc_info=True)


def _is_timeout(exc: BaseException) -> bool:
    """Whether a wait failed because its timeout ran out (the SDK raises the HTTP
    library's read timeout or connection error)."""
    names = {type(e).__name__ for e in (exc, exc.__cause__, exc.__context__) if e}
    return bool(names & {"ReadTimeout", "ReadTimeoutError", "ConnectionError", "TimeoutError"})


def _reason(exc: BaseException) -> str:
    """A one-line reason for a Docker error."""
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0]}" if text else type(exc).__name__
