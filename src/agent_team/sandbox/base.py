"""What every command runner shares: the command, its result, and the workspace layout.

Dependencies install into ``.deps`` inside the workspace, so they survive between
commands and tasks and a resumed run still has them. Commands see them through
``PYTHONPATH`` and ``PATH``. The workspace's ``.gitignore`` keeps ``.deps`` and the
caches out of the per-version commits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

#: Where the run's workspace is mounted in the container.
CONTAINER_WORKSPACE = "/workspace"
#: The workspace folder that install_deps installs into.
DEPS_DIR = ".deps"
#: Exit code reported for a command killed at its timeout, as ``timeout(1)`` does.
TIMEOUT_EXIT = 124

GITIGNORE_LINES = (
    f"{DEPS_DIR}/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
)

#: The environment every command gets, network on or off.
COMMAND_ENV = {
    "HOME": "/tmp",
    "PYTHONPATH": f"{CONTAINER_WORKSPACE}/{DEPS_DIR}",
    "PATH": f"{CONTAINER_WORKSPACE}/{DEPS_DIR}/bin:/usr/local/bin:/usr/bin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_CACHE_DIR": "1",
}


@dataclass(frozen=True)
class CommandSpec:
    """One command to run in the sandbox."""

    argv: list[str]
    workspace: str  # absolute host path of the run's workspace
    run_id: str
    task_id: str | None
    timeout_s: float
    cwd: str = "."  # workspace-relative, or an absolute path in the container
    network: bool = False
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandResult:
    """What a command did: its exit code, the ends of its output, and how long it took."""

    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


class CommandRunner(Protocol):
    """Runs commands in isolation and cleans up after them."""

    async def run(self, spec: CommandSpec) -> CommandResult:
        """Run one command to completion or its timeout."""
        ...

    async def cleanup(self, *, run_id: str | None = None, task_id: str | None = None) -> int:
        """Remove what is left of the run's or task's commands; return how many."""
        ...

    async def sweep(self) -> int:
        """Remove what crashed processes left behind; return how many."""
        ...

    def close(self) -> None:
        """Remove this process's leftovers and release the connection."""
        ...


def tail(text: str, limit: int) -> str:
    """The last ``limit`` characters of ``text``, saying how much was cut."""
    if len(text) <= limit:
        return text
    return f"[... {len(text) - limit} chars cut]\n" + text[-limit:]


def ensure_gitignore(workspace: str | Path) -> None:
    """Add the deps folder and caches to the workspace's .gitignore, keeping its lines."""
    path = Path(workspace) / ".gitignore"
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    missing = [line for line in GITIGNORE_LINES if line not in existing]
    if not missing:
        return
    lines = [*existing, *missing]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
