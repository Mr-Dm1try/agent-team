"""A git history of the workspace, with one commit per Engineer version that passes.

The commit hook listens to ``audit_verdict``. When an Engineer task's verdict passes
(the user keeping failed work also records a passing verdict), it commits the whole
workspace with a message naming the artifact version and tags the commit with it,
such as ``code_change-v3``. A change request can then diff two versions.

Git runs on the host, never in the container, with a fixed identity, no hooks, no
fsmonitor, no external diff drivers and no signing, so the user's git config can't
change or stall it. The command tools create the repository before the first command
runs, so the container only ever sees ``.git`` read-only, and the file tools refuse to
write there: agents can't rewrite the history or plant config the host's git would run.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from agent_team.core.artifacts import ArtifactRef
from agent_team.core.errors import SandboxError
from agent_team.core.interfaces import AfterHook, TaskStore
from agent_team.core.models import AgentName, Event
from agent_team.sandbox.base import ensure_gitignore

log = logging.getLogger(__name__)

_GIT_CONFIG = (
    "-c",
    "user.name=agent-team",
    "-c",
    "user.email=agent-team@localhost",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "tag.gpgsign=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "init.defaultBranch=main",
)


def version_tag(ref: ArtifactRef) -> str:
    """The tag that marks one artifact version, such as ``code_change-v3``."""
    return f"{ref.kind}-v{ref.version}"


class WorkspaceGit:
    """Commits, tags and diffs one run's workspace."""

    def __init__(self, workspace: str | Path) -> None:
        """Point at the run's workspace folder."""
        self.workspace = Path(workspace)

    async def _git(self, *args: str) -> str:
        """Run one git command in the workspace and return its stdout; SandboxError on
        failure."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *_GIT_CONFIG,
                *args,
                cwd=self.workspace,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "PATH": _path()},
            )
        except FileNotFoundError as exc:
            raise SandboxError("git isn't installed, so versions can't be committed") from exc
        out, err = await proc.communicate()
        if proc.returncode != 0:
            message = err.decode(errors="replace").strip() or f"exit {proc.returncode}"
            raise SandboxError(f"git {args[0]} failed: {message}")
        return out.decode(errors="replace")

    async def ensure_repo(self) -> None:
        """Create the repository and its .gitignore if the workspace has none yet. Runs
        before the first sandboxed command, so the container only ever sees a .git the
        harness made, mounted read-only; a .git that isn't a plain folder is refused."""
        await asyncio.to_thread(ensure_gitignore, self.workspace)
        dot_git = self.workspace / ".git"
        if dot_git.is_symlink() or (dot_git.exists() and not dot_git.is_dir()):
            raise SandboxError(f"{dot_git} is not a folder the harness created; refusing it")
        if not dot_git.is_dir():
            await self._git("init", "-q")

    async def commit_version(self, ref: ArtifactRef, *, subject: str, body: str = "") -> str:
        """Commit everything in the workspace for one artifact version, tag it and
        return the commit sha. A version with no file changes still gets a commit."""
        await self.ensure_repo()
        await self._git("add", "-A")
        message = f"{ref.kind} v{ref.version}: {subject}".strip()
        if body:
            message += f"\n\n{body}"
        await self._git("commit", "-q", "--allow-empty", "-m", message)
        await self._git("tag", "-f", version_tag(ref))
        return (await self._git("rev-parse", "HEAD")).strip()

    async def diff(self, old: ArtifactRef, new: ArtifactRef) -> str:
        """The workspace diff between two committed versions."""
        return await self._git(
            "diff", "--no-ext-diff", "--no-textconv", version_tag(old), version_tag(new)
        )


def make_commit_hook(tasks: TaskStore) -> AfterHook:
    """An ``audit_verdict`` hook that commits the workspace when an Engineer version
    passes. Failures are logged, never raised: a missing commit costs a diff, not the run."""

    async def hook(event: Event) -> None:
        """Commit the Engineer task's output version if its verdict passed."""
        if not event.data.get("passed") or event.task_id is None:
            return
        task = await tasks.get_task(event.task_id)
        if task.agent is not AgentName.ENGINEER or task.output is None:
            return
        run = await tasks.get_run(event.run_id)
        try:
            sha = await WorkspaceGit(run.workspace).commit_version(
                task.output, subject=task.objective[:72], body=f"Task {task.id}"
            )
        except SandboxError:
            log.warning("couldn't commit %s for task %s", task.output, task.id, exc_info=True)
            return
        log.info("committed %s as %s", task.output, sha[:12])

    return hook


def _path() -> str:
    """The host PATH, so git is found where the user installed it."""
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
