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
A core dump (a file named ``core`` or ``core.*`` that is an ELF core file) is never
committed, should a crash leave one. Files the code writes when it runs (``GENERATED``:
databases and Python caches) aren't part of the change either: they are left out of
every version and every diff, unless the Engineer reports writing one on purpose.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from pathlib import Path

from agent_team.core.artifacts import ArtifactRef, CodeChange
from agent_team.core.errors import SandboxError
from agent_team.core.interfaces import AfterHook, ArtifactStore, TaskStore
from agent_team.core.models import AgentName, Event
from agent_team.sandbox.base import ensure_gitignore

log = logging.getLogger(__name__)

#: ELF type of a core file (``ET_CORE``).
_ET_CORE = 4
#: Paths a core dump can have: ``core`` or ``core.*`` in any folder.
_CORE_PATHS = (":(glob)**/core", ":(glob)**/core.*")
#: Files the code writes when it runs, such as a SQLite database a module creates on
#: import, left out of every version and every diff.
GENERATED = ("**/*.db", "**/*.sqlite", "**/*.sqlite3", "**/__pycache__/**", "**/*.pyc")
_SKIP_GENERATED = tuple(f":(exclude,glob){g}" for g in GENERATED)
_GENERATED_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".pyc")


def is_generated(path: str) -> bool:
    """Whether a workspace path looks like a file the code writes when it runs."""
    return path.endswith(_GENERATED_SUFFIXES) or "__pycache__" in path.split("/")


def is_core_dump(path: Path) -> bool:
    """Whether ``path`` is a core dump: named ``core`` or ``core.*`` and an ELF core
    file, so a ``core.py`` or a ``core`` package folder never counts."""
    if not (path.name == "core" or path.name.startswith("core.")) or not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            head = f.read(18)
    except OSError:
        return False
    if len(head) < 18 or head[:4] != b"\x7fELF" or head[5] not in (1, 2):
        return False
    kind = int.from_bytes(head[16:18], "little" if head[5] == 1 else "big")
    return kind == _ET_CORE


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

    async def commit_version(
        self, ref: ArtifactRef, *, subject: str, body: str = "", keep: Iterable[str] = ()
    ) -> str:
        """Commit everything in the workspace for one artifact version, tag it and
        return the commit sha. A version with no file changes still gets a commit.
        Generated files are left out, except those in ``keep``: the files the Engineer
        reports writing, such as a seed database the spec plans."""
        await self.ensure_repo()
        dumps = await self.core_dumps()
        await self._git(
            "add", "-A", "--", ".", *_SKIP_GENERATED, *(f":(exclude,literal){d}" for d in dumps)
        )
        wanted = sorted(
            p
            for p in keep
            if is_generated(p) and self._inside(p) and (self.workspace / p).is_file()
        )
        if wanted:
            await self._git("add", "-f", "--", *(f":(literal){p}" for p in wanted))
        if dumps:
            log.warning("leaving core dumps out of the commit: %s", ", ".join(dumps))
            # One a version before this committed leaves the history from here on.
            await self._git(
                "rm", "-q", "--cached", "--ignore-unmatch", "--", *(f":(literal){d}" for d in dumps)
            )
        message = f"{ref.kind} v{ref.version}: {subject}".strip()
        if body:
            message += f"\n\n{body}"
        await self._git("commit", "-q", "--allow-empty", "-m", message)
        await self._git("tag", "-f", version_tag(ref))
        return (await self._git("rev-parse", "HEAD")).strip()

    def _inside(self, path: str) -> bool:
        """Whether a reported path is a relative path inside the workspace."""
        if Path(path).is_absolute():
            return False
        root = self.workspace.resolve()
        return (root / path).resolve().is_relative_to(root)

    async def core_dumps(self) -> list[str]:
        """The workspace's core dumps that git would see (tracked, or untracked and not
        ignored), found by name through git and confirmed by their ELF header, so
        nothing is hashed or walked that the commit wouldn't touch."""
        out = await self._git(
            "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *_CORE_PATHS
        )
        names = {n for n in out.split("\0") if n}
        return sorted(n for n in names if is_core_dump(self.workspace / n))

    async def diff(self, old: ArtifactRef, new: ArtifactRef) -> str:
        """The workspace diff between two committed versions, without generated files,
        which a version from before they were left out may still hold."""
        return await self._git(
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            version_tag(old),
            version_tag(new),
            "--",
            ".",
            *_SKIP_GENERATED,
        )


def make_commit_hook(tasks: TaskStore, artifacts: ArtifactStore | None = None) -> AfterHook:
    """An ``audit_verdict`` hook that commits the workspace when an Engineer version
    passes, keeping the generated-looking files its code change reports writing.
    Failures are logged, never raised: a missing commit costs a diff, not the run."""

    async def hook(event: Event) -> None:
        """Commit the Engineer task's output version if its verdict passed."""
        if not event.data.get("passed") or event.task_id is None:
            return
        task = await tasks.get_task(event.task_id)
        if task.agent is not AgentName.ENGINEER or task.output is None:
            return
        run = await tasks.get_run(event.run_id)
        written: list[str] = []
        if artifacts is not None:
            code = await artifacts.load(run.id, task.output)
            written = code.files_written if isinstance(code, CodeChange) else []
        try:
            sha = await WorkspaceGit(run.workspace).commit_version(
                task.output, subject=task.objective[:72], body=f"Task {task.id}", keep=written
            )
        except SandboxError:
            log.warning("couldn't commit %s for task %s", task.output, task.id, exc_info=True)
            return
        log.info("committed %s as %s", task.output, sha[:12])

    return hook


def _path() -> str:
    """The host PATH, so git is found where the user installed it."""
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
