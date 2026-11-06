"""The workspace's git history: one tagged commit per passing Engineer version."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest
from audit.fakes import FakeTaskStore

from agent_team.core.artifacts import ArtifactRef
from agent_team.core.errors import SandboxError
from agent_team.core.models import AgentName, Event, EventName, Phase, Run, Task
from agent_team.sandbox import WorkspaceGit, make_commit_hook

V1 = ArtifactRef(kind="code_change", version=1)
V2 = ArtifactRef(kind="code_change", version=2)


def git(ws: Path, *args: str) -> list[str]:
    """Run git in the workspace and return its output lines."""
    return subprocess.run(
        ["git", *args], cwd=ws, capture_output=True, text=True, check=True
    ).stdout.splitlines()


async def test_versions_are_committed_tagged_and_diffable(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "app.py").write_text("x = 1\n")
    (ws / ".deps").mkdir()
    (ws / ".deps" / "lib.py").write_text("vendored\n")
    repo = WorkspaceGit(ws)
    sha = await repo.commit_version(V1, subject="books API", body="Task t1")
    (ws / "app.py").write_text("x = 2\n")
    await repo.commit_version(V2, subject="books API, fixed")
    assert git(ws, "rev-parse", "code_change-v1") == [sha]
    assert git(ws, "log", "--format=%s") == [
        "code_change v2: books API, fixed",
        "code_change v1: books API",
    ]
    assert sorted(git(ws, "ls-files")) == [".gitignore", "app.py"]  # no deps, no caches
    diff = await repo.diff(V1, V2)
    assert "-x = 1" in diff and "+x = 2" in diff
    # A version with no file changes still gets its own commit and tag.
    await repo.commit_version(ArtifactRef(kind="code_change", version=3), subject="same")
    assert await repo.diff(V2, ArtifactRef(kind="code_change", version=3)) == ""


async def test_the_users_git_config_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text("[commit]\n\tgpgsign = true\n[user]\n\tname = Someone\n")
    monkeypatch.setenv("HOME", str(home))
    ws = tmp_path / "workspace"
    ws.mkdir()
    await WorkspaceGit(ws).commit_version(V1, subject="s")
    assert git(ws, "log", "--format=%an") == ["agent-team"]


async def test_a_missing_git_is_a_sandbox_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    with pytest.raises(SandboxError, match="git isn't installed"):
        await WorkspaceGit(tmp_path).commit_version(V1, subject="s")


def _setup(tmp_path: Path, agent: AgentName) -> tuple[FakeTaskStore, Run, Task]:
    """A run with one finished task of ``agent``."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True)
    (ws / "app.py").write_text("print('hi')\n")
    tasks = FakeTaskStore()
    run = Run(title="t", request="r", workspace=str(ws))
    tasks.runs[run.id] = run
    task = Task(
        run_id=run.id,
        phase=Phase.IMPLEMENTATION,
        agent=agent,
        objective="Build the books API",
        output_kind="code_change" if agent is AgentName.ENGINEER else "qa_report",
        output=V1 if agent is AgentName.ENGINEER else ArtifactRef(kind="qa_report", version=1),
    )
    tasks.tasks[task.id] = task
    return tasks, run, task


def has_repo(run: Run) -> bool:
    """Whether the run's workspace has a git repository."""
    return Path(run.workspace, ".git").exists()


def _verdict(run: Run, task: Task, **data: Any) -> Event:
    """An audit_verdict event for the task."""
    return Event(
        name=EventName.AUDIT_VERDICT,
        run_id=run.id,
        task_id=task.id,
        agent=AgentName.AUDITOR,
        data={"passed": True, **data},
    )


async def test_the_hook_commits_only_passing_engineer_versions(tmp_path: Path) -> None:
    tasks, run, task = _setup(tmp_path, AgentName.ENGINEER)
    hook = make_commit_hook(tasks)
    await hook(_verdict(run, task, passed=False))
    assert not has_repo(run)
    await hook(_verdict(run, task))
    ws = Path(run.workspace)
    assert git(ws, "log", "--format=%s%n%b") == [
        "code_change v1: Build the books API",
        f"Task {task.id}",
        "",
    ]

    tasks, run, qa = _setup(tmp_path / "qa", AgentName.QA)
    await make_commit_hook(tasks)(_verdict(run, qa))
    assert not has_repo(run)


async def test_a_failed_commit_is_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    tasks, run, task = _setup(tmp_path, AgentName.ENGINEER)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    with caplog.at_level(logging.WARNING):
        await make_commit_hook(tasks)(_verdict(run, task))
    assert "couldn't commit" in caplog.text


def _elf(kind: int) -> bytes:
    """The start of a little-endian ELF file of type ``kind`` (4 is a core file)."""
    return b"\x7fELF\x02\x01\x01" + bytes(9) + kind.to_bytes(2, "little") + bytes(46)


async def test_a_core_dump_is_never_committed(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    (ws / "src" / "core").mkdir(parents=True)  # a package called core is code
    (ws / "src" / "core" / "__init__.py").write_text("")
    (ws / "core.py").write_text("x = 1\n")
    (ws / "core").write_bytes(_elf(4))
    (ws / "src" / "core.1234").write_bytes(_elf(4))
    (ws / "core.bin").write_bytes(_elf(2))  # an executable named core.* is kept
    await WorkspaceGit(ws).commit_version(V1, subject="s")
    assert sorted(git(ws, "ls-files")) == [
        ".gitignore",
        "core.bin",
        "core.py",
        "src/core/__init__.py",
    ]
    # One committed before is dropped from the next version.
    git(ws, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-f", "core")
    git(ws, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "oops")
    await WorkspaceGit(ws).commit_version(V2, subject="s")
    assert "core" not in git(ws, "ls-files")
    assert (ws / "core").is_file()  # left in place, just not committed
