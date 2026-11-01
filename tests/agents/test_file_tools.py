from __future__ import annotations

from pathlib import Path

import pytest

from agent_team.core.errors import ToolError
from agent_team.core.models import AgentName, Phase, Run, Task
from agent_team.tools import RunToolContext, file_tools
from agents.fakes import FakeBus


@pytest.fixture
def ctx(tmp_path: Path) -> RunToolContext:
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True)
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "notes.md").write_text("# Notes\nuse SQLite\n")
    run = Run(title="t", request="r", workspace=str(ws), context_dirs=[str(ref)])
    task = Task(
        run_id=run.id,
        phase=Phase.IMPLEMENTATION,
        agent=AgentName.ENGINEER,
        objective="o",
        output_kind="code_change",
    )
    return RunToolContext(run=run, task=task, bus=FakeBus())


async def test_write_then_read_list_grep(ctx: RunToolContext) -> None:
    tools = file_tools()
    out = await tools["write_file"]({"path": "src/app.py", "content": "a = 1\nb = 2\n"}, ctx)
    assert "created src/app.py (2 lines)" in out.content
    read = await tools["read_file"]({"path": "src/app.py", "offset": 2, "limit": 1}, ctx)
    assert "    2  b = 2" in read.content and "a = 1" not in read.content
    listing = await tools["list_dir"]({}, ctx)
    assert "src/app.py" in listing.content
    found = await tools["grep"]({"pattern": r"b = \d", "glob": "*.py"}, ctx)
    assert found.content == "src/app.py:2: b = 2"


async def test_context_folder_is_read_only(ctx: RunToolContext) -> None:
    tools = file_tools()
    notes = str(Path(ctx.run.context_dirs[0]) / "notes.md")
    read = await tools["read_file"]({"path": notes}, ctx)
    assert "use SQLite" in read.content
    with pytest.raises(ToolError, match="read-only"):
        await tools["write_file"]({"path": notes, "content": "x"}, ctx)


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../x"])
async def test_paths_outside_workspace_are_rejected(ctx: RunToolContext, path: str) -> None:
    with pytest.raises(ToolError, match="outside the workspace"):
        await file_tools()["read_file"]({"path": path}, ctx)


async def test_symlink_escape_is_rejected(ctx: RunToolContext, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("key")
    (Path(ctx.run.workspace) / "link").symlink_to(secret)
    with pytest.raises(ToolError, match="outside"):
        await file_tools()["read_file"]({"path": "link"}, ctx)


async def test_git_folder_is_not_writable(ctx: RunToolContext) -> None:
    with pytest.raises(ToolError, match=r"\.git"):
        await file_tools()["write_file"]({"path": ".git/config", "content": "x"}, ctx)


async def test_missing_file_and_bad_args(ctx: RunToolContext) -> None:
    tools = file_tools()
    with pytest.raises(ToolError, match="no such file"):
        await tools["read_file"]({"path": "nope.py"}, ctx)
    with pytest.raises(ToolError, match="must be a string"):
        await tools["write_file"]({"path": "a.py", "content": 3}, ctx)
    with pytest.raises(ToolError, match="bad pattern"):
        await tools["grep"]({"pattern": "("}, ctx)


def test_safe_to_repeat_flags() -> None:
    tools = file_tools()
    assert tools["read_file"].safe_to_repeat and tools["grep"].safe_to_repeat
    assert not tools["write_file"].safe_to_repeat
