from __future__ import annotations

from pathlib import Path

import pytest

from agent_team.core.errors import ToolError
from agent_team.core.interfaces import ToolContext
from agent_team.core.models import AgentName, Phase, Run, Task
from agent_team.tools import file_tools
from agents.fakes import FakeBus


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
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
    return ToolContext(run=run, task=task, bus=FakeBus())


async def test_write_then_read_list_grep(ctx: ToolContext) -> None:
    tools = file_tools()
    out = await tools["write_file"]({"path": "src/app.py", "content": "a = 1\nb = 2\n"}, ctx)
    assert "created src/app.py (2 lines)" in out.content
    read = await tools["read_file"]({"path": "src/app.py", "offset": 2, "limit": 1}, ctx)
    assert "    2  b = 2" in read.content and "a = 1" not in read.content
    listing = await tools["list_dir"]({}, ctx)
    assert "src/app.py" in listing.content
    found = await tools["grep"]({"pattern": r"b = \d", "glob": "*.py"}, ctx)
    assert found.content == "src/app.py:2: b = 2"


async def test_context_folder_is_read_only(ctx: ToolContext) -> None:
    tools = file_tools()
    notes = str(Path(ctx.run.context_dirs[0]) / "notes.md")
    read = await tools["read_file"]({"path": notes}, ctx)
    assert "use SQLite" in read.content
    with pytest.raises(ToolError, match="read-only"):
        await tools["write_file"]({"path": notes, "content": "x"}, ctx)


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../x"])
async def test_paths_outside_workspace_are_rejected(ctx: ToolContext, path: str) -> None:
    with pytest.raises(ToolError, match="outside the workspace"):
        await file_tools()["read_file"]({"path": path}, ctx)


async def test_symlink_escape_is_rejected(ctx: ToolContext, tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("key")
    (Path(ctx.run.workspace) / "link").symlink_to(secret)
    with pytest.raises(ToolError, match="outside"):
        await file_tools()["read_file"]({"path": "link"}, ctx)


async def test_git_folder_is_not_writable(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match=r"\.git"):
        await file_tools()["write_file"]({"path": ".git/config", "content": "x"}, ctx)


async def test_missing_file_and_bad_args(ctx: ToolContext) -> None:
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


async def test_stored_tool_results_are_readable_not_writable(ctx: ToolContext) -> None:
    from agent_team.tools.base import store_large

    path = store_large("x" * 20, ctx.run, label="grep")
    tools = file_tools()
    assert "xxxx" in (await tools["read_file"]({"path": path}, ctx)).content
    with pytest.raises(ToolError, match="read-only"):
        await tools["write_file"]({"path": path, "content": "y"}, ctx)


async def test_walks_skip_symlinks_that_escape(ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("token = 1\n")
    ws = Path(ctx.run.workspace)
    (ws / "ok.py").write_text("token = 2\n")
    (ws / "linked").symlink_to(outside)
    tools = file_tools()
    found = await tools["grep"]({"pattern": "token"}, ctx)
    assert found.content == "ok.py:1: token = 2"
    for recursive in (True, False):
        listing = await tools["list_dir"]({"recursive": recursive}, ctx)
        assert "secret.py" not in listing.content and "linked" not in listing.content


@pytest.mark.parametrize(
    "name", [".env", ".env.local", ".ENV", ".Env.Production", "sub/.env", ".envrc", "prod.env"]
)
async def test_env_files_in_context_folders_are_refused(ctx: ToolContext, name: str) -> None:
    """Secrets files in a --context folder can't be read, listed or searched."""
    ref = Path(ctx.run.context_dirs[0])
    secret = ref / name
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("ANTHROPIC_API_KEY=sk-secret\n")
    tools = file_tools()
    with pytest.raises(ToolError, match="secrets file"):
        await tools["read_file"]({"path": str(secret)}, ctx)
    with pytest.raises(ToolError, match="secrets file"):
        await tools["grep"]({"pattern": "sk-", "path": str(secret)}, ctx)
    listing = await tools["list_dir"]({"path": str(ref)}, ctx)
    assert Path(name).name not in listing.content and "notes.md" in listing.content
    found = await tools["grep"]({"pattern": "sk-secret", "path": str(ref)}, ctx)
    assert found.content == "no matches"


@pytest.mark.parametrize("name", [".env.example", ".env.sample", ".env.template", "env.md"])
async def test_env_templates_in_a_context_folder_are_readable(ctx: ToolContext, name: str) -> None:
    """Templates such as ``.env.example`` hold no secrets, so they stay readable."""
    example = Path(ctx.run.context_dirs[0]) / name
    example.write_text("ANTHROPIC_API_KEY=\n")
    read = await file_tools()["read_file"]({"path": str(example)}, ctx)
    assert "ANTHROPIC_API_KEY=" in read.content


async def test_writes_to_a_context_env_file_say_read_only(ctx: ToolContext) -> None:
    """A write is refused for being in a --context folder, the reason that applies."""
    path = str(Path(ctx.run.context_dirs[0]) / ".env")
    with pytest.raises(ToolError, match="read-only"):
        await file_tools()["write_file"]({"path": path, "content": "x"}, ctx)


async def test_symlinks_to_or_named_env_are_refused(ctx: ToolContext) -> None:
    """A link to a context .env, or a link called .env, is refused like the file."""
    ref = Path(ctx.run.context_dirs[0])
    (ref / ".env").write_text("ANTHROPIC_API_KEY=sk-secret\n")
    (ref / "harmless.txt").symlink_to(ref / ".env")
    (Path(ctx.run.workspace) / "link").symlink_to(ref / ".env")
    (ref / "keys").mkdir()
    (ref / "keys" / ".env").symlink_to(ref / "notes.md")
    (ref / "conf").mkdir()
    (ref / "conf" / "keys.txt").write_text("sk-secret\n")
    (ref / ".env.secrets").symlink_to(ref / "conf")
    tools = file_tools()
    paths = (
        str(ref / "harmless.txt"),
        "link",
        str(ref / "keys" / ".env"),
        str(ref / ".env.secrets" / "keys.txt"),
    )
    for path in paths:
        with pytest.raises(ToolError, match="secrets file"):
            await tools["read_file"]({"path": path}, ctx)


async def test_workspace_env_files_stay_usable(ctx: ToolContext) -> None:
    """The rule covers --context folders only; the agents' own workspace keeps .env."""
    tools = file_tools()
    await tools["write_file"]({"path": ".env", "content": "DEBUG=1\n"}, ctx)
    assert "DEBUG=1" in (await tools["read_file"]({"path": ".env"}, ctx)).content
