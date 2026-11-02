"""File tools: read_file, list_dir, grep and write_file.

They raise ToolError on bad input; the agent loop turns that into an error tool result
so the model can adapt (docs/DESIGN.md, error table).
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any, ClassVar

from agent_team.core.errors import ToolError
from agent_team.core.interfaces import Tool, ToolContext
from agent_team.core.models import Run, ToolResult
from agent_team.llm.types import ToolSpec
from agent_team.tools.base import display_path, resolve_path

_SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"})


def _str_arg(args: dict[str, Any], key: str, tool: str, default: str | None = None) -> str:
    value = args.get(key, default)
    if not isinstance(value, str):
        raise ToolError(f"argument {key!r} must be a string", tool=tool)
    return value


def _int_arg(args: dict[str, Any], key: str, tool: str, default: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ToolError(f"argument {key!r} must be a non-negative integer", tool=tool)
    return value


def _allowed(path: Path, run: Run) -> bool:
    """A walked entry is listed or searched only if read_file could read it, so a
    symlink pointing outside the workspace is skipped, not followed."""
    try:
        resolve_path(str(path), run, write=False, tool="walk")
    except ToolError:
        return False
    return True


def _walk(root: Path, run: Run) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if _allowed(path, run):
            out.append(path)
    return out


class ReadFileTool:
    """Read a text file, or a range of its lines."""

    spec: ToolSpec = ToolSpec(
        name="read_file",
        description=(
            "Read a text file from the workspace or a --context folder. Returns numbered "
            "lines. Use offset and limit to read a range of a long file."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative or absolute"},
                "offset": {"type": "integer", "description": "First line, 1-based"},
                "limit": {"type": "integer", "description": "Max lines, default 400"},
            },
            "required": ["path"],
        },
    )
    safe_to_repeat: bool = True

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = self.spec.name
        path = resolve_path(_str_arg(args, "path", name), ctx.run, write=False, tool=name)
        offset = max(_int_arg(args, "offset", name, 1), 1)
        limit = _int_arg(args, "limit", name, 400) or 400
        if not path.is_file():
            raise ToolError(f"no such file: {args['path']}", tool=name)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ToolError(f"{args['path']} is not a UTF-8 text file", tool=name) from exc
        chunk = lines[offset - 1 : offset - 1 + limit]
        body = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start=offset))
        shown = display_path(path, ctx.run)
        more = len(lines) - (offset - 1 + len(chunk))
        footer = f"\n[{more} more lines; read again with offset]" if more > 0 else ""
        return ToolResult(
            content=f"{shown} ({len(lines)} lines)\n{body}{footer}",
            data={"path": shown, "lines": len(lines), "offset": offset, "shown": len(chunk)},
        )


class ListDirTool:
    """List files under a folder, skipping anything outside the allowed folders."""

    spec: ToolSpec = ToolSpec(
        name="list_dir",
        description="List files under a folder in the workspace or a --context folder.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Folder, default the workspace"},
                "recursive": {"type": "boolean", "description": "Default true"},
            },
        },
    )
    safe_to_repeat: bool = True
    max_entries: ClassVar[int] = 500

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = self.spec.name
        root = resolve_path(_str_arg(args, "path", name, "."), ctx.run, write=False, tool=name)
        if not root.is_dir():
            raise ToolError(f"no such folder: {args.get('path', '.')}", tool=name)
        recursive = args.get("recursive", True) is not False
        entries = (
            _walk(root, ctx.run)
            if recursive
            else [p for p in sorted(root.iterdir()) if _allowed(p, ctx.run)]
        )
        names = [
            p.relative_to(root).as_posix() + ("/" if p.is_dir() else "")
            for p in entries
            if p.name not in _SKIP_DIRS
        ]
        cut = names[: self.max_entries]
        extra = len(names) - len(cut)
        footer = f"\n[{extra} more entries not shown]" if extra else ""
        shown = display_path(root, ctx.run)
        return ToolResult(
            content=f"{shown}/\n" + "\n".join(cut) + footer if cut else f"{shown}/ is empty",
            data={"path": shown, "count": len(names)},
        )


class GrepTool:
    """Search files for a regular expression."""

    spec: ToolSpec = ToolSpec(
        name="grep",
        description=(
            "Search files for a regular expression. Returns path:line: text for each match."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Folder or file, default workspace"},
                "glob": {"type": "string", "description": "Filename filter such as *.py"},
            },
            "required": ["pattern"],
        },
    )
    safe_to_repeat: bool = True
    max_matches: ClassVar[int] = 200

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = self.spec.name
        try:
            regex = re.compile(_str_arg(args, "pattern", name))
        except re.error as exc:
            raise ToolError(f"bad pattern: {exc}", tool=name) from exc
        root = resolve_path(_str_arg(args, "path", name, "."), ctx.run, write=False, tool=name)
        glob = args.get("glob")
        files = [root] if root.is_file() else [p for p in _walk(root, ctx.run) if p.is_file()]
        matches: list[str] = []
        for file in files:
            if isinstance(glob, str) and not fnmatch.fnmatch(file.name, glob):
                continue
            try:
                text = file.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{display_path(file, ctx.run)}:{lineno}: {line.strip()}")
                    if len(matches) >= self.max_matches:
                        break
            if len(matches) >= self.max_matches:
                break
        if not matches:
            return ToolResult(content="no matches", data={"count": 0})
        return ToolResult(content="\n".join(matches), data={"count": len(matches)})


class WriteFileTool:
    """Create or replace a file in the workspace."""

    spec: ToolSpec = ToolSpec(
        name="write_file",
        description=(
            "Create or overwrite a file in the workspace with the full new content. "
            "Paths are workspace-relative; --context folders are read-only."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    )
    safe_to_repeat: bool = False

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = self.spec.name
        path = resolve_path(_str_arg(args, "path", name), ctx.run, write=True, tool=name)
        content = _str_arg(args, "content", name)
        if path.is_dir():
            raise ToolError(f"{args['path']} is a folder", tool=name)
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        path.write_text(content, encoding="utf-8")
        shown = display_path(path, ctx.run)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        verb = "updated" if existed else "created"
        return ToolResult(
            content=f"{verb} {shown} ({lines} lines)",
            data={"path": shown, "lines": lines, "created": not existed},
        )


def file_tools() -> dict[str, Tool]:
    """The four file tools, keyed by name."""
    tools: list[Tool] = [ReadFileTool(), ListDirTool(), GrepTool(), WriteFileTool()]
    return {t.spec.name: t for t in tools}
