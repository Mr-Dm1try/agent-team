"""Shared pieces for tools: workspace path rules and large-result
storage.

Path rules (docs/DESIGN.md, "Sandbox"): every path resolves inside the run's
workspace, except folders passed with ``--context``, which tools may read but never
write, and large tool results saved under ``runs/<id>/tool_results``, which are
read-only too. Symlinks and ``..`` are resolved before the check, so neither can escape.
"""

from __future__ import annotations

from pathlib import Path

from agent_team.core.errors import ToolError
from agent_team.core.models import Run, new_id

#: Tool output longer than this is saved to a file and only a preview is inlined.
MAX_INLINE_CHARS = 12_000


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_path(raw: str, run: Run, *, write: bool, tool: str) -> Path:
    """Resolve ``raw`` against the run's workspace and enforce the path rules."""
    if not raw or "\x00" in raw:
        raise ToolError("path must be a non-empty string", tool=tool)
    workspace = Path(run.workspace).resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()

    if _inside(resolved, workspace):
        if write and ".git" in resolved.relative_to(workspace).parts:
            raise ToolError("writing inside .git is not allowed", tool=tool)
        return resolved
    # Large tool results saved by the agent loop can be read back, never written.
    if _inside(resolved, results_dir(run).resolve()):
        if write:
            raise ToolError(f"{raw} is a saved tool result and is read-only", tool=tool)
        return resolved
    for ctx_dir in run.context_dirs:
        if _inside(resolved, Path(ctx_dir).resolve()):
            if write:
                raise ToolError(
                    f"{raw} is in a read-only --context folder; write inside the workspace",
                    tool=tool,
                )
            return resolved
    raise ToolError(
        f"{raw} is outside the workspace and the --context folders",
        tool=tool,
    )


def display_path(path: Path, run: Run) -> str:
    """Workspace-relative where possible, so logs and artifacts stay portable."""
    workspace = Path(run.workspace).resolve()
    if _inside(path, workspace):
        rel = path.relative_to(workspace).as_posix()
        return rel or "."
    return str(path)


def results_dir(run: Run) -> Path:
    """runs/<run_id>/tool_results, next to the workspace."""
    return Path(run.workspace).resolve().parent / "tool_results"


def store_large(content: str, run: Run, *, label: str) -> str:
    """Save a large tool result to a file in the run folder and return its path."""
    folder = results_dir(run)
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    path = folder / f"{new_id(safe or 'result')}.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)
