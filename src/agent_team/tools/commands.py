"""run_command and install_deps: the Engineer's and QA's way to run code.

Both run in the sandbox. run_command has the network off. install_deps is the one
step with the network on, so it takes only plain requirement specifiers, never pip
options, URLs or paths, and the pre-check can read every package it installs (S4).

Each call's ``data`` carries ``argv`` and ``exit_code``, which the auditor reads from
the tool log: R6 matches the reported test command against ``argv`` and compares the
exit codes. A command killed by SIGILL (exit 132) or SIGSEGV (exit 139) gets a line
saying so in plain words, with the likely cause and what to try.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from agent_team.core.config import SandboxConfig
from agent_team.core.errors import SandboxError, ToolError
from agent_team.core.interfaces import Tool, ToolContext
from agent_team.core.models import ToolResult
from agent_team.llm.types import ToolSpec
from agent_team.sandbox.base import (
    CONTAINER_WORKSPACE,
    DEPS_DIR,
    CommandResult,
    CommandRunner,
    CommandSpec,
)
from agent_team.sandbox.git import WorkspaceGit
from agent_team.tools.base import display_path, resolve_path

#: A requirement specifier pip can install from the index: a name, optional extras
#: and optional version clauses. No options, URLs, paths or environment markers.
_VERSION = r"(?:===?|[<>!~]=|[<>])\s*[A-Za-z0-9.*+!_-]+"
_REQUIREMENT = re.compile(
    rf"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?:\[[A-Za-z0-9._,\s-]+\])?"
    rf"(?:\s*{_VERSION}(?:\s*,\s*{_VERSION})*)?$"
)
MAX_PACKAGES = 100
#: Names pip would read as a local archive rather than a package on the index.
_ARCHIVE = re.compile(r"\.(?:whl|zip|tar|tgz|tbz|txz|gz|bz2|xz|egg)$", re.IGNORECASE)

log = logging.getLogger(__name__)

#: What an exit code from a crash signal means, and what to try, keyed by exit code.
SIGNAL_HINTS: dict[int, tuple[str, str]] = {
    132: (
        "SIGILL",
        "the process ran a CPU instruction this machine doesn't offer. That almost "
        "always means a compiled package (a binary wheel) was built for a newer CPU "
        "baseline than the sandbox exposes, as in Linux VMs on Apple Silicon, where "
        "cryptography 47 and later crash on import. Find the package whose import "
        "crashes and pin an older release, such as cryptography<47 (46.0.7).",
    ),
    139: (
        "SIGSEGV",
        "compiled code crashed (a segmentation fault). If it happens on import, a "
        "binary wheel probably doesn't match the sandbox's CPU or C library: find the "
        "package whose import crashes and pin an older release.",
    ),
}


def signal_hint(exit_code: int) -> str | None:
    """A plain-words line for an exit code that means a crash signal, else None."""
    hint = SIGNAL_HINTS.get(exit_code)
    if hint is None:
        return None
    name, meaning = hint
    return f"note: exit {exit_code} means the process was killed by {name}: {meaning}"


def _workspace_cwd(raw: Any, ctx: ToolContext, tool: str) -> str:
    """The workspace-relative folder to run in; must be a folder inside the workspace."""
    if raw is None:
        return "."
    if not isinstance(raw, str):
        raise ToolError("argument 'cwd' must be a string", tool=tool)
    path = resolve_path(raw, ctx.run, write=True, tool=tool)
    if not path.is_dir():
        raise ToolError(f"cwd {raw!r} is not a folder in the workspace", tool=tool)
    return display_path(path, ctx.run)


def _argv(raw: Any, tool: str) -> list[str]:
    """A non-empty list of strings, or a ToolError the model can read."""
    if isinstance(raw, str):
        raise ToolError('argv must be a list of strings, such as ["pytest", "-q"]', tool=tool)
    if not isinstance(raw, list) or not raw or not all(isinstance(a, str) for a in raw):
        raise ToolError("argv must be a non-empty list of strings", tool=tool)
    if any("\x00" in a for a in raw) or not raw[0].strip():
        raise ToolError("argv has an empty program or a NUL byte", tool=tool)
    return list(raw)


async def prepare_workspace(workspace: str, tool: str) -> None:
    """Create the workspace's git repository and .gitignore before any command runs,
    so the container never sees a writable .git. Without git there is nothing on the
    host to protect, so the command still runs; a .git the harness didn't make stops
    it."""
    try:
        await WorkspaceGit(workspace).ensure_repo()
    except SandboxError as exc:
        if "isn't installed" not in str(exc):
            raise ToolError(str(exc), tool=tool) from exc
        log.warning("git isn't installed; versions won't be committed")


async def _run(runner: CommandRunner, spec: CommandSpec, tool: str) -> CommandResult:
    """Run one command; a sandbox failure becomes an error the model can read and retry."""
    try:
        return await runner.run(spec)
    except SandboxError as exc:
        raise ToolError(f"the sandbox failed: {exc}", tool=tool) from exc


def _render(argv: list[str], result: CommandResult, *, label: str = "$") -> str:
    """The command's result as the model reads it."""
    status = (
        f"timed out after {result.duration_s}s (exit {result.exit_code})"
        if result.timed_out
        else f"exit {result.exit_code} in {result.duration_s}s"
    )
    parts = [f"{label} {' '.join(argv)}", status]
    hint = None if result.timed_out else signal_hint(result.exit_code)
    if hint:
        parts.append(hint)
    if result.stdout.strip():
        parts += ["--- stdout ---", result.stdout.rstrip()]
    if result.stderr.strip():
        parts += ["--- stderr ---", result.stderr.rstrip()]
    return "\n".join(parts)


def _data(argv: list[str], result: CommandResult, *, network: bool) -> dict[str, Any]:
    """What the trace and the auditor read from one command."""
    return {
        "argv": argv,
        "exit_code": result.exit_code,
        "duration_s": result.duration_s,
        "timed_out": result.timed_out,
        "network": network,
        "signal": SIGNAL_HINTS[result.exit_code][0]
        if result.exit_code in SIGNAL_HINTS and not result.timed_out
        else None,
    }


class RunCommandTool:
    """Run a command in the sandbox with the network off."""

    spec: ToolSpec = ToolSpec(
        name="run_command",
        description=(
            "Run a command in the Docker sandbox, in the workspace, with no network. "
            'Pass argv as a list, such as ["python", "-m", "pytest", "-q"]; use '
            '["sh", "-c", "..."] for pipes or several commands. Install packages '
            "first with install_deps. Returns the exit code and the end of stdout and "
            "stderr."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "argv": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string", "description": "Workspace folder, default the root"},
                "timeout_s": {"type": "number", "description": "Seconds before it is killed"},
            },
            "required": ["argv"],
        },
    )
    safe_to_repeat: bool = False

    def __init__(self, runner: CommandRunner, config: SandboxConfig) -> None:
        """Run commands through ``runner`` within ``config``'s limits."""
        self.runner = runner
        self.config = config

    def _timeout(self, raw: Any) -> float:
        """The requested timeout, capped at the config's longest; the default if unset."""
        if raw is None:
            return self.config.timeout_s
        if isinstance(raw, bool) or not isinstance(raw, int | float) or raw <= 0:
            raise ToolError("timeout_s must be a positive number", tool=self.spec.name)
        return min(float(raw), self.config.max_timeout_s)

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Run argv in the sandbox and report its exit code and output."""
        name = self.spec.name
        argv = _argv(args.get("argv"), name)
        cwd = _workspace_cwd(args.get("cwd"), ctx, name)
        timeout = self._timeout(args.get("timeout_s"))
        await prepare_workspace(ctx.run.workspace, name)
        result = await _run(
            self.runner,
            CommandSpec(
                argv=argv,
                workspace=ctx.run.workspace,
                run_id=ctx.run.id,
                task_id=ctx.task.id,
                timeout_s=timeout,
                cwd=cwd,
            ),
            name,
        )
        return ToolResult(
            content=_render(argv, result),
            is_error=result.exit_code != 0,
            data=_data(argv, result, network=False),
        )


def requirement_lines(text: str) -> list[str]:
    """The requirements in a requirements file, without comments and blank lines."""
    out: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", line).strip()
        if line:
            out.append(line)
    return out


def read_requirements(raw: Any, ctx: ToolContext, tool: str) -> tuple[str, list[str]]:
    """The workspace-relative path and requirement lines of a requirements file."""
    if not isinstance(raw, str):
        raise ToolError("argument 'requirements' must be a path", tool=tool)
    path = resolve_path(raw, ctx.run, write=True, tool=tool)  # workspace files only
    if not path.is_file():
        raise ToolError(f"no such requirements file: {raw}", tool=tool)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolError(f"can't read {raw}: {exc}", tool=tool) from exc
    return display_path(path, ctx.run), requirement_lines(text)


def _name(spec: str) -> str:
    """The package name at the start of a requirement specifier."""
    m = re.match(r"[A-Za-z0-9._-]+", spec)
    return m.group(0) if m else spec


def check_specifiers(specs: list[str], tool: str) -> None:
    """Refuse anything that isn't a plain requirement from the index."""
    bad = [s for s in specs if not _REQUIREMENT.match(s) or _ARCHIVE.search(_name(s))]
    if bad:
        raise ToolError(
            "install_deps takes only package names with optional extras and versions, "
            f"such as 'fastapi==0.115.0'; not options, URLs, paths or markers: {bad[:5]}",
            tool=tool,
        )
    if len(specs) > MAX_PACKAGES:
        raise ToolError(f"at most {MAX_PACKAGES} packages per call", tool=tool)


class InstallDepsTool:
    """Install Python packages into the workspace's deps folder, with the network on."""

    spec: ToolSpec = ToolSpec(
        name="install_deps",
        description=(
            "Install Python packages from PyPI so run_command can use them, pytest "
            "included. This is the only step with network access. Give packages as "
            'requirement specifiers, pinned where you can ("fastapi==0.115.0"), or a '
            "requirements file in the workspace. Packages are checked before install."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "packages": {"type": "array", "items": {"type": "string"}},
                "requirements": {"type": "string", "description": "e.g. requirements.txt"},
            },
        },
    )
    safe_to_repeat: bool = False

    def __init__(self, runner: CommandRunner, config: SandboxConfig) -> None:
        """Install through ``runner`` within ``config``'s install timeout."""
        self.runner = runner
        self.config = config

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Check the specifiers, then pip-install them into the deps folder."""
        name = self.spec.name
        packages = args.get("packages") or []
        if not isinstance(packages, list) or not all(isinstance(p, str) for p in packages):
            raise ToolError("packages must be a list of strings", tool=name)
        specs = [p.strip() for p in packages if p.strip()]
        source = "packages"
        if args.get("requirements") is not None:
            source, lines = read_requirements(args["requirements"], ctx, name)
            specs += lines
        if not specs:
            raise ToolError("give packages or a requirements file with at least one", tool=name)
        check_specifiers(specs, name)
        await prepare_workspace(ctx.run.workspace, name)
        await asyncio.to_thread(Path(ctx.run.workspace, DEPS_DIR).mkdir, exist_ok=True)
        # -I: pip must not import anything from the workspace or the deps folder, since
        # this is the one step with the network on. It runs outside the workspace too,
        # so a bare name can't resolve to a local folder.
        argv = [
            "python",
            "-I",
            "-m",
            "pip",
            "install",
            "--no-input",
            "--upgrade",
            "--target",
            f"{CONTAINER_WORKSPACE}/{DEPS_DIR}",
            *specs,
        ]
        result = await _run(
            self.runner,
            CommandSpec(
                argv=argv,
                workspace=ctx.run.workspace,
                run_id=ctx.run.id,
                task_id=ctx.task.id,
                timeout_s=self.config.install_timeout_s,
                cwd="/tmp",
                network=True,
            ),
            name,
        )
        data = _data(argv, result, network=True)
        data.update(packages=specs, source=source)
        return ToolResult(
            content=_render(argv, result, label="installing:"),
            is_error=result.exit_code != 0,
            data=data,
        )


def command_tools(runner: CommandRunner, config: SandboxConfig) -> dict[str, Tool]:
    """run_command and install_deps over one runner, keyed by name."""
    tools: list[Tool] = [RunCommandTool(runner, config), InstallDepsTool(runner, config)]
    return {t.spec.name: t for t in tools}
