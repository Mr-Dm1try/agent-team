"""A command runner that runs nothing: it records each command and answers from a script."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agent_team.sandbox import CommandResult, CommandSpec

PASSED = CommandResult(exit_code=0, stdout="1 passed in 0.01s\n", stderr="", duration_s=0.01)


@dataclass
class FakeRunner:
    """Implements ``CommandRunner`` without Docker. ``answer`` maps a command to its
    result; by default every command passes."""

    answer: Callable[[CommandSpec], CommandResult] = lambda spec: PASSED
    specs: list[CommandSpec] = field(default_factory=list)
    cleaned: list[dict[str, str | None]] = field(default_factory=list)
    sweeps: int = 0
    closed: bool = False

    async def run(self, spec: CommandSpec) -> CommandResult:
        """Record the command and return the scripted result."""
        self.specs.append(spec)
        return self.answer(spec)

    async def cleanup(self, *, run_id: str | None = None, task_id: str | None = None) -> int:
        """Record what would be cleaned up."""
        self.cleaned.append({"run_id": run_id, "task_id": task_id})
        return 0

    async def sweep(self) -> int:
        """Count the startup sweeps."""
        self.sweeps += 1
        return 0

    def close(self) -> None:
        """Note that the app closed the runner."""
        self.closed = True
