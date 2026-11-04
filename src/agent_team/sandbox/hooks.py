"""Hooks that keep the sandbox clean: containers go when their task or run ends."""

from __future__ import annotations

from agent_team.core.interfaces import EventBus
from agent_team.core.models import Event, EventName
from agent_team.sandbox.base import CommandRunner


def attach_cleanup(bus: EventBus, runner: CommandRunner) -> None:
    """Remove a task's containers when it finishes, and a run's when it ends."""

    async def task_done(event: Event) -> None:
        """Remove what the finished task's commands left behind."""
        if event.task_id is not None:
            await runner.cleanup(task_id=event.task_id)

    async def run_done(event: Event) -> None:
        """Remove what the ended run's commands left behind."""
        await runner.cleanup(run_id=event.run_id)

    bus.on_after(EventName.TASK_FINISHED, task_done)
    bus.on_after(EventName.RUN_FINISHED, run_done)
