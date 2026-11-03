"""Kill switch. ``agent-team stop <run_id>`` drops a STOP file in the run folder, so
it works from another process. The gateway checks it before every call and raises
``RunStopped``; the orchestrator catches that at a safe point and saves state."""

from __future__ import annotations

from pathlib import Path

from agent_team.core.errors import RunStopped


class KillSwitch:
    """Per-run stop flag stored as a file, shared between processes."""

    def __init__(self, runs_dir: str | Path) -> None:
        """Keep the runs folder that holds each run's STOP file."""
        self.runs_dir = Path(runs_dir)

    def _flag(self, run_id: str) -> Path:
        """Path of the run's STOP file."""
        return self.runs_dir / run_id / "STOP"

    def stop(self, run_id: str, reason: str = "stopped by user") -> None:
        """Ask the run to stop at its next model call."""
        flag = self._flag(run_id)
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(reason, encoding="utf-8")

    def clear(self, run_id: str) -> None:
        """Remove the stop flag, e.g. before resuming."""
        self._flag(run_id).unlink(missing_ok=True)

    def is_stopped(self, run_id: str) -> bool:
        """Whether the run has a stop flag."""
        return self._flag(run_id).exists()

    def check(self, run_id: str) -> None:
        """Raise RunStopped if the run has a stop flag."""
        flag = self._flag(run_id)
        if flag.exists():
            reason = flag.read_text(encoding="utf-8").strip() or "stopped"
            raise RunStopped(f"run {run_id} stopped: {reason}")
