"""Kill switch and drive lock.

``agent-team stop <run_id>`` drops a STOP file in the run folder, so it works from
another process. The gateway checks it before every call and raises ``RunStopped``;
the orchestrator catches that at a safe point and saves state.

The session driving a run holds an exclusive lock on the run's DRIVING file. The
operating system drops the lock when the process ends, crash included, so ``stop``
can tell for certain whether a session will see the flag or the run is idle.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path
from typing import IO

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


ACQUIRE_TRIES = 5
ACQUIRE_PAUSE_S = 0.01


class DriveLock:
    """The lock a session holds on each run it drives; released when the process ends."""

    def __init__(self, runs_dir: str | Path) -> None:
        """Keep the runs folder that holds each run's DRIVING file."""
        self.runs_dir = Path(runs_dir)
        self._held: dict[str, IO[str]] = {}

    def _path(self, run_id: str) -> Path:
        """Path of the run's DRIVING file."""
        return self.runs_dir / run_id / "DRIVING"

    def acquire(self, run_id: str) -> bool:
        """Take the run's lock; False if another session holds it. A few short tries,
        since ``is_driven`` in another process takes the lock for a moment to probe it."""
        if run_id in self._held:
            return True
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+", encoding="utf-8")
        for attempt in range(ACQUIRE_TRIES):
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if attempt == ACQUIRE_TRIES - 1:
                    handle.close()
                    return False
                time.sleep(ACQUIRE_PAUSE_S)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._held[run_id] = handle
        return True

    def release(self, run_id: str) -> None:
        """Give the run's lock back, if this session holds it."""
        handle = self._held.pop(run_id, None)
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()

    def release_all(self) -> None:
        """Give back every lock this session holds."""
        for run_id in list(self._held):
            self.release(run_id)

    def is_driven(self, run_id: str) -> bool:
        """Whether some session, this one or another, holds the run's lock."""
        if run_id in self._held:
            return True
        path = self._path(run_id)
        if not path.exists():
            return False
        with path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(handle, fcntl.LOCK_UN)
        return False
