"""The drive lock: one session per run, seen from other sessions and processes, and
gone when the holder ends."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from agent_team.core.control import DriveLock

HOLD = """
import sys, time
from agent_team.core.control import DriveLock
lock = DriveLock(sys.argv[1])
assert lock.acquire("run_1")
print("held", flush=True)
sys.stdin.readline()
"""


def test_a_run_has_one_driver_at_a_time(tmp_path: Path) -> None:
    first, second = DriveLock(tmp_path), DriveLock(tmp_path)
    assert not first.is_driven("run_1")
    assert first.acquire("run_1") and first.acquire("run_1")
    assert first.is_driven("run_1") and second.is_driven("run_1")
    assert not second.acquire("run_1")
    first.release_all()
    assert not second.is_driven("run_1")
    assert second.acquire("run_1")
    second.release("run_1")
    second.release("run_1")  # releasing twice is harmless


def test_the_lock_of_another_process_is_seen_and_dropped_when_it_ends(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        lock = DriveLock(tmp_path)
        assert lock.is_driven("run_1") and not lock.acquire("run_1")
        assert (tmp_path / "run_1" / "DRIVING").read_text().strip() == str(holder.pid)
    finally:
        holder.kill()  # no clean release: the OS drops the lock
        holder.wait()
    assert not DriveLock(tmp_path).is_driven("run_1")
