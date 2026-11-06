"""One terminal session with the team: start or resume a run, show it live, pass on
what the user types, and review the finished run.

While the team works, the live view (or plain lines) shows it. When the run waits on
the user, each line typed is sent as a message; ending the input (Ctrl-D) leaves the
run waiting. When the run is done, the session shows the summary and asks the user to
accept it or describe a change, and after a change re-runs it shows what changed.
A question asked at that prompt is answered and the prompt comes back.

The session holds the run's drive lock while the team works on the run, and lets go
while it waits for the user's next line. So ``agent-team stop`` knows whether a
session will see its flag at the next model call, or must mark the run stopped itself.
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Awaitable, Callable

from rich.console import Console

from agent_team.app import App
from agent_team.core.control import DriveLock
from agent_team.core.models import Event, EventName, RunStatus
from agent_team.orchestrator.replies import Reply
from agent_team.ui.diffs import Snapshot, change_report, snapshot
from agent_team.ui.summary import summarize
from agent_team.ui.view import RunView

ACCEPT = re.compile(
    r"^\s*(accept(ed)?|approve[ds]?|ok(ay)?|yes|y|lgtm|looks good|done)[\s.!]*$", re.I
)
REVIEW_PROMPT = "Type 'accept' to finish, or describe a change."

#: Reads one line of input; None at the end of input.
LineReader = Callable[[], Awaitable[str | None]]


async def read_stdin() -> str | None:
    """One line from stdin without blocking the event loop; None at end of input."""
    sys.stdout.write("> ")
    sys.stdout.flush()
    line = await asyncio.to_thread(sys.stdin.readline)
    return line.strip() if line else None


class Session:
    """Talks to the team for one run. See the module docstring."""

    def __init__(
        self,
        team: App,
        console: Console,
        *,
        plain: bool = False,
        read_line: LineReader = read_stdin,
    ) -> None:
        """Show the run on ``console`` and read the user's lines with ``read_line``."""
        self.team = team
        self.console = console
        self.read_line = read_line
        self.view = RunView(console, plain=plain)
        self.view.attach(team.bus)
        self.lock = DriveLock(team.runs_dir)
        team.bus.on_after(EventName.RUN_STARTED, self._on_run_started)
        self._seen: dict[str, Snapshot] = {}  # per run, the versions last reviewed

    async def _on_run_started(self, event: Event) -> None:
        """Hold the lock of a run this session just started."""
        self.lock.acquire(event.run_id)

    def say(self, text: str) -> None:
        """Print text as it is, without Rich markup."""
        self.console.print(text, markup=False, highlight=False)

    async def _drive(self, work: Awaitable[Reply]) -> Reply:
        """Await the team's work with the live view running."""
        self.view.start()
        try:
            return await work
        finally:
            self.view.stop()

    async def start(self, request: str, *, context_dirs: list[str] | None = None) -> int:
        """Start a run for ``request`` and talk until it is done or left waiting."""
        try:
            reply = await self._drive(
                self.team.orchestrator.handle(request, context_dirs=context_dirs)
            )
            return await self._talk(reply, reply.run_id)
        finally:
            self.lock.release_all()

    async def resume(self, run_id: str) -> int:
        """Resume a run after a stop, a crash or a question left open, and talk on."""
        try:
            await self.team.stores.tasks.get_run(run_id)
        except KeyError:
            self.say(f"No run {run_id} in {self.team.config.paths.db_file()}.")
            return 1
        if not self.lock.acquire(run_id):
            self.say(f"Run {run_id} is being driven by another session.")
            return 1
        try:
            self.team.kill_switch.clear(run_id)  # a stop is lifted by resuming
            self.view.load(run_id, await self.team.stores.tasks.list_tasks(run_id))
            reply = await self._drive(self.team.orchestrator.resume(run_id))
            return await self._talk(reply, run_id)
        finally:
            self.lock.release_all()

    def _show(self, reply: Reply) -> None:
        """Print a reply, with the run id and state."""
        self.say(reply.text)
        if reply.run_id:
            self.say(f"[run {reply.run_id}: {reply.kind}]")

    async def _talk(self, reply: Reply, run_id: str | None) -> int:
        """Show each reply and pass on the user's lines. Returns the exit code: 0 once
        the run is done and reviewed or left waiting, 1 when it stopped or failed with
        nothing to answer, or another session took it over."""
        while True:
            self._show(reply)
            run_id = reply.run_id or run_id
            if run_id:
                self.lock.release(run_id)  # nothing runs while the user types
            line: str | None
            if run_id and reply.kind == "done":
                line = await self._review(run_id)
                if line is None:
                    return 0
            elif run_id and reply.kind == "answer" and await self._is_done(run_id):
                line = await self._prompt_review(run_id)
                if line is None:
                    return 0
            elif reply.kind in ("stopped", "error") and not reply.pending:
                return 1
            else:
                line = await self._next_line()
                if line is None:
                    if run_id:
                        self.say(f"Run {run_id} is waiting. Continue: agent-team resume {run_id}")
                    return 0
            if run_id and not self.lock.acquire(run_id):
                self.say(f"Run {run_id} is being driven by another session.")
                return 1
            reply = await self._drive(self.team.orchestrator.handle(line, run_id=run_id))

    async def _is_done(self, run_id: str) -> bool:
        """Whether the run is finished, so the user is reviewing it."""
        run = await self.team.stores.tasks.get_run(run_id)
        return run.status is RunStatus.DONE

    async def _next_line(self) -> str | None:
        """The next non-empty line, or None at the end of input."""
        line = await self.read_line()
        while line == "":
            line = await self.read_line()
        return line

    async def _review(self, run_id: str) -> str | None:
        """Show the finished run, and what changed since the user last reviewed it; then
        ask to accept or change. Returns the change to send, or None when the user
        accepts or ends the input."""
        stores = self.team.stores
        summary = await summarize(run_id, stores.tasks, stores.artifacts, self.team.runs_dir)
        for text in summary.lines():
            self.say(text)
        seen = self._seen.get(run_id)
        if seen is not None:
            run = await stores.tasks.get_run(run_id)
            self.say("")
            self.say("What changed:")
            for text in await change_report(run, seen, stores.artifacts, stores.tasks):
                self.say(text)
        self._seen[run_id] = await snapshot(run_id, stores.tasks)
        return await self._prompt_review(run_id)

    async def _prompt_review(self, run_id: str) -> str | None:
        """Ask to accept or change. Returns the line to send, or None when the user
        accepts or ends the input."""
        self.say("")
        self.say(REVIEW_PROMPT)
        line = await self._next_line()
        if line is None:
            self.say(f"Run {run_id} is done.")
            return None
        if ACCEPT.match(line):
            self.say(f"Accepted. Run {run_id} is done.")
            return None
        return line
