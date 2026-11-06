"""One terminal session with the team: start or resume a run, show it live, pass on
what the user types, and review the finished run.

While the team works, the live view (or plain lines) shows it, and each line typed is
read at once. "stop" stops the run at its next model call. Any other line goes to the
message loop: a question about the run is answered under the view, and a line that
changes the run (an answer, a change, new work) is held and sent again as soon as the
team stops. When the run waits on the user, each line typed is sent as a message;
ending the input (Ctrl-D) leaves the run waiting. When the run is done, the session
shows the summary and asks the user to accept it or describe a change, and after a
change re-runs it shows what changed. A question asked at that prompt is answered and
the prompt comes back.

The session holds the run's drive lock while the team works on the run, and lets go
while it waits for the user's next line. So ``agent-team stop`` knows whether a
session will see its flag at the next model call, or must mark the run stopped itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sys
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console

from agent_team.app import App
from agent_team.core.control import DriveLock
from agent_team.core.models import Event, EventName, RunStatus
from agent_team.orchestrator.messages import says_stop
from agent_team.orchestrator.replies import Reply
from agent_team.ui.diffs import Snapshot, change_report, snapshot
from agent_team.ui.summary import summarize
from agent_team.ui.view import RunView

log = logging.getLogger(__name__)

ACCEPT = re.compile(
    r"^\s*(accept(ed)?|approve[ds]?|ok(ay)?|yes|y|lgtm|looks good|done)[\s.!]*$", re.I
)
REVIEW_PROMPT = "Type 'accept' to finish, or describe a change."
#: The status line's word for each run status, as the user would say it.
STATE_WORDS = {RunStatus.WAITING: "waiting for you"}

#: The hint under the live view while the team works. It ends in ": " because what the
#: user types appears right after it.
TYPE_HINT = "Type a question or a change, or stop to stop the run: "
#: What the session says when the user types stop while the team works.
STOPPING = "Stopping the run at its next model call."
#: The reason logged for that stop.
STOP_REASON = "the user typed stop while the team worked"

#: Reads one line of input; None at the end of input.
LineReader = Callable[[], Awaitable[str | None]]


class StdinLines:
    """Reads stdin on a daemon thread, so a line typed while the team works is read at
    once, and a read still waiting when the session ends doesn't keep the process
    alive (a thread from ``asyncio.to_thread`` would)."""

    def __init__(self) -> None:
        """The thread starts on the first read, on the running loop."""
        self._lines: asyncio.Queue[str | None] | None = None

    async def __call__(self) -> str | None:
        """The next line, stripped; None at the end of input."""
        if self._lines is None:
            self._lines = asyncio.Queue()
            loop = asyncio.get_running_loop()
            threading.Thread(target=self._pump, args=(loop, self._lines), daemon=True).start()
        return await self._lines.get()

    @staticmethod
    def _pump(loop: asyncio.AbstractEventLoop, lines: asyncio.Queue[str | None]) -> None:
        """Hand each line of stdin to the loop, then None at the end of input."""
        while True:
            line = sys.stdin.readline()
            try:
                loop.call_soon_threadsafe(lines.put_nowait, line.strip() if line else None)
            except RuntimeError:
                return  # the loop has closed: the session is over
            if not line:
                return


class Session:
    """Talks to the team for one run. See the module docstring."""

    def __init__(
        self,
        team: App,
        console: Console,
        *,
        plain: bool = False,
        read_line: LineReader | None = None,
        listen: bool | None = None,
    ) -> None:
        """Show the run on ``console`` and read the user's lines with ``read_line``
        (stdin by default). ``listen`` also reads them while the team works; by default
        it does when stdin is a terminal, so piped answers wait for their prompts."""
        self.team = team
        self.console = console
        self.read_line = read_line or StdinLines()
        self.listen = sys.stdin.isatty() if listen is None else listen
        self.view = RunView(console, plain=plain)
        self.view.attach(team.bus)
        self.lock = DriveLock(team.runs_dir)
        team.bus.on_after(EventName.RUN_STARTED, self._on_run_started)
        self._seen: dict[str, Snapshot] = {}  # per run, the versions last reviewed
        #: The read in flight, kept when the team stops first so no line is lost.
        self._reading: asyncio.Task[str | None] | None = None
        self._ended = False  # the input has ended
        #: Lines typed while the team worked that change the run, sent once it stops.
        self._held: list[str] = []
        #: The run the team is working on now, which a run started meanwhile replaces.
        self._driving: str | None = None
        #: The run a typed stop was sent to in this drive, if any.
        self._stopping: str | None = None

    async def _on_run_started(self, event: Event) -> None:
        """Hold the lock of a run this session just started; lines typed from now on
        are about it."""
        self._driving = event.run_id
        self.lock.acquire(event.run_id)

    def say(self, text: str) -> None:
        """Print text as it is, without Rich markup."""
        self.console.print(text, markup=False, highlight=False)

    async def _drive(self, work: Awaitable[Reply], run_id: str | None = None) -> Reply:
        """Await the team's work on ``run_id`` (or the run it starts) with the live view
        running, reading what the user types meanwhile when the session listens."""
        self.view.start(hint=TYPE_HINT if self.listen else None)
        self._driving, self._stopping = run_id, None
        try:
            if not self.listen:
                return await work
            reply = await self._listen(asyncio.ensure_future(work))
        finally:
            self.view.stop()
        return await self._settle_stop(reply)

    async def _settle_stop(self, reply: Reply) -> Reply:
        """After a typed stop, make sure the run's state matches: a run that finished
        before its next model call keeps no stop flag (it would refuse every later
        message), and one left waiting on the user is marked stopped."""
        run_id, self._stopping = self._stopping, None
        if run_id is None:
            return reply
        run = await self.team.stores.tasks.get_run(run_id)
        if run.status is RunStatus.DONE:
            self.team.kill_switch.clear(run_id)
            self.say("The run finished before it could stop.")
        elif run.status is not RunStatus.STOPPED:
            await self.team.orchestrator.stop(run_id, STOP_REASON)
            return Reply(kind="stopped", text="Stopped the run.", run_id=run_id)
        return reply

    async def _listen(self, work: asyncio.Future[Reply]) -> Reply:
        """Read lines while ``work`` runs and hand each to ``_while_working``. A read
        still waiting when the work ends is kept for the next prompt."""
        try:
            while True:
                if self._reading is None and not self._ended:
                    self._reading = asyncio.ensure_future(self.read_line())
                waits: set[asyncio.Future[Any]] = {work}
                if self._reading is not None:
                    waits.add(self._reading)
                await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                if work.done():
                    return work.result()
                line = self._take()
                if line is not None:
                    self.view.line_typed()
                if line:
                    await self._guarded(line)
        finally:
            if not work.done():
                work.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await work

    def _take(self) -> str | None:
        """The finished read's line; None marks the end of input."""
        assert self._reading is not None and self._reading.done()
        reading, self._reading = self._reading, None
        line = reading.result()
        if line is None:
            self._ended = True
        return line

    async def _guarded(self, line: str) -> None:
        """Handle a line typed while the team works; a failure there is logged and
        said, and never reaches the team's work."""
        try:
            await self._while_working(line, self._driving)
        except Exception:
            log.exception("couldn't handle a line typed while the team worked")
            self._under_view(
                line, "I couldn't read that just now. Send it again once the team stops."
            )

    async def _while_working(self, line: str, run_id: str | None) -> None:
        """One line typed while the team works: "stop" stops the run at its next model
        call; anything else goes to the message loop, and its reply prints under the
        view. A line that changes the run, or one typed before the run exists, is held
        until the team stops."""
        if run_id is None:
            self._held.append(line)
            self._under_view(line, "I'll take that up as soon as the team stops.")
            return
        if says_stop(line):
            self.team.kill_switch.stop(run_id, STOP_REASON)
            self._stopping = run_id
            self._under_view(line, STOPPING)
            return
        reply = await self.team.orchestrator.handle_while_working(line, run_id)
        if reply.held:
            self._held.append(line)
        self._under_view(line, reply.text)

    def _under_view(self, line: str, text: str) -> None:
        """Print the typed line and the reply to it under the live view: the view pauses,
        they print, and the view starts again below them, so the screen keeps one tree.
        The line is echoed because a redraw of the live view clears what was typed."""
        self.view.pause()
        self.say(f"> {line}")
        self.say(text)
        self.view.start(hint=TYPE_HINT)

    async def start(self, request: str, *, context_dirs: list[str] | None = None) -> int:
        """Start a run for ``request`` and talk until it is done or left waiting."""
        try:
            reply = await self._drive(
                self.team.orchestrator.handle(request, context_dirs=context_dirs)
            )
            return await self._talk(reply, reply.run_id)
        finally:
            self.lock.release_all()
            self._outside_runs()

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
            stores = self.team.stores
            tasks = await stores.tasks.list_tasks(run_id)
            self.view.load(run_id, tasks, await stores.decisions.list(run_id))
            reply = await self._drive(self.team.orchestrator.resume(run_id), run_id)
            return await self._talk(reply, run_id)
        finally:
            self.lock.release_all()
            self._outside_runs()

    def _outside_runs(self) -> None:
        """Say what reading messages outside any run cost, since no run's stats hold it."""
        spent = self.view.session
        if spent.calls:
            self.say(f"Outside any run: {spent.text()}")

    async def _show(self, reply: Reply) -> None:
        """Print a reply, with the run id and the run's state as the store has it, such
        as done, paused (it can resume after the budget) or waiting for you."""
        self.say(reply.text)
        if reply.run_id:
            run = await self.team.stores.tasks.get_run(reply.run_id)
            self.say(f"[run {reply.run_id}: {STATE_WORDS.get(run.status, run.status.value)}]")

    async def _talk(self, reply: Reply, run_id: str | None) -> int:
        """Show each reply and pass on the user's lines. Returns the exit code: 0 once
        the run is done and reviewed or left waiting, 1 when it stopped or failed with
        nothing to answer, or another session took it over."""
        while True:
            await self._show(reply)
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
                self._drop_held()
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
            reply = await self._drive(self.team.orchestrator.handle(line, run_id=run_id), run_id)

    def _drop_held(self) -> None:
        """Say which held lines won't be sent, now that the session ends."""
        for line in self._held:
            self.say(f"Not sent, since the run ended: {line}")
        self._held.clear()

    async def _is_done(self, run_id: str) -> bool:
        """Whether the run is finished, so the user is reviewing it."""
        run = await self.team.stores.tasks.get_run(run_id)
        return run.status is RunStatus.DONE

    async def _next_line(self) -> str | None:
        """The next non-empty line, or None at the end of input. A line held while the
        team worked comes first, echoed as if typed now."""
        if self._held:
            held = self._held.pop(0)
            self.say(f"> {held}")
            return held
        line = await self._read()
        while line == "":
            line = await self._read()
        return line

    async def _read(self) -> str | None:
        """One line at a prompt: the read already in flight, or a new one."""
        if self._ended:
            return None
        self.console.print("> ", end="", markup=False, highlight=False)
        if self._reading is None:
            self._reading = asyncio.ensure_future(self.read_line())
        await asyncio.wait({self._reading})
        return self._take()

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
