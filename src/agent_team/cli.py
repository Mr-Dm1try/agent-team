"""Command-line entry point: start, resume and stop runs, read their logs and stats, and
search memory.

``run`` and ``resume`` show the run live and talk to the team (see ui/session.py).
While the run waits on you, each line typed is sent as a message; end the input
(Ctrl-D) to leave the run waiting, and pick it up later with ``resume``. When it is
done you accept it or describe a change.

At startup ``entry`` reads ``.env`` from the current folder, so the API key can live
there instead of in the shell. Variables already set in the shell win.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Column, Table

from agent_team import __version__
from agent_team.app import App, build_app
from agent_team.core.config import Config, load_config
from agent_team.core.control import DriveLock, KillSwitch
from agent_team.core.db import Database
from agent_team.core.errors import SandboxError
from agent_team.core.events import InMemoryEventBus
from agent_team.core.interfaces import Embedder
from agent_team.core.models import Event, MemoryKind, RunStatus
from agent_team.core.stores import SqliteDecisionLog, SqliteTaskStore
from agent_team.core.tracing import JsonlTracer, load_spans, read_events, render_tree, span_tree
from agent_team.llm.chaos import parse_chaos
from agent_team.memory import open_search
from agent_team.memory.rank import line
from agent_team.orchestrator.base import mark_stopped
from agent_team.ui.session import Session
from agent_team.ui.stats import HEADER, overview_table, run_stats, run_table

app = typer.Typer(help="Multi-agent harness: Planning, PRD, Spec and Implementation.")
memory_app = typer.Typer(help="Long-term memory from earlier runs.")
app.add_typer(memory_app, name="memory")

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="Config file (default: config/default.yaml).")
]
ChaosOption = Annotated[
    str | None,
    typer.Option(help="Inject model faults, e.g. 0.3 or rate_limit,timeout@0.5."),
]
#: Builds the app from a config; tests swap it for one over a scripted model.
AppFactory = Callable[[Config], Awaitable[App]]
_factory: AppFactory = build_app
#: The embedder ``memory search`` uses; None is the configured Ollama one.
_embedder: Embedder | None = None


def load_env(folder: Path | None = None) -> bool:
    """Load ``.env`` from ``folder`` (default: the current folder) into the
    environment without replacing variables already set; False when there is none."""
    return load_dotenv((folder or Path.cwd()) / ".env", override=False)


def entry() -> None:
    """The ``agent-team`` command: load ``.env``, then run the CLI. It loads here, not in
    the ``app`` callback, so tests that drive ``app`` never pick up a developer's .env."""
    try:
        load_env()
    except (OSError, UnicodeDecodeError) as exc:
        typer.echo(f"Can't read .env: {exc}", err=True)
        raise SystemExit(2) from exc
    app()


@app.callback()
def main() -> None:
    """Multi-agent harness: Planning, PRD, Spec and Implementation."""


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


PlainOption = Annotated[
    bool, typer.Option("--plain", help="Print one line per event instead of the live view.")
]
ContextOption = Annotated[
    list[Path] | None,
    typer.Option(
        "--context",
        help="A folder the Researcher may read but never write; repeat for more.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
]


@app.command()
def run(
    request: str,
    config: ConfigOption = None,
    chaos: ChaosOption = None,
    plain: PlainOption = False,
    context: ContextOption = None,
) -> None:
    """Start a run for REQUEST, show it live and talk to it until it is done."""
    folders = [str(p.resolve()) for p in context or []]
    raise typer.Exit(
        asyncio.run(_talk(_config(config, chaos), plain=plain, request=request, context=folders))
    )


@app.command()
def resume(
    run_id: str, config: ConfigOption = None, chaos: ChaosOption = None, plain: PlainOption = False
) -> None:
    """Continue RUN_ID after a stop, a crash or a question left open."""
    raise typer.Exit(asyncio.run(_talk(_config(config, chaos), plain=plain, run_id=run_id)))


@app.command()
def stop(
    run_id: str,
    reason: Annotated[str, typer.Option(help="Why, for the decision log.")] = "stopped by the user",
    config: ConfigOption = None,
) -> None:
    """Stop RUN_ID. A session driving it stops at its next model call; a run nobody is
    driving is marked stopped now. Either way, ``resume`` continues it."""
    raise typer.Exit(asyncio.run(_stop(load_config(config), run_id, reason)))


@app.command()
def stats(
    run_id: Annotated[str | None, typer.Argument(help="One run; all runs if left out.")] = None,
    config: ConfigOption = None,
) -> None:
    """Tokens, cost, time, retries and first-pass audit rate, per run and per agent."""
    raise typer.Exit(asyncio.run(_stats(load_config(config), run_id)))


@app.command()
def logs(
    run_id: str,
    agent: Annotated[str | None, typer.Option(help="Only this agent's events.")] = None,
    task: Annotated[str | None, typer.Option(help="Only this task's events.")] = None,
    spans: Annotated[bool, typer.Option(help="Print the run's trace as a tree.")] = False,
    config: ConfigOption = None,
) -> None:
    """Print a run's events, one per line, oldest first."""
    runs_dir = load_config(config).paths.runs_dir
    if spans:
        for line in render_tree(span_tree(load_spans(runs_dir, run_id))):
            typer.echo(line)
        return
    events = read_events(runs_dir, run_id, agent=agent, task_id=task)
    if not events:
        typer.echo(f"No events for run {run_id}.", err=True)
        raise typer.Exit(1)
    for event in events:
        typer.echo(format_event(event))


@memory_app.command("search")
def memory_search(
    query: str,
    kind: Annotated[
        list[MemoryKind] | None, typer.Option(help="Only this kind; repeat for several.")
    ] = None,
    limit: Annotated[int, typer.Option(min=1, help="Most results to print.")] = 10,
    config: ConfigOption = None,
) -> None:
    """Print the memories that best match QUERY, best first, with their scores."""
    raise typer.Exit(
        asyncio.run(_search(load_config(config), query, kind or list(MemoryKind), limit))
    )


async def _search(config: Config, query: str, kinds: list[MemoryKind], limit: int) -> int:
    """Run one search and print it; returns 1 when nothing matched."""
    parts = await open_search(config, embedder=_embedder)
    try:
        found = await parts.search.search(query, kinds, run_id=None)
    finally:
        parts.close()
    if found.mode == "bm25":
        typer.echo(f"Keyword search only ({found.embedder_error}).", err=True)
    if not found.ranked:
        typer.echo("No memory matched.", err=True)
        return 1
    for r in found.ranked[:limit]:
        typer.echo(f"{r.score:.3f}  {r.memory.id}  {line(r.memory, None)}")
    return 0


def _config(path: Path | None, chaos: str | None) -> Config:
    """The config file, with ``--chaos`` in place of its chaos settings."""
    config = load_config(path)
    if chaos is not None:
        config = config.model_copy(update={"chaos": parse_chaos(chaos)})
    return config


def _console() -> Console:
    """A console on the current stdout (tests replace it)."""
    return Console(file=sys.stdout, soft_wrap=True)


async def _talk(
    config: Config,
    *,
    plain: bool = False,
    request: str | None = None,
    run_id: str | None = None,
    context: list[str] | None = None,
) -> int:
    """Start the request (or resume the run) in a session. Returns the exit code: 0
    once the run is done or left waiting, 1 when it stopped or failed with nothing to
    answer, or the sandbox couldn't start."""
    try:
        team = await _factory(config)
    except SandboxError as exc:
        typer.echo(f"Can't start: {exc}", err=True)
        return 1
    with team:
        session = Session(team, _console(), plain=plain)
        if run_id is not None:
            return await session.resume(run_id)
        return await session.start(request or "", context_dirs=context)


@contextmanager
def _stores(config: Config) -> Iterator[tuple[SqliteTaskStore, SqliteDecisionLog]]:
    """The run and decision stores over the config's database, without the team.
    Decisions also go to the run's events.jsonl, as the team's do."""
    db = Database(config.paths.db)
    bus = InMemoryEventBus()
    JsonlTracer(config.paths.runs_dir).attach(bus)
    try:
        yield SqliteTaskStore(db), SqliteDecisionLog(db, bus)
    finally:
        db.close()


async def _stop(config: Config, run_id: str, reason: str) -> int:
    """Stop one run; see ``stop``."""
    runs_dir = config.paths.runs_dir
    with _stores(config) as (tasks, decisions):
        try:
            run = await tasks.get_run(run_id)
        except KeyError:
            typer.echo(f"No run {run_id}.", err=True)
            return 1
        if run.status is RunStatus.DONE:
            typer.echo(f"Run {run_id} is already done.")
            return 0
        KillSwitch(runs_dir).stop(run_id, reason)
        if DriveLock(runs_dir).is_driven(run_id):
            typer.echo(f"Asked the session driving {run_id} to stop at its next model call.")
            return 0
        await mark_stopped(tasks, decisions, run_id, reason)
    typer.echo(f"Stopped run {run_id}. Continue: agent-team resume {run_id}")
    return 0


async def _stats(config: Config, run_id: str | None) -> int:
    """Print the stats of one run, or of every run; see ``stats``."""
    runs_dir = config.paths.runs_dir
    with _stores(config) as (tasks, _):
        if run_id is not None:
            try:
                rows = run_table(await run_stats(run_id, tasks, runs_dir))
            except KeyError:
                typer.echo(f"No run {run_id}.", err=True)
                return 1
        else:
            runs = await tasks.list_runs()
            if not runs:
                typer.echo("No runs yet.")
                return 0
            rows = overview_table([await run_stats(r.id, tasks, runs_dir) for r in runs])
    name, *numbers = HEADER
    # Numbers never wrap; on a narrow terminal the names fold instead.
    columns = [
        Column(name, overflow="fold"),
        *(Column(h, justify="right", no_wrap=True) for h in numbers),
    ]
    table = Table(*columns, box=None, pad_edge=False)
    for r in rows:
        table.add_row(*r)
    _console().print(table)
    return 0


def format_event(event: Event) -> str:
    """One log line: time, agent, task and attempt, event name and its data."""
    who = event.agent.value if event.agent else "-"
    where = f"{event.task_id}#{event.attempt}" if event.task_id else "-"
    data = json.dumps(_short(event.data), ensure_ascii=False, default=str)
    return f"{event.ts:%H:%M:%S.%f}"[:-3] + f"  {who:<12} {where:<24} {event.name.value}  {data}"


def _short(value: Any, limit: int = 200) -> Any:
    """``value`` with long strings cut, so one event stays on one readable line."""
    if isinstance(value, str):
        value = value.replace("\n", " ")
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _short(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_short(v, limit) for v in value]
    return value


if __name__ == "__main__":
    entry()
