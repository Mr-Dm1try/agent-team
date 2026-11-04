"""Command-line entry point: start a run, resume one, read its logs.

``run`` and ``resume`` talk to the team until the run finishes. While the run waits on
you, each line typed is sent as a message; end the input (Ctrl-D) to leave the run
waiting, and pick it up later with ``resume``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from agent_team import __version__
from agent_team.app import App, build_app
from agent_team.core.config import Config, load_config
from agent_team.core.errors import SandboxError
from agent_team.core.models import Event
from agent_team.core.tracing import load_spans, read_events, render_tree, span_tree
from agent_team.llm.chaos import parse_chaos
from agent_team.orchestrator.replies import Reply

app = typer.Typer(help="Multi-agent harness: Planning, PRD, Spec and Implementation.")

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


@app.callback()
def main() -> None:
    """Multi-agent harness: Planning, PRD, Spec and Implementation."""


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def run(request: str, config: ConfigOption = None, chaos: ChaosOption = None) -> None:
    """Start a run for REQUEST and answer its questions from stdin until it is done."""
    raise typer.Exit(asyncio.run(_talk(_config(config, chaos), request=request)))


@app.command()
def resume(run_id: str, config: ConfigOption = None, chaos: ChaosOption = None) -> None:
    """Continue RUN_ID after a stop, a crash or a question left open."""
    raise typer.Exit(asyncio.run(_talk(_config(config, chaos), run_id=run_id)))


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


def _config(path: Path | None, chaos: str | None) -> Config:
    """The config file, with ``--chaos`` in place of its chaos settings."""
    config = load_config(path)
    if chaos is not None:
        config = config.model_copy(update={"chaos": parse_chaos(chaos)})
    return config


async def _talk(config: Config, *, request: str | None = None, run_id: str | None = None) -> int:
    """Send the request (or resume the run), then pass each typed line on while the run
    waits. Returns the exit code: 0 once the run is done or left waiting, 1 when it
    stopped or failed with nothing to answer, or the sandbox couldn't start."""
    try:
        team = await _factory(config)
    except SandboxError as exc:
        typer.echo(f"Can't start: {exc}", err=True)
        return 1
    with team:
        if run_id is not None:
            reply = await team.orchestrator.resume(run_id)
        else:
            reply = await team.orchestrator.handle(request or "")
        while True:
            _show(reply)
            run_id = reply.run_id or run_id
            if reply.kind == "done":
                return 0
            if reply.kind in ("stopped", "error") and not reply.pending:
                return 1
            line = await _read_line()
            while line == "":
                line = await _read_line()
            if line is None:
                if run_id:
                    typer.echo(f"Run {run_id} is waiting. Continue: agent-team resume {run_id}")
                return 0
            reply = await team.orchestrator.handle(line, run_id=run_id)


async def _read_line() -> str | None:
    """One line from stdin without blocking the event loop; None at end of input."""
    typer.echo("> ", nl=False)
    line = await asyncio.to_thread(sys.stdin.readline)
    return line.strip() if line else None


def _show(reply: Reply) -> None:
    """Print a reply, with the run id the first time it is known."""
    typer.echo(reply.text)
    if reply.run_id:
        typer.echo(f"[run {reply.run_id}: {reply.kind}]")


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
    app()
