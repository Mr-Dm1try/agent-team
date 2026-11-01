"""Command-line entry point. The streaming session lands in step 3b."""

from __future__ import annotations

import typer

from agent_team import __version__

app = typer.Typer(help="Multi-agent harness: Planning, PRD, Spec and Implementation.")


@app.callback()
def main() -> None:
    """Multi-agent harness: Planning, PRD, Spec and Implementation."""


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
