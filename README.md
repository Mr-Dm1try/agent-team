# agent-team

A Python multi-agent harness that takes a medium-size software task through
**Planning → PRD → Spec → Implementation**, with expert agents, an auditor that
checks every step, and a streaming CLI where you review what was built and ask
for changes.

Built for the AI Engineer Challenge: Research Agent Teams.

> **Status:** the team, the auditor, the gateway, the Docker sandbox, long-term memory
> and the CLI run end to end (the tests drive a whole build with a scripted model). A
> real build needs Docker running.

## What it does

- **Orchestrator** you talk to through a streaming terminal session. It plans,
  delegates every piece of work through a dispatcher, and decides retries and re-runs.
- **Five expert agents:** Researcher (files and web), Product (PRD), Architect
  (spec), Engineer (code and unit tests) and QA (reviews and runs the solution).
- **Auditor** that checks the end of every task and the whole run against rules
  stored in SQLite, covering process errors and dangerous code or agent actions.
- **Task tracker** with a full lifecycle, versioned artifacts with lineage,
  checkpoints, crash resume, and re-runs that update work instead of rebuilding it.
- **Per-agent log trails, traces and tracebacks**, plus a decision log that
  records why every choice was made.
- **Memory** with automatic retrieval (embeddings plus BM25) so later runs reuse
  lessons, decisions and preferences.
- **LLM gateway** with per-agent model routing, fallbacks, budgets and a kill switch.
  Claude Haiku 5.5 by default, with Ollama for local open-source models.
- **Docker sandbox** for all generated code.

The full design is in [docs/DESIGN.md](docs/DESIGN.md).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Docker (Docker Desktop, OrbStack or Colima)
- An Anthropic API key, and optionally [Ollama](https://ollama.com) for local models
  and memory embeddings (`ollama pull nomic-embed-text`; without it, memory search
  uses keywords alone)

## Quick start

```bash
uv sync
cp .env.example .env                       # then put your key in .env
uv run agent-team run "Build a REST API for a small library inventory"
uv run agent-team run "..." --plain --context ./docs  # line output; a read-only folder
uv run agent-team resume <run_id>              # continue a waiting, stopped or crashed run
uv run agent-team stop <run_id>                # stop at the next model call
uv run agent-team stats [run_id]               # tokens, cost, time, retries, first-pass rate
uv run agent-team logs <run_id> --agent qa      # one agent's events
uv run agent-team logs <run_id> --spans         # the run's trace tree
uv run agent-team memory search "FastAPI tests"  # what earlier runs learned
```

`config/default.yaml` runs the Researcher on local gpt-oss through Ollama. For a live
demo, use `--config config/demo.yaml`: it is the same except that the Researcher runs on
Claude Haiku, which is faster and steadier than local gpt-oss. In rehearsals a short
research question took 1 to 2.5 minutes on Haiku against 4.5 to 21 on gpt-oss.

`agent-team` reads `.env` from the folder you run it in. A variable already set in
your shell wins over the file.

While the team works you can type a question about the run and press Enter; the
answer prints under the live view. Type `stop` to stop the run at its next model call.
While a run waits on you, type your answer and press Enter. Ctrl-D leaves the run
waiting for `resume`. When it is done, type `accept` or describe a change; after the
change re-runs, you see what changed before you accept.

## License

GNU Affero General Public License v3.0 only. See [LICENSE](LICENSE).
