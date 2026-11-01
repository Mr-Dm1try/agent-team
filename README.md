# agent-team

A Python multi-agent harness that takes a medium-size software task through
**Planning → PRD → Spec → Implementation**, with expert agents, an auditor that
checks every step, and a streaming CLI where you review what was built and ask
for changes.

Built for the AI Engineer Challenge: Research Agent Teams.

> **Status:** design done, implementation starting. Nothing runs yet.

## What it does

- **Orchestrator** you talk to through a streaming terminal session. It plans,
  delegates every piece of work through a dispatcher, and decides retries and re-runs.
- **Five expert agents:** Researcher (files and web), Product (PRD), Architect
  (spec), Engineer (code and unit tests) and QA (reviews and runs the solution).
- **Auditor** that sweeps every task and phase gate against rules stored in
  SQLite, covering process errors and dangerous code or agent actions.
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

## Quick start

Planned usage once the CLI lands:

```bash
uv sync
export ANTHROPIC_API_KEY=...
uv run agent-team run "Build a REST API for a small library inventory"
```

## License

GNU Affero General Public License v3.0 only. See [LICENSE](LICENSE).
