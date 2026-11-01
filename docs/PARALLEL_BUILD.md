# Parallel build: steps 2a, 2b and 2c

Steps 2a, 2b and 2c from [DESIGN.md](DESIGN.md) are built at the same time by
separate agents. They meet only at the contracts in:

- `src/agent_team/core/models.py`: tasks, runs, envelopes, results, rules, findings,
  verdicts, decisions, events, and the allowed state transitions
- `src/agent_team/core/artifacts.py`: the typed artifacts and the ASK, REQ, AC id chain
- `src/agent_team/core/interfaces.py`: Protocols for every service a step provides
- `src/agent_team/core/errors.py`: the typed errors
- `src/agent_team/llm/types.py`: provider-neutral messages and the LLM client Protocol

## Rules for every step

1. **Code against the Protocols, never against another step's classes.** Write small
   fakes in your own tests for anything another step provides.
2. **Don't edit the contract files.** If you need a change, ask the orchestrating
   thread. Adding an optional field is usually fine; say so in the PR.
3. **Stay in your folders** (below). `pyproject.toml` already lists every dependency;
   if you truly need another, add only that line and call it out in the PR.
4. **Every PR passes** `uv run ruff check .`, `uv run ruff format --check .`,
   `uv run mypy` and `uv run pytest -q`. No real model calls in tests: use the fake
   provider (2a) or your own stub.
5. One branch and one PR per step, against `main`.

## Ownership

| Step | Owns | Provides |
| --- | --- | --- |
| 2a | `core/` except the contract files, `llm/`, `config/default.yaml`, `tests/core/`, `tests/llm/` | SQLite stores (runs, tasks, artifacts with lineage, checkpoints, decisions, feedback, verdicts) with WAL; event bus with before and after hooks; JSONL tracer per agent plus `events.jsonl`, spans and tracebacks; config loading; LLM gateway (routing, fallback, retries with backoff, rate limit, budgets, kill switch, `--chaos` fault injection); fake, Anthropic and Ollama providers |
| 2b | `agents/`, `orchestrator/`, `tools/`, `tests/agents/`, `tests/orchestrator/` | Base agent tool loop with repair on bad JSON, checkpoints per step and context compaction; Researcher, Product, Architect, Engineer and QA with prompts; file tools with workspace and `--context` path rules; dispatcher with timeouts and error capture; orchestrator with request classification, planning, control tools, phase state machine, question answering and the decision log calls |
| 2c | `audit/`, `config/rules.yaml`, `tests/audit/` | Rule catalog (P1, P2, R1 to R12, S1 to S7) and SQLite rule store filtered by scope, phase and agent; evidence helpers; auditor agent with fail-closed verdicts and a second check for unsupported blockers; tool pre-check; verdict policy (retry as revision, no-progress detection, one attempt on the stronger model, upstream re-run, gate fix routing, caps) returned as Decisions for the orchestrator to apply |

## Done when

- **2a:** unit tests cover every store, the bus, the gateway's retry, fallback, budget
  and chaos paths, and the fake provider replays a scripted conversation.
- **2b:** with fake stores, a fake LLM and an always-pass auditor, a run goes through
  all four phases and produces every artifact; a research question returns a
  `research_answer` without PRD or code.
- **2c:** each rule has a passing and a failing example; the verdict policy has a test
  for every branch in DESIGN.md's "What a verdict triggers" list.

Integration (wiring the three together for one end-to-end fake-model run) happens
after all three PRs merge.
