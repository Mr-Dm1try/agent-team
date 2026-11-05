"""The real team for end-to-end tests: the composition root over SQLite, the gateway,
the rule catalog and the auditor, with only the model scripted."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agents.scenario import CODE, PRD, Responder, team_script
from sandbox.fakes import FakeRunner

from agent_team.app import App, build_app
from agent_team.audit.auditor import SECOND_CHECK_PROMPT
from agent_team.audit.precheck import PRECHECK_PROMPT
from agent_team.core.config import Config, deep_merge, load_config
from agent_team.core.models import AgentName, Event, EventName, Usage
from agent_team.llm.fake import FakeProvider
from agent_team.llm.types import LLMRequest, LLMResponse, ToolResultBlock, ToolUseBlock
from agent_team.memory.embed import FakeEmbedder

PASS = {"summary": "fine", "findings": []}
NOTES = "SQLite needs no server.\n"
#: The auditor model's verdict on one sweep: given the agent and phase under check
#: (both None for the run sweep), the JSON reply.
AuditorModel = Callable[[str | None, str | None], Any]
_SWEPT = re.compile(r"^Task \S+: agent (\w+), phase (\w+)", re.MULTILINE)


def e2e_config(tmp_path: Path, **extra: Any) -> Config:
    """The default config with every route on the fake provider, instant retries and
    everything stored under ``tmp_path``."""
    fake = {"provider": "fake", "model": "fake-1"}
    base = load_config()
    overrides = {
        "defaults": fake,
        "agents": {agent.value: fake for agent in base.agents},
        "stronger_model": {"provider": "fake", "model": "fake-strong"},
        "fallbacks": [],
        "fallback_models": {},
        "retry": {"delays_s": [0, 0, 0]},
        "budget": {"max_usd_per_run": None, "max_minutes_per_run": None},
        "limits": {"task_timeout_s": 30},
        "hooks": [],
        "paths": {"runs_dir": str(tmp_path / "runs"), "db": str(tmp_path / "team.db")},
    }
    return load_config(overrides=deep_merge(overrides, extra))


def passing_auditor(agent: str | None, phase: str | None) -> Any:
    """An auditor model that finds nothing."""
    return PASS


def scripted_provider(
    script: dict[AgentName, Responder], auditor: AuditorModel = passing_auditor
) -> FakeProvider:
    """A fake provider that answers each agent from ``script`` and the auditor from
    ``auditor``. Pre-checks are allowed and second checks confirm."""

    def respond(request: LLMRequest) -> LLMResponse | str:
        """Route one request to its agent's scripted reply."""
        if request.agent is AgentName.AUDITOR:
            return json.dumps(_audit(request, auditor))
        out = script[request.agent](request)
        if isinstance(out, str):
            return out
        tools = any(isinstance(b, ToolUseBlock) for b in out)
        return LLMResponse(
            content=out,
            stop_reason="tool_use" if tools else "end_turn",
            usage=Usage(input_tokens=100, output_tokens=10),
            provider="fake",
            model="fake-1",
        )

    return FakeProvider(responder=respond)


def _audit(request: LLMRequest, auditor: AuditorModel) -> Any:
    """The auditor model's reply: pre-checks allow, second checks confirm, and sweeps
    go to ``auditor`` with the agent and phase being checked."""
    if request.system == PRECHECK_PROMPT:
        return {"decision": "allow", "reason": "normal work"}
    if request.system == SECOND_CHECK_PROMPT:
        return {"confirmed": True, "reason": "it holds"}
    prompt = getattr(request.messages[0].content[0], "text", "")
    swept = _SWEPT.search(prompt)
    return auditor(swept.group(1), swept.group(2)) if swept else auditor(None, None)


def _tool_results(request: LLMRequest) -> int:
    """How many tool results the conversation holds so far."""
    return sum(isinstance(b, ToolResultBlock) for m in request.messages for b in m.content)


def researcher_with_notes(request: LLMRequest) -> Any:
    """The scripted Researcher, which first opens the notes it cites."""
    if _tool_results(request) == 0:
        return [ToolUseBlock(id="call_notes", name="read_file", arguments={"path": "notes.md"})]
    return team_script()[AgentName.RESEARCHER](request)


TESTS = {"command": "pytest -q", "passed": 1, "failed": 0, "exit_code": 0, "output_tail": "ok"}


def engineer_with_tests(request: LLMRequest) -> Any:
    """The scripted Engineer: write the module, install pytest, run the tests, report
    them."""
    done = _tool_results(request)
    if done == 0:
        args = {"path": "app.py", "content": "print('books')\n"}
        return [ToolUseBlock(id="call_write", name="write_file", arguments=args)]
    if done == 1:
        args = {"packages": ["pytest==8.3.3"]}
        return [ToolUseBlock(id="call_deps", name="install_deps", arguments=args)]
    if done == 2:
        args = {"argv": ["pytest", "-q"]}
        return [ToolUseBlock(id="call_tests", name="run_command", arguments=args)]
    return json.dumps(dict(CODE, tests=TESTS))


def e2e_script() -> dict[AgentName, Responder]:
    """The scripted team, with a Researcher that opens its sources and an Engineer
    that runs its tests."""
    return dict(
        team_script(),
        **{AgentName.RESEARCHER: researcher_with_notes, AgentName.ENGINEER: engineer_with_tests},
    )


QUESTION = [{"id": "Q-1", "text": "Paginate the list?", "why": "list size"}]


def asking_script() -> dict[AgentName, Responder]:
    """``e2e_script`` with a Product that asks one question before writing the PRD."""
    script = e2e_script()
    calls: list[int] = []

    def product(request: LLMRequest) -> str:
        """Ask on the first attempt, then write a plain PRD."""
        calls.append(1)
        return json.dumps(dict(PRD, questions=QUESTION) if len(calls) == 1 else PRD)

    script[AgentName.PRODUCT] = product
    return script


async def scripted_app(
    config: Config,
    *,
    script: dict[AgentName, Responder] | None = None,
    auditor: AuditorModel = passing_auditor,
    runner: FakeRunner | None = None,
) -> App:
    """The real app from ``config`` over a scripted model and the fake embedder, with
    the real command tools over a fake runner. Each new run's workspace starts with the
    notes the Researcher reads."""
    provider = scripted_provider(script or e2e_script(), auditor)
    app = await build_app(
        config,
        providers={"fake": provider},
        runner=runner or FakeRunner(),
        embedder=FakeEmbedder(),
    )

    async def seed(event: Event) -> None:
        """Put the project notes in the new run's workspace."""
        run = await app.stores.tasks.get_run(event.run_id)
        await asyncio.to_thread(Path(run.workspace, "notes.md").write_text, NOTES)

    app.bus.on_after(EventName.RUN_STARTED, seed)
    return app


async def e2e_app(
    tmp_path: Path,
    *,
    script: dict[AgentName, Responder] | None = None,
    auditor: AuditorModel = passing_auditor,
    runner: FakeRunner | None = None,
    **config: Any,
) -> App:
    """``scripted_app`` over ``e2e_config(tmp_path, **config)``."""
    return await scripted_app(
        e2e_config(tmp_path, **config), script=script, auditor=auditor, runner=runner
    )
