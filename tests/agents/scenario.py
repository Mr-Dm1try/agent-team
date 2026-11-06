"""A scripted team: canned replies per agent that make a whole run go through."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from agent_team.agents.prompts import RUN_SUMMARY
from agent_team.core.models import AgentName
from agent_team.llm.types import ContentBlock, LLMRequest, ToolResultBlock, ToolUseBlock
from agents.fakes import reply, tool_call

Responder = Callable[[LLMRequest], list[ContentBlock] | str]

PLAN_BUILD: dict[str, Any] = {
    "summary": "One ask; the database choice needs research.",
    "request_kind": "build",
    "asks": [{"id": "ASK-1", "text": "A REST API that lists books"}],
    "research_questions": [{"question": "Which database?", "ask": "ASK-1"}],
}

PLAN_RESEARCH: dict[str, Any] = {
    "summary": "One question with one sub-question.",
    "request_kind": "research_question",
    "asks": [{"id": "ASK-1", "text": "Which Python web framework is fastest?"}],
    "research_questions": [{"question": "Which benchmarks are current?", "ask": "ASK-1"}],
}

BRIEF = {
    "summary": "SQLite fits.",
    "answers": [
        {"question": "Which database?", "answer": "SQLite", "sources": [{"ref": "notes.md"}]}
    ],
    "findings": [{"claim": "SQLite needs no server", "sources": [{"ref": "notes.md"}]}],
}
ANSWER = {
    "summary": "Starlette-based frameworks lead.",
    "question": "Which Python web framework is fastest?",
    "answer": "Starlette-based frameworks lead the benchmarks.",
    "sources": [{"ref": "https://example.com/bench"}],
}
PRD = {
    "summary": "One requirement.",
    "title": "Books API",
    "requirements": [
        {
            "id": "REQ-1",
            "text": "List books",
            "asks": ["ASK-1"],
            "acceptance_criteria": [{"id": "AC-1.1", "text": "GET /books returns 200"}],
        }
    ],
}
SPEC = {
    "summary": "One module.",
    "components": [
        {"name": "api", "responsibility": "routes", "requirements": ["REQ-1"], "files": ["app.py"]}
    ],
    "checks": [{"id": "CHK-1", "criteria": ["AC-1.1"], "kind": "unit_test", "description": "test"}],
    "dependencies": ["fastapi==0.115.0"],
}
CODE = {"summary": "Wrote app.py.", "files_written": ["app.py"]}
QA = {
    "summary": "All good.",
    "results": [{"criterion": "AC-1.1", "result": "pass", "evidence": "app.py:1"}],
    "ran_end_to_end": False,
}


def _last_is_tool_result(request: LLMRequest) -> bool:
    last = request.messages[-1]
    return any(isinstance(b, ToolResultBlock) for b in last.content)


MESSAGE_PROMPT = "You are the orchestrator of a software-building agent team"


def message_reply(text: str, *kinds: str) -> str:
    """The message loop's final JSON: the message's kinds and the reply."""
    return json.dumps({"kinds": list(kinds) or ["run_question"], "reply": text})


def read_message(request: LLMRequest) -> tuple[str, list[str]]:
    """The user's message and the waiting task ids from a message-loop prompt."""
    text = getattr(request.messages[0].content[0], "text", "")
    said = text.split("The user's message:\n", 1)[-1]
    return said, re.findall(r"task_id=(\S+),", text)


def default_message(kind: str) -> Responder:
    """A careful model reading one message: start a run when there is none, answer the
    first waiting item with the user's words, and otherwise just reply."""

    def respond(request: LLMRequest) -> list[ContentBlock] | str:
        """One turn of the scripted message loop."""
        if _last_is_tool_result(request):
            return message_reply("On it.", *called_kinds(request, kind))
        said, waiting = read_message(request)
        text = getattr(request.messages[0].content[0], "text", "")
        if text.startswith("No run is in progress."):
            return tool_call("start_run", {"request": said, "kind": kind})
        if waiting:
            return tool_call("answer_pending", {"task_id": waiting[0], "text": said})
        return message_reply("Noted.")

    return respond


def called_kinds(request: LLMRequest, kind: str) -> list[str]:
    """The kinds that fit the action tools the scripted loop called so far."""
    names = {
        b.name
        for m in request.messages
        if m.role == "assistant"
        for b in m.content
        if isinstance(b, ToolUseBlock)
    }
    kinds = {"start_run": kind, "answer_pending": "answer", "request_change": "change"}
    return [k for name, k in kinds.items() if name in names] or ["run_question"]


def orchestrator_responder(
    *,
    kind: str = "build",
    plan: dict[str, Any] | None = None,
    message: Responder | None = None,
) -> Responder:
    """The scripted orchestrator: the message loop (``message``, or a careful default),
    the planner, run questions and summaries."""
    on_message = message or default_message(kind)

    def respond(request: LLMRequest) -> list[ContentBlock] | str:
        """Route by system prompt to the scripted reply for that call."""
        if request.system.startswith(MESSAGE_PROMPT):
            return on_message(request)
        if "Orchestrator's planner" in request.system:
            chosen = plan or (PLAN_RESEARCH if kind == "research_question" else PLAN_BUILD)
            return json.dumps(chosen)
        if "questions about a run" in request.system:
            if not _last_is_tool_result(request):
                return tool_call("list_tasks", {})
            return json.dumps({"answer": "The PRD retried once.", "cited": ["prd@v2"]})
        if request.system == RUN_SUMMARY:
            return json.dumps({"summary": "Built the books API with one test."})
        if request.system.startswith("You compress"):
            return json.dumps({"done": ["things"], "decisions": [], "remaining": []})
        raise AssertionError("unexpected orchestrator call")

    return respond


def engineer_responder(request: LLMRequest) -> list[ContentBlock] | str:
    if not _last_is_tool_result(request):
        return tool_call("write_file", {"path": "app.py", "content": "print('books')\n"})
    return json.dumps(CODE)


def researcher_responder(request: LLMRequest) -> list[ContentBlock] | str:
    first = request.messages[0].content[0]
    text = getattr(first, "text", "")
    if "Produce: research_answer" in text:
        return json.dumps(ANSWER)
    return json.dumps(BRIEF)


def team_script(kind: str = "build") -> dict[AgentName, Any]:
    return {
        AgentName.ORCHESTRATOR: orchestrator_responder(kind=kind),
        AgentName.RESEARCHER: researcher_responder,
        AgentName.PRODUCT: lambda r: json.dumps(PRD),
        AgentName.ARCHITECT: lambda r: json.dumps(SPEC),
        AgentName.ENGINEER: engineer_responder,
        AgentName.QA: lambda r: json.dumps(QA),
    }


__all__ = ["reply"]
