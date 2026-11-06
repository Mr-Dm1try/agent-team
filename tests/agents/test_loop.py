from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from llm.conftest import Sleeps, make_config
from llm.test_ollama import error, ollama

from agent_team.agents import Engineer, Product
from agent_team.agents.compaction import Compactor
from agent_team.agents.loop import (
    FINAL_REPLY_NOTE,
    INTERRUPTED_NOTE,
    STEPS_LEFT_NOTE,
    LoopSettings,
    ToolLoop,
)
from agent_team.core.artifacts import CodeChange, ResearchBrief
from agent_team.core.errors import BudgetExceeded
from agent_team.core.interfaces import AgentContext, ToolContext
from agent_team.core.models import (
    AgentName,
    EventName,
    HookOutcome,
    Phase,
    ResultStatus,
    ReviewMode,
    Run,
    Task,
    TaskEnvelope,
    Usage,
)
from agent_team.llm.fake import FakeProvider
from agent_team.llm.gateway import Gateway
from agent_team.llm.types import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from agent_team.tools import file_tools
from agents.fakes import FakeBus, FakeCheckpoints, FakeLLM, prompt_tokens, tool_call
from agents.scenario import BRIEF, CODE, PRD


def envelope(tmp_path: Path, agent: AgentName, kind: str, **run_kw: object) -> TaskEnvelope:
    ws = tmp_path / "run" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    run = Run(title="t", request="build books API", workspace=str(ws), **run_kw)  # type: ignore[arg-type]
    phase = Phase.IMPLEMENTATION if agent is AgentName.ENGINEER else Phase.PRD
    task = Task(
        run_id=run.id, phase=phase, agent=agent, objective="do it", output_kind=kind, attempt=1
    )
    return TaskEnvelope(task=task, run=run, trace_id="trace_1", span_id="span_1")


def ctx(
    llm: FakeLLM,
    bus: FakeBus | None = None,
    cps: FakeCheckpoints | None = None,
    resume: bool = False,
) -> AgentContext:
    return AgentContext(
        llm=llm,
        bus=bus or FakeBus(),
        checkpoints=cps or FakeCheckpoints(),
        tools=file_tools(),
        resume=resume,
    )


async def test_tool_loop_writes_file_and_checkpoints_each_step(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    llm = FakeLLM(
        {
            AgentName.ENGINEER: [
                tool_call("write_file", {"path": "app.py", "content": "x = 1\n"}),
                json.dumps(CODE),
            ]
        }
    )
    bus, cps = FakeBus(), FakeCheckpoints()
    result = await Engineer().run(env, ctx(llm, bus, cps))
    assert result.status is ResultStatus.OK
    assert isinstance(result.artifact, CodeChange)
    assert (Path(env.run.workspace) / "app.py").read_text() == "x = 1\n"
    steps = cps.steps[(env.task.id, 1)]
    # initial, assistant tool call, tool result, final assistant
    assert [s for s, _ in steps] == [1, 2, 3, 4]
    assert EventName.TOOL_CALL_BEFORE not in bus.names()  # before events go to check(), not emit
    assert bus.names().count(EventName.TOOL_CALL_AFTER) == 1
    assert result.usage.input_tokens == sum(prompt_tokens(r) for r in llm.requests)
    # The tool specs went to the model.
    assert {t.name for t in llm.requests[0].tools} >= {"write_file", "read_file"}


async def test_bad_json_is_repaired_then_succeeds(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.PRODUCT, "prd")
    llm = FakeLLM({AgentName.PRODUCT: ["not json", '{"summary": "s"}', json.dumps(PRD)]})
    result = await Product().run(env, ctx(llm))
    assert result.status is ResultStatus.OK
    repair = llm.requests[2].messages[-1].content[0]
    assert isinstance(repair, TextBlock) and "did not validate as PRD" in repair.text


async def test_gives_up_after_two_repairs(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.PRODUCT, "prd")
    llm = FakeLLM({AgentName.PRODUCT: ["nope", "nope", "nope"]})
    result = await Product().run(env, ctx(llm))
    assert result.status is ResultStatus.ERROR
    assert result.error_type == "OutputValidationError"
    assert len(llm.requests) == 3


async def test_code_check_failure_goes_back_as_repair(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    llm = FakeLLM(
        {
            AgentName.ENGINEER: [
                json.dumps(CODE),  # claims app.py but never wrote it
                tool_call("write_file", {"path": "app.py", "content": "ok\n"}),
                json.dumps(CODE),
            ]
        }
    )
    result = await Engineer().run(env, ctx(llm))
    assert result.status is ResultStatus.OK
    repair = llm.requests[1].messages[-1].content[0]
    assert isinstance(repair, TextBlock) and "doesn't exist" in repair.text


async def test_blocked_and_unknown_tools_return_errors(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    bus = FakeBus()

    async def deny(event: object) -> HookOutcome:
        return HookOutcome(allow=False, reason="no writes today", rule_id="S5")

    bus.on_before(EventName.TOOL_CALL_BEFORE, deny)
    llm = FakeLLM(
        {
            AgentName.ENGINEER: [
                tool_call("write_file", {"path": "app.py", "content": "x"}),
                tool_call("search", {"q": "mcp"}),
                "give up",
                "give up",
                "give up",
            ]
        }
    )
    await Engineer().run(env, ctx(llm, bus))
    blocked = llm.requests[1].messages[-1].content[0]
    assert isinstance(blocked, ToolResultBlock) and blocked.is_error
    assert "blocked by rule S5" in blocked.content
    unknown = llm.requests[2].messages[-1].content[0]
    assert isinstance(unknown, ToolResultBlock) and unknown.is_error
    assert "unknown tool 'search'; available tools: " in unknown.content
    assert "read_file" in unknown.content and "write_file" in unknown.content
    assert not (Path(env.run.workspace) / "app.py").exists()


async def test_a_reply_after_the_budget_stop_starts_no_tool_turn(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    llm = FakeLLM({AgentName.ENGINEER: [tool_call("write_file", {"path": "a.py", "content": "x"})]})
    llm.stop_tools = BudgetExceeded("run has been calling models for 30.2 min", limit="time")
    checkpoints = FakeCheckpoints()
    result = await Engineer().run(env, ctx(llm, FakeBus(), checkpoints))
    assert result.status is ResultStatus.ERROR and result.error_type == "BudgetExceeded"
    assert not (Path(env.run.workspace) / "a.py").exists()
    # The reply isn't kept, so a resumed attempt asks again rather than finding
    # tool calls that look interrupted.
    saved = [m for steps in checkpoints.steps.values() for _, msgs in steps for m in msgs]
    assert all(m.role == "user" for m in saved)


async def test_auto_assumptions_mode_requires_assumptions(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.PRODUCT, "prd", review_mode=ReviewMode.AUTO_ASSUMPTIONS)
    asking = dict(PRD, questions=[{"id": "Q-1", "text": "Which DB?", "why": "storage"}])
    assuming = dict(
        PRD,
        questions=[{"id": "Q-1", "text": "Which DB?", "why": "storage", "assumption": "SQLite"}],
    )
    llm = FakeLLM({AgentName.PRODUCT: [json.dumps(asking), json.dumps(assuming)]})
    result = await Product().run(env, ctx(llm))
    assert result.status is ResultStatus.OK
    assert result.artifact is not None and result.artifact.questions[0].assumption == "SQLite"


async def test_questions_mode_returns_questions(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.PRODUCT, "prd", review_mode=ReviewMode.QUESTIONS_ONLY)
    asking = dict(PRD, questions=[{"id": "Q-1", "text": "Which DB?", "why": "storage"}])
    result = await Product().run(env, ctx(FakeLLM({AgentName.PRODUCT: [json.dumps(asking)]})))
    assert result.status is ResultStatus.QUESTIONS


async def test_resume_skips_unsafe_tool_and_repeats_safe_one(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    (Path(env.run.workspace) / "app.py").write_text("old\n")
    cps = FakeCheckpoints()
    pending = Message(
        role="assistant",
        content=[
            ToolUseBlock(id="c1", name="read_file", arguments={"path": "app.py"}),
            ToolUseBlock(
                id="c2", name="write_file", arguments={"path": "app.py", "content": "new\n"}
            ),
        ],
    )
    await cps.save_step(env.task.id, 1, 2, [Message.user("envelope"), pending])
    llm = FakeLLM({AgentName.ENGINEER: [json.dumps(CODE)]})
    result = await Engineer().run(env, ctx(llm, cps=cps, resume=True))
    assert result.status is ResultStatus.OK
    results = llm.requests[0].messages[-1].content
    assert isinstance(results[0], ToolResultBlock) and "old" in results[0].content
    assert isinstance(results[1], ToolResultBlock) and results[1].content == INTERRUPTED_NOTE
    assert (Path(env.run.workspace) / "app.py").read_text() == "old\n"
    # Steps continue from the checkpoint's number.
    assert cps.steps[(env.task.id, 1)][-1][0] == 4


async def test_compaction_runs_inside_the_loop(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    big = "line\n" * 2000
    (Path(env.run.workspace) / "big.txt").write_text(big)
    script = []
    for i in range(6):
        script.append(tool_call("read_file", {"path": "big.txt", "limit": 2000}, call_id=f"c{i}"))
    script += [
        tool_call("write_file", {"path": "app.py", "content": "x\n"}, call_id="w"),
        json.dumps(CODE),
    ]
    llm = FakeLLM(
        {
            AgentName.ENGINEER: lambda r: (
                '{"done": ["read big.txt"], "decisions": [], "remaining": ["write app.py"]}'
                if r.system.startswith("You compress")
                else script.pop(0)
            )
        },
        budget=3000,
    )
    bus = FakeBus()
    agent = Engineer(settings=LoopSettings(max_steps=60), compactor=Compactor(keep_last=2))
    result = await agent.run(env, ctx(llm, bus))
    assert result.status is ResultStatus.OK
    compactions = [e for e in bus.events if e.name is EventName.CONTEXT_COMPACTED]
    assert compactions
    steps = {s for e in compactions for s in e.data["steps"]}
    assert {"trim", "stub"} <= steps
    assert all(e.data["tokens_after"] < e.data["tokens_before"] for e in compactions)
    # The envelope survives every compaction.
    last_main = [r for r in llm.requests if not r.system.startswith("You compress")][-1]
    first = last_main.messages[0].content[0]
    assert isinstance(first, TextBlock) and first.text.startswith("# Task")


@pytest.mark.parametrize(
    ("route", "counted"), [(["anthropic", "ollama"], "ollama"), (["anthropic"], "anthropic")]
)
async def test_compaction_counts_for_the_provider_that_would_send_most(
    tmp_path: Path, route: list[str], counted: str
) -> None:
    from agent_team.llm.types import ThinkingBlock

    seen: list[str | None] = []

    class Spy(Compactor):
        """Records which provider each compaction check counted for."""

        def needs_compaction(self, *args: Any, **kwargs: Any) -> bool:
            """Note the provider, then decide as usual."""
            seen.append(kwargs.get("provider"))
            return super().needs_compaction(*args, **kwargs)

    env = envelope(tmp_path, AgentName.PRODUCT, "prd")
    thought = ThinkingBlock(provider="ollama", model="m", raw={"thinking": "t" * 4000})
    replies: list[Any] = [[thought, TextBlock(text="not json")], json.dumps(PRD)]
    llm = FakeLLM({AgentName.PRODUCT: replies})
    llm.providers = route
    await Product(compactor=Spy()).run(env, ctx(llm))
    # After Ollama wrote thinking, a call on a route that can fall back to Ollama may
    # send it all back, so it counts; Anthropic alone drops it.
    assert seen[-1] == counted


@pytest.mark.parametrize("budget", [10_000_000])
async def test_no_compaction_under_budget(tmp_path: Path, budget: int) -> None:
    env = envelope(tmp_path, AgentName.PRODUCT, "prd")
    bus = FakeBus()
    await Product().run(
        env, ctx(FakeLLM({AgentName.PRODUCT: [json.dumps(PRD)]}, budget=budget), bus)
    )
    assert EventName.CONTEXT_COMPACTED not in bus.names()


async def test_compactor_summarizes_when_stubs_are_not_enough(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    msgs = [Message.user("ENVELOPE " + "e" * 400)]
    for i in range(30):
        msgs.append(
            Message(
                role="assistant",
                content=[
                    TextBlock(text="thinking " * 40),
                    ToolUseBlock(id=f"c{i}", name="read_file", arguments={"path": f"f{i}.py"}),
                ],
            )
        )
        msgs.append(
            Message(role="user", content=[ToolResultBlock(tool_use_id=f"c{i}", content="x" * 3000)])
        )
    llm = FakeLLM({AgentName.ENGINEER: ['{"done": ["read 30 files"], "remaining": ["write"]}']})
    bus = FakeBus()
    out = await Compactor(keep_last=2).compact(
        system="sys",
        messages=msgs,
        budget=500,
        llm=llm,
        bus=bus,
        run=env.run,
        agent=AgentName.ENGINEER,
        task_id=env.task.id,
        attempt=1,
    )
    assert out.steps == ["trim", "stub", "summarize"]
    assert out.after < out.before
    head = out.messages[0]
    assert head.content[0] == msgs[0].content[0]  # envelope kept verbatim
    note = head.content[-1]
    assert isinstance(note, TextBlock) and "read 30 files" in note.text
    # Tail starts on an assistant turn so no tool result is orphaned.
    assert out.messages[1].role == "assistant"
    assert len(out.messages) == 3
    # Trimmed results were saved to disk first.
    assert list((Path(env.run.workspace).parent / "tool_results").iterdir())
    event = bus.events[-1]
    assert event.name is EventName.CONTEXT_COMPACTED and event.data["tokens_before"] == out.before


async def test_os_error_in_a_tool_goes_back_to_the_model(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    (Path(env.run.workspace) / "afile").write_text("")
    llm = FakeLLM(
        {
            AgentName.ENGINEER: [
                # The parent is a file, so creating the folder raises an OSError.
                tool_call("write_file", {"path": "afile/x.py", "content": "x"}),
                tool_call("write_file", {"path": "app.py", "content": "x = 1\n"}),
                json.dumps(CODE),
            ]
        }
    )
    result = await Engineer().run(env, ctx(llm))
    failed = llm.requests[1].messages[-1].content[0]
    assert isinstance(failed, ToolResultBlock) and failed.is_error
    assert "write_file failed" in failed.content
    assert result.status is ResultStatus.OK


async def test_compaction_keeps_thinking_and_assistant_turns_intact(tmp_path: Path) -> None:
    from agent_team.agents.compaction import estimate_tokens
    from agent_team.llm.types import ThinkingBlock

    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")

    def thought(i: int) -> ThinkingBlock:
        raw = {"type": "thinking", "thinking": f"SECRET-{i} " * 50, "signature": "sig" * 40}
        return ThinkingBlock(provider="anthropic", model="m", raw=raw)

    msgs = [Message.user("ENVELOPE")]
    for i in range(12):
        msgs.append(
            Message(
                role="assistant",
                content=[
                    thought(i),
                    ToolUseBlock(id=f"c{i}", name="read_file", arguments={"path": f"f{i}.py"}),
                ],
            )
        )
        msgs.append(
            Message(role="user", content=[ToolResultBlock(tool_use_id=f"c{i}", content="x" * 3000)])
        )
    plain = [
        m.model_copy(update={"content": m.content[1:]}) if m.role == "assistant" else m
        for m in msgs
    ]
    assert estimate_tokens("", msgs) > estimate_tokens("", plain)  # thinking counts at raw size

    llm = FakeLLM({AgentName.ENGINEER: ['{"done": ["read 12 files"], "remaining": ["write"]}']})
    out = await Compactor(keep_last=3).compact(
        system="sys",
        messages=msgs,
        budget=800,
        llm=llm,
        bus=FakeBus(),
        run=env.run,
        agent=AgentName.ENGINEER,
        task_id=env.task.id,
        attempt=1,
    )
    assert out.steps == ["trim", "stub", "summarize"]
    # Assistant turns are never edited: the kept ones are the original objects. (The
    # API may still drop their thinking, because earlier history changed.)
    kept = [m for m in out.messages if m.role == "assistant"]
    assert kept and all(any(m is k for m in msgs) for k in kept)
    summary = llm.requests[0]
    assert "SECRET" not in summary.messages[0].content[0].text  # type: ignore[union-attr]
    assert summary.max_tokens == 16_000  # room for thinking on the agent's route


async def test_repeated_summaries_replace_the_progress_note(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")

    def history(n: int) -> list[Message]:
        out: list[Message] = []
        for i in range(n):
            out.append(
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id=f"c{i}", name="read_file", arguments={"path": "f"})],
                )
            )
            out.append(
                Message(
                    role="user", content=[ToolResultBlock(tool_use_id=f"c{i}", content="y" * 500)]
                )
            )
        return out

    llm = FakeLLM({AgentName.ENGINEER: lambda r: '{"done": ["NOTE"], "remaining": []}'})
    compactor = Compactor(keep_last=2)
    kwargs: dict[str, Any] = {
        "system": "sys",
        "budget": 300,
        "llm": llm,
        "bus": FakeBus(),
        "run": env.run,
        "agent": AgentName.ENGINEER,
        "task_id": env.task.id,
        "attempt": 1,
    }
    first = await compactor.compact(messages=[Message.user("ENVELOPE"), *history(6)], **kwargs)
    second = await compactor.compact(messages=[*first.messages, *history(6)], **kwargs)
    assert "summarize" in second.steps
    head = second.messages[0].content
    assert len(head) == 2 and head[0] == TextBlock(text="ENVELOPE")
    # The first note went into the second summary rather than piling up.
    assert "NOTE" in llm.requests[-1].messages[0].content[0].text  # type: ignore[union-attr]


def test_tool_specs_count_toward_the_estimate() -> None:
    from agent_team.agents.compaction import estimate_tokens

    msgs = [Message.user("hi")]
    specs = [t.spec for t in file_tools().values()]
    assert estimate_tokens("s", msgs, specs) > estimate_tokens("s", msgs) + 100


class CountingLLM(FakeLLM):
    """A fake whose calls report the prompt sizes in ``reported`` (the last one
    repeats) and are served by ``providers`` in a cycle."""

    def __init__(
        self,
        script: dict[AgentName, Any],
        *,
        reported: list[int],
        providers: list[str],
        budget: int = 10_000,
    ) -> None:
        super().__init__(script, budget=budget)
        self.reported = reported
        self.providers = providers

    async def complete(self, request: LLMRequest) -> LLMResponse:
        response = await super().complete(request)
        n = len(self.requests) - 1
        return response.model_copy(
            update={
                "usage": Usage(input_tokens=self.reported[min(n, len(self.reported) - 1)]),
                "provider": self.providers[n % len(self.providers)],
            }
        )


def _reads(n: int) -> list[Any]:
    """``n`` small tool calls, then the final output."""
    calls: list[Any] = [tool_call("list_dir", {"path": "."}, call_id=f"c{i}") for i in range(n)]
    return [*calls, json.dumps(CODE)]


async def test_compaction_anchors_on_the_reported_prompt_size(tmp_path: Path) -> None:
    # The history is tiny by estimate, but the provider says it is near the budget.
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    script = _reads(3)
    llm = CountingLLM(
        {
            AgentName.ENGINEER: lambda r: (
                _summary(r) or (script.pop(0) if script else json.dumps(CODE))
            )
        },
        reported=[9_000],
        providers=["p"],
    )
    bus = FakeBus()
    await Engineer(compactor=Compactor(keep_last=2)).run(env, ctx(llm, bus))
    compactions = [e for e in bus.events if e.name is EventName.CONTEXT_COMPACTED]
    assert compactions and compactions[0].data["tokens_before"] > 9_000


async def test_a_provider_switch_drops_the_anchor(tmp_path: Path) -> None:
    # After the first call every call is served by another provider than the one
    # before it (fallbacks), so their large counts are not anchored on and the small
    # history is estimated in full.
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    script = _reads(3)
    llm = CountingLLM(
        {
            AgentName.ENGINEER: lambda r: (
                _summary(r) or (script.pop(0) if script else json.dumps(CODE))
            )
        },
        reported=[100, 18_000],
        providers=["p", "q"],
        # Room for the full estimate of the history, which a large count also
        # calibrates (per model) to the fewest chars per token; the anchor would not fit.
        budget=20_000,
    )
    bus = FakeBus()
    await Engineer(compactor=Compactor(keep_last=2)).run(env, ctx(llm, bus))
    assert EventName.CONTEXT_COMPACTED not in bus.names()


def _summary(request: LLMRequest) -> str | None:
    """The summarize step's reply, for compaction requests."""
    if request.system.startswith("You compress"):
        return '{"done": [], "decisions": [], "remaining": []}'
    return None


async def test_a_tool_call_cut_off_at_the_token_limit_is_not_run(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    replies: list[tuple[list[ContentBlock], str]] = [
        (
            [
                ToolUseBlock(
                    id="c1", name="write_file", arguments={"path": "a.py", "content": "a\n"}
                ),
                ToolUseBlock(
                    id="c2", name="write_file", arguments={"path": "app.py", "content": "x"}
                ),
            ],
            "max_tokens",
        ),
        (
            [
                ToolUseBlock(
                    id="c3", name="write_file", arguments={"path": "app.py", "content": "x\n"}
                )
            ],
            "tool_use",
        ),
        ([TextBlock(text=json.dumps(CODE))], "end_turn"),
    ]

    class CutOff(FakeLLM):
        """Replies in order, the first one cut off at the output token limit."""

        async def complete(self, request: LLMRequest) -> LLMResponse:
            """The next scripted reply."""
            self.requests.append(request)
            blocks, stop = replies.pop(0)
            return LLMResponse(
                content=blocks,
                stop_reason=stop,  # type: ignore[arg-type]
                usage=Usage(),
                provider="fake",
                model="fake-1",
            )

    llm = CutOff()
    result = await Engineer().run(env, ctx(llm))
    assert result.status is ResultStatus.OK
    ws = Path(env.run.workspace)
    assert (ws / "a.py").read_text() == "a\n"  # the complete call ran
    results = [b for b in llm.requests[1].messages[-1].content if isinstance(b, ToolResultBlock)]
    assert [r.tool_use_id for r in results] == ["c1", "c2"]
    assert results[1].is_error and "output token limit" in results[1].content
    assert (ws / "app.py").read_text() == "x\n"  # written by the retried call


async def test_repair_prompt_allows_a_tool_call_first(tmp_path: Path) -> None:
    env = envelope(tmp_path, AgentName.ENGINEER, "code_change")
    llm = FakeLLM(
        {
            AgentName.ENGINEER: [
                json.dumps({"summary": "nothing yet", "files_written": []}),
                tool_call("write_file", {"path": "app.py", "content": "x = 1\n"}),
                json.dumps(CODE),
            ]
        }
    )
    result = await Engineer().run(env, ctx(llm))
    assert result.status is ResultStatus.OK
    repair = llm.requests[1].messages[-1].content[0]
    text = getattr(repair, "text", "")
    assert (
        "write the spec's files with write_file" in text
        and "tool call first, make that call" in text
    )


def _reading_loop(
    tmp_path: Path, llm: FakeLLM, max_steps: int, *, final_reply: bool = False
) -> ToolLoop[ResearchBrief]:
    """A Researcher loop over ``llm`` with the file tools, ``max_steps`` long."""
    env = envelope(tmp_path, AgentName.RESEARCHER, "research_brief")
    (Path(env.run.workspace) / "notes.md").write_text("notes\n")
    bus = FakeBus()
    return ToolLoop(
        agent=AgentName.RESEARCHER,
        system="research",
        output_model=ResearchBrief,
        llm=llm,
        bus=bus,
        checkpoints=FakeCheckpoints(),
        tools=file_tools(),
        tool_ctx=ToolContext(run=env.run, task=env.task, bus=bus),
        run=env.run,
        task_id=env.task.id,
        attempt=1,
        settings=LoopSettings(max_steps=max_steps, final_reply=final_reply),
    )


def _notes(request: LLMRequest) -> list[str]:
    """The step notes in a request's last message."""
    last = request.messages[-1].content
    return [b.text for b in last if isinstance(b, TextBlock) and "turn(s) left" in b.text]


async def test_the_model_is_told_to_write_its_output_when_few_steps_are_left(
    tmp_path: Path,
) -> None:
    read = tool_call("read_file", {"path": "notes.md"})
    llm = FakeLLM({AgentName.RESEARCHER: lambda request: read})
    outcome = await _reading_loop(tmp_path, llm, max_steps=20).run_loop(Message.user("go"))
    assert outcome.output is None and "20 steps" in str(outcome.error)
    told = [i for i, r in enumerate(llm.requests) if _notes(r)]
    # 20 steps: 15% is 3 left, after the 9th tool round (step 17 saved), so the 9th call
    # (index 8) is told it has one turn left; the note is given once.
    assert told == [8]
    assert _notes(llm.requests[8]) == [STEPS_LEFT_NOTE.format(turns=1)]


async def test_a_loop_near_its_step_limit_gets_a_reply_only_turn(tmp_path: Path) -> None:
    """A model that would keep reading is called with tool choice "none" once only the
    steps for a reply and its repairs are left, so the attempt ends with an answer."""
    read = tool_call("read_file", {"path": "notes.md"})

    def researcher(request: LLMRequest) -> Any:
        """Read again whenever a tool may be called; reply only when it can't."""
        return json.dumps(BRIEF) if request.tool_choice == "none" else read

    llm = FakeLLM({AgentName.RESEARCHER: researcher})
    loop = _reading_loop(tmp_path, llm, max_steps=15, final_reply=True)
    outcome = await loop.run_loop(Message.user("go"))
    assert outcome.output is not None, outcome.error
    choices = [r.tool_choice for r in llm.requests]
    # 15 steps and 2 repairs: forced once 6 are left, after the 4th tool round (step 9).
    assert choices == ["auto"] * 4 + ["none"]
    last = llm.requests[-1].messages[-1].content
    assert [b.text for b in last if isinstance(b, TextBlock)] == [FINAL_REPLY_NOTE]
    assert llm.requests[-1].tools  # still sent: the history holds tool calls


async def test_without_final_reply_every_call_may_use_tools(tmp_path: Path) -> None:
    read = tool_call("read_file", {"path": "notes.md"})
    llm = FakeLLM({AgentName.RESEARCHER: lambda request: read})
    outcome = await _reading_loop(tmp_path, llm, max_steps=9).run_loop(Message.user("go"))
    assert outcome.output is None
    assert {r.tool_choice for r in llm.requests} == {"auto"}


async def test_an_ollama_tool_call_error_takes_no_loop_step(tmp_path: Path) -> None:
    """Ollama's "error parsing tool call" 500 is retried once inside the call, so the
    loop sees one model step, not several."""
    garbled = error(500, "error parsing tool call: raw='We need to fetch the docs'")
    provider, server = ollama(garbled, garbled, garbled)
    config = make_config(
        defaults={"provider": "ollama", "model": "gpt-oss:20b", "context_budget": 20_000},
        fallbacks=["ollama", "fake"],
        fallback_models={"fake": "fake-small"},
    )
    brief = json.dumps(BRIEF)
    gw = Gateway(config, {"ollama": provider, "fake": FakeProvider([brief])}, sleep=Sleeps())
    loop = _reading_loop(tmp_path, gw, max_steps=20)  # type: ignore[arg-type]
    outcome = await loop.run_loop(Message.user("go"))
    assert outcome.output is not None
    assert outcome.steps == 2  # the first message and the one reply
    assert len(server.chats) == 2  # the garbled call and its one retry
