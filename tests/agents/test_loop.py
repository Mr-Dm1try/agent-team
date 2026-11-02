from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent_team.agents import Engineer, Product
from agent_team.agents.compaction import Compactor
from agent_team.agents.loop import INTERRUPTED_NOTE, LoopSettings
from agent_team.core.artifacts import CodeChange
from agent_team.core.interfaces import AgentContext
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
)
from agent_team.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from agent_team.tools import file_tools
from agents.fakes import FakeBus, FakeCheckpoints, FakeLLM, tool_call
from agents.scenario import CODE, PRD


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
    assert result.usage.input_tokens == 20
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
                tool_call("teleport", {}),
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
    assert isinstance(unknown, ToolResultBlock) and "unknown tool" in unknown.content
    assert not (Path(env.run.workspace) / "app.py").exists()


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
        budget=5000,
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
