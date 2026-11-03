import asyncio
import sqlite3
from pathlib import Path

import pytest

from agent_team.core.artifacts import (
    PRD,
    AcceptanceCriterion,
    ArtifactRef,
    Ask,
    Plan,
    Requirement,
)
from agent_team.core.db import Database
from agent_team.core.errors import InvalidTransition
from agent_team.core.events import InMemoryEventBus
from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
    Event,
    EventName,
    Feedback,
    FeedbackSource,
    Phase,
    RuleScope,
    Run,
    RunStatus,
    Task,
    TaskStatus,
    Verdict,
)
from agent_team.core.stores import (
    SqliteArtifactStore,
    SqliteCheckpointStore,
    SqliteDecisionLog,
    SqliteFeedbackStore,
    SqliteTaskStore,
    SqliteVerdictStore,
)
from agent_team.llm.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "agent_team.db")


@pytest.fixture
def events() -> list[Event]:
    return []


@pytest.fixture
def bus(events: list[Event]) -> InMemoryEventBus:
    b = InMemoryEventBus()

    async def collect(e: Event) -> None:
        events.append(e)

    b.on_after(None, collect)
    return b


def make_run(tmp_path: Path) -> Run:
    return Run(title="Inventory API", request="build it", workspace=str(tmp_path / "ws"))


def make_task(run: Run, **kw: object) -> Task:
    fields: dict[str, object] = {
        "run_id": run.id,
        "phase": Phase.PRD,
        "agent": AgentName.PRODUCT,
        "objective": "write the PRD",
        "output_kind": "prd",
    }
    fields.update(kw)
    return Task.model_validate(fields)


def make_prd(run_id: str, built_from: list[ArtifactRef] | None = None) -> PRD:
    return PRD(
        run_id=run_id,
        task_id="task_1",
        summary="s",
        title="Inventory",
        built_from=built_from or [],
        requirements=[
            Requirement(
                id="REQ-1",
                text="list books",
                asks=["ASK-1"],
                acceptance_criteria=[AcceptanceCriterion(id="AC-1.1", text="GET /books")],
            )
        ],
    )


def test_database_uses_wal(db: Database) -> None:
    assert db.journal_mode == "wal"


async def test_runs_round_trip(db: Database, tmp_path: Path) -> None:
    store = SqliteTaskStore(db)
    run = await store.create_run(make_run(tmp_path))
    assert await store.get_run(run.id) == run
    updated = await store.update_run(run.model_copy(update={"status": RunStatus.PAUSED}))
    assert updated.updated_at >= run.updated_at
    assert (await store.get_run(run.id)).status is RunStatus.PAUSED
    assert [r.id for r in await store.list_runs()] == [run.id]
    with pytest.raises(KeyError):
        await store.get_run("run_missing")


async def test_transition_checks_and_emits(
    db: Database, bus: InMemoryEventBus, events: list[Event], tmp_path: Path
) -> None:
    store = SqliteTaskStore(db, bus)
    run = await store.create_run(make_run(tmp_path))
    task = await store.create_task(make_task(run))

    with pytest.raises(InvalidTransition):
        await store.transition(task.id, TaskStatus.DONE, reason="skip ahead")

    await store.transition(task.id, TaskStatus.READY, reason="deps done")
    running = await store.transition(task.id, TaskStatus.RUNNING, reason="picked up")
    assert running.started_at is not None and running.finished_at is None
    await store.transition(task.id, TaskStatus.AUDITING, reason="output ready")
    done = await store.transition(task.id, TaskStatus.DONE, reason="rules pass")
    assert done.finished_at is not None
    assert (await store.get_task(task.id)).status is TaskStatus.DONE

    changes = [e for e in events if e.name is EventName.TASK_STATE_CHANGED]
    assert [(e.data["from"], e.data["to"]) for e in changes] == [
        ("pending", "ready"),
        ("ready", "running"),
        ("running", "auditing"),
        ("auditing", "done"),
    ]
    assert changes[-1].data["reason"] == "rules pass"
    assert changes[-1].task_id == task.id


async def test_update_task_refuses_status_change(db: Database, tmp_path: Path) -> None:
    store = SqliteTaskStore(db)
    run = await store.create_run(make_run(tmp_path))
    task = await store.create_task(make_task(run))
    await store.update_task(task.model_copy(update={"attempt": 1, "model": "m"}))
    assert (await store.get_task(task.id)).attempt == 1
    with pytest.raises(InvalidTransition):
        await store.update_task(task.model_copy(update={"status": TaskStatus.DONE}))


async def test_list_and_ready_tasks(db: Database, tmp_path: Path) -> None:
    store = SqliteTaskStore(db)
    run = await store.create_run(make_run(tmp_path))
    a = await store.create_task(make_task(run, phase=Phase.PLANNING, output_kind="plan"))
    b = await store.create_task(make_task(run, depends_on=[a.id]))
    c = await store.create_task(make_task(run, depends_on=[a.id, b.id]))

    assert [t.id for t in await store.ready_tasks(run.id)] == [a.id]
    for to in (TaskStatus.RUNNING, TaskStatus.AUDITING, TaskStatus.DONE):
        await store.transition(a.id, to, reason="test")
    assert [t.id for t in await store.ready_tasks(run.id)] == [b.id]
    assert [t.id for t in await store.list_tasks(run.id, phase=Phase.PRD)] == [b.id, c.id]
    assert [t.id for t in await store.list_tasks(run.id, status=TaskStatus.PENDING)] == [c.id]


async def test_artifacts_version_files_and_lineage(db: Database, tmp_path: Path) -> None:
    store = SqliteArtifactStore(db, tmp_path / "runs")
    plan = Plan(
        run_id="run_1",
        task_id="t",
        summary="s",
        request_kind="build",
        asks=[Ask(id="ASK-1", text="list books")],
    )
    plan_ref = await store.save(plan)
    assert plan_ref == ArtifactRef(kind="plan", version=1)

    prd1 = await store.save(make_prd("run_1", [plan_ref]))
    prd2 = await store.save(make_prd("run_1", [plan_ref]))
    assert (prd1.version, prd2.version) == (1, 2)
    assert (tmp_path / "runs/run_1/artifacts/prd.v2.json").exists()

    loaded = await store.load("run_1", prd1)
    assert isinstance(loaded, PRD) and loaded.version == 1
    latest = await store.latest("run_1", "prd")
    assert latest is not None and latest.version == 2
    assert await store.latest("run_1", "spec") is None
    with pytest.raises(KeyError):
        await store.load("run_1", ArtifactRef(kind="spec", version=1))

    spec_like = make_prd("run_1", [prd2]).model_copy(update={"title": "child"})
    grandchild = await store.save(spec_like)
    deps = await store.dependents("run_1", plan_ref)
    assert set(map(str, deps)) == {"prd@v1", "prd@v2", str(grandchild)}
    assert await store.dependents("run_1", prd1) == []
    # versions are per run
    assert (await store.save(make_prd("run_2"))).version == 1


async def test_artifact_needs_run_id(db: Database, tmp_path: Path) -> None:
    store = SqliteArtifactStore(db, tmp_path)
    with pytest.raises(ValueError):
        await store.save(make_prd(""))


async def test_checkpoints_keep_latest_and_offload_large_results(
    db: Database, tmp_path: Path
) -> None:
    tasks = SqliteTaskStore(db)
    run = await tasks.create_run(make_run(tmp_path))
    task = await tasks.create_task(make_task(run))
    store = SqliteCheckpointStore(db, tmp_path / "runs", inline_limit=100, preview_chars=50)

    assert await store.load_latest(task.id, 1) is None
    big = "x" * 500
    msgs = [
        Message.user("go"),
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="read", arguments={})]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content=big)]),
    ]
    await store.save_step(task.id, 1, 1, msgs[:2])
    await store.save_step(task.id, 1, 2, msgs)
    await store.save_step(task.id, 2, 1, msgs[:1])

    loaded = await store.load_latest(task.id, 1)
    assert loaded is not None
    step, back = loaded
    assert step == 2 and len(back) == 3
    result = back[2].content[0]
    assert isinstance(result, ToolResultBlock) and result.stored_at is not None
    assert await asyncio.to_thread(Path(result.stored_at).read_text) == big
    assert str(tmp_path / "runs" / run.id / "blobs") in result.stored_at
    assert len(result.content) < len(big)
    assert isinstance(back[0].content[0], TextBlock)
    # The full result is written once even though steps keep resaving the history.
    await store.save_step(task.id, 1, 3, msgs)
    blobs = await asyncio.to_thread(lambda: list((tmp_path / "runs" / run.id / "blobs").iterdir()))
    assert len(blobs) == 1


async def test_checkpoints_round_trip_thinking_blocks_unchanged(
    db: Database, tmp_path: Path
) -> None:
    tasks = SqliteTaskStore(db)
    run = await tasks.create_run(make_run(tmp_path))
    task = await tasks.create_task(make_task(run))
    store = SqliteCheckpointStore(db, tmp_path / "runs")
    raw = {"type": "thinking", "thinking": "", "signature": "c2lnbmF0dXJl"}
    redacted = {"type": "redacted_thinking", "data": "b3BhcXVl"}
    msgs = [
        Message.user("go"),
        Message(
            role="assistant",
            content=[
                ThinkingBlock(provider="anthropic", model="claude-haiku-5-5", raw=raw),
                ThinkingBlock(provider="anthropic", model="claude-haiku-5-5", raw=redacted),
                ToolUseBlock(id="c1", name="read", arguments={}),
            ],
        ),
    ]
    await store.save_step(task.id, 1, 1, msgs)
    loaded = await store.load_latest(task.id, 1)
    assert loaded is not None and loaded[1] == msgs
    first, second = loaded[1][1].content[:2]
    assert isinstance(first, ThinkingBlock) and first.raw == raw
    assert isinstance(second, ThinkingBlock) and second.raw == redacted


async def test_decision_log_records_and_emits(
    db: Database, bus: InMemoryEventBus, events: list[Event]
) -> None:
    log = SqliteDecisionLog(db, bus)
    d1 = Decision(run_id="r", kind=DecisionKind.PLAN, choice="4 tasks", reason="r", made_by="model")
    d2 = Decision(
        run_id="r", kind=DecisionKind.RETRY, choice="retry", reason="r", made_by="code", rule="x"
    )
    await log.record(d1)
    await log.record(d2)
    await log.record(
        Decision(run_id="other", kind=DecisionKind.STOP, choice="c", reason="r", made_by="code")
    )
    assert [d.id for d in await log.list("r")] == [d1.id, d2.id]
    emitted = [e for e in events if e.name is EventName.DECISION]
    assert emitted[0].data["id"] == d1.id and emitted[0].data["kind"] == "plan"


async def test_feedback_by_artifact_version_and_task(db: Database) -> None:
    store = SqliteFeedbackStore(db)
    pkg_a = ArtifactRef(kind="code_change", version=1)
    pkg_b = ArtifactRef(kind="code_change", version=2)
    f1 = Feedback(artifact=pkg_a, source=FeedbackSource.USER_COMMENT, text="add auth")
    f2 = Feedback(artifact=None, task_id="t_new", source=FeedbackSource.USER_ANSWER, text="yes")
    f3 = Feedback(artifact=pkg_b, source=FeedbackSource.AUDITOR, text="REQ-2 untested")
    f4 = Feedback(artifact=pkg_a, task_id="t_a", source=FeedbackSource.AUDITOR, text="S2")
    for f in (f1, f2, f3, f4):
        await store.add("r", f)
    # One work package's feedback never reaches its sibling of the same kind.
    found = await store.for_work("r", refs=[pkg_a], task_ids=["t_a"])
    assert [f.id for f in found] == [f1.id, f4.id]
    assert [f.id for f in await store.for_work("r", refs=[pkg_b], task_ids=[])] == [f3.id]
    # Feedback for a task with no output yet is found by the task.
    assert [f.id for f in await store.for_work("r", refs=[], task_ids=["t_new"])] == [f2.id]
    assert await store.for_work("r", refs=[], task_ids=[]) == []
    assert await store.for_work("other", refs=[pkg_a], task_ids=["t_a"]) == []


async def test_verdicts_per_task(db: Database) -> None:
    store = SqliteVerdictStore(db)
    assert await store.latest_for_task("t") is None
    v1 = Verdict(run_id="r", scope=RuleScope.TASK, task_id="t", passed=False)
    v2 = Verdict(run_id="r", scope=RuleScope.TASK, task_id="t", passed=True, summary="all good")
    await store.save(v1)
    await store.save(v2)
    await store.save(Verdict(run_id="r", scope=RuleScope.TASK, task_id="u", passed=True))
    latest = await store.latest_for_task("t")
    assert latest is not None and latest.id == v2.id
    assert latest == v2 and latest.summary == "all good"
    assert [v.id for v in await store.for_task("t")] == [v1.id, v2.id]


async def test_store_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    first = SqliteTaskStore(Database(path))
    run = await first.create_run(make_run(tmp_path))
    first.db.close()
    again = SqliteTaskStore(Database(path))
    assert (await again.get_run(run.id)).id == run.id


async def test_reused_tool_call_ids_keep_their_own_blobs(db: Database, tmp_path: Path) -> None:
    store = SqliteCheckpointStore(db, tmp_path / "runs", inline_limit=10, preview_chars=5)
    first = Message(role="user", content=[ToolResultBlock(tool_use_id="call_0", content="a" * 50)])
    second = Message(role="user", content=[ToolResultBlock(tool_use_id="call_0", content="b" * 50)])
    await store.save_step("task_x", 1, 1, [first, second])
    loaded = await store.load_latest("task_x", 1)
    assert loaded is not None
    paths = [b.stored_at for m in loaded[1] for b in m.content if isinstance(b, ToolResultBlock)]
    assert paths[0] != paths[1]
    texts = [await asyncio.to_thread(Path(str(p)).read_text) for p in paths]
    assert texts == ["a" * 50, "b" * 50]


async def test_two_stores_on_one_file_get_distinct_versions(tmp_path: Path) -> None:
    path = tmp_path / "shared.db"
    a = SqliteArtifactStore(Database(path), tmp_path / "runs")
    b = SqliteArtifactStore(Database(path), tmp_path / "runs")
    refs = await asyncio.gather(*(s.save(make_prd("run_1")) for s in (a, b, a, b)))
    assert sorted(r.version for r in refs) == [1, 2, 3, 4]


async def test_failed_transaction_rolls_back_and_db_stays_usable(db: Database) -> None:
    def boom(conn: sqlite3.Connection) -> None:
        conn.execute("INSERT INTO runs (id, status, created_at, data) VALUES ('x', 'a', 't', '{}')")
        raise RuntimeError("fail mid-transaction")

    with pytest.raises(RuntimeError):
        await db.transaction(boom)
    assert await db.fetchone("SELECT id FROM runs WHERE id = 'x'") is None
    await db.execute("INSERT INTO runs (id, status, created_at, data) VALUES ('y', 'a', 't', '{}')")
    assert await db.fetchone("SELECT id FROM runs WHERE id = 'y'") is not None
