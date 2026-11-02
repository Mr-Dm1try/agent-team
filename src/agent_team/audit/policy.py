"""The verdict policy: turns a verdict into Decisions for the orchestrator to apply.

Code makes every choice here, so each Decision has ``made_by="code"`` and names the
policy branch in ``rule``. The branches follow docs/DESIGN.md, "What a verdict
triggers":

1. Pass: no decisions; the orchestrator marks the task done.
2. Fail with attempts left: RETRY as a revision of the last output, findings attached.
   If the retry fails the same rules on the same items as the attempt before, that is
   NO_PROGRESS and it goes straight to escalation.
3. Fail with no attempts left: ESCALATE. A finding that blames an upstream artifact
   reopens the earliest one at fault (UPSTREAM_RERUN); a fault in the task's own
   output gets one STRONGER_MODEL attempt before the user is asked.
4. A phase gate fails: each blocker goes to the task that produced the item at fault
   (GATE_FIX, a revision), and the tasks after it in the phase re-run too. QA's
   blocking issues become Engineer fixes for the files named, then a fresh QA review.
5. Still failing after an upstream re-run, or past a cap: ASK_USER.

Two more come from the auditor section: a dangerous agent action (S5, S6, S7) STOPs
the run and asks the user, since it already happened, and this wins over everything
else, including a fail-closed verdict that still carries such a helper hit; and a
fail-closed verdict (an ``AUDIT`` finding) escalates and asks, since the expert isn't
at fault.

Extra ``inputs`` keys beyond the contract's: ``finding_ids`` on most kinds,
``previous_output`` on RETRY, ``phase``/``gate``/``producer_task_id`` on
UPSTREAM_RERUN and ``rerun_after`` (task ids) on GATE_FIX.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_team.audit.auditor import AUDIT_RULE
from agent_team.audit.evidence import KIND_ORDER, KIND_PHASE
from agent_team.core.artifacts import Artifact, ArtifactRef, CodeChange
from agent_team.core.interfaces import ArtifactStore, DecisionLog, TaskStore
from agent_team.core.models import (
    AgentName,
    Decision,
    DecisionKind,
    Finding,
    Phase,
    Run,
    Task,
    TaskStatus,
    Verdict,
)

DANGEROUS_ACTION_RULES = frozenset({"S5", "S6", "S7"})


@dataclass(frozen=True)
class PolicyLimits:
    """Caps the policy enforces beyond each task's own ``max_attempts``."""

    max_upstream_reruns: int = 2
    stronger_model: str = "claude-sonnet-5-5"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PolicyLimits:
        """Reads ``limits.max_upstream_reruns`` and ``stronger_model.model``."""
        limits = config.get("limits") or {}
        stronger = config.get("stronger_model") or {}
        return cls(
            max_upstream_reruns=int(limits.get("max_upstream_reruns", cls.max_upstream_reruns)),
            stronger_model=str(stronger.get("model", cls.stronger_model)),
        )


def _decision(
    run: Run,
    kind: DecisionKind,
    *,
    task_id: str | None,
    choice: str,
    reason: str,
    rule: str,
    **inputs: Any,
) -> Decision:
    return Decision(
        run_id=run.id,
        task_id=task_id,
        kind=kind,
        inputs={"task_id": task_id, **inputs},
        choice=choice,
        reason=reason,
        made_by="code",
        rule=rule,
    )


def _blocker_keys(verdict: Verdict) -> frozenset[tuple[str, str, str]]:
    """What a blocker is about, ignoring versions, so two attempts can be compared."""
    return frozenset(
        (f.rule_id, (f.item or "").strip().lower(), f.artifact.kind if f.artifact else "")
        for f in verdict.blockers
    )


def _summary(findings: Iterable[Finding], limit: int = 5) -> str:
    items = [f"{f.rule_id}" + (f" {f.item}" if f.item else "") + f": {f.message}" for f in findings]
    more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return "; ".join(items[:limit]) + more


def _ids(findings: Iterable[Finding]) -> list[str]:
    return [f.id for f in findings]


def _lineage(task: Task) -> set[str]:
    return {task.id} | ({task.parent_task_id} if task.parent_task_id else set())


def _earliest_upstream(findings: Iterable[Finding], below: int) -> ArtifactRef | None:
    upstream = [
        f.artifact
        for f in findings
        if f.artifact is not None and KIND_ORDER.get(f.artifact.kind, 99) < below
    ]
    if not upstream:
        return None
    return min(upstream, key=lambda r: (KIND_ORDER.get(r.kind, 99), r.version))


def _producer(ref: ArtifactRef, tasks: Iterable[Task]) -> Task | None:
    exact = [t for t in tasks if t.output == ref]
    if exact:
        return exact[-1]
    same_kind = [t for t in tasks if t.output is not None and t.output.kind == ref.kind]
    return max(same_kind, key=lambda t: t.created_at) if same_kind else None


# --- task verdicts -------------------------------------------------------------------


def decide_task(
    run: Run,
    task: Task,
    verdict: Verdict,
    history: list[Verdict],
    past: list[Decision],
    limits: PolicyLimits,
    tasks: list[Task] | None = None,
) -> list[Decision]:
    """Decisions for one task verdict. ``history`` is the task's earlier verdicts,
    oldest first; ``past`` is the run's decision log so far."""
    if verdict.passed and not verdict.blockers:
        return []
    blockers = verdict.blockers
    fids = _ids(blockers)
    out: list[Decision] = []

    dangerous = [f for f in blockers if f.rule_id in DANGEROUS_ACTION_RULES]
    if dangerous:
        return [
            _decision(
                run,
                DecisionKind.STOP,
                task_id=task.id,
                choice="stop the run",
                reason="an agent took a dangerous action, which already happened: "
                + _summary(dangerous),
                rule="policy.dangerous_action",
                finding_ids=_ids(dangerous),
            ),
            _ask(
                run,
                task.id,
                f"Task {task.id} ({task.agent.value}) did something unsafe: "
                f"{_summary(dangerous)}. The run is stopped. How should I proceed?",
                "policy.dangerous_action",
                _ids(dangerous),
            ),
        ]

    if any(f.rule_id == AUDIT_RULE for f in blockers):
        reason = "the auditor produced no valid verdict, so the step stays unpassed"
        return [
            _decision(
                run,
                DecisionKind.ESCALATE,
                task_id=task.id,
                choice="escalate",
                reason=reason,
                rule="policy.fail_closed",
                finding_ids=fids,
            ),
            _ask(
                run,
                task.id,
                f"The audit of task {task.id} failed: {_summary(blockers)}. "
                "Re-run the audit, or how should I proceed?",
                "policy.fail_closed",
                fids,
            ),
        ]

    no_progress = False
    earlier = [v for v in history if v.id != verdict.id]
    if earlier and not earlier[-1].passed and _blocker_keys(earlier[-1]) == _blocker_keys(verdict):
        no_progress = True
        out.append(
            _decision(
                run,
                DecisionKind.NO_PROGRESS,
                task_id=task.id,
                choice="escalate now",
                reason="the retry failed the same rules on the same items: " + _summary(blockers),
                rule="policy.no_progress",
                finding_ids=fids,
            )
        )

    if not no_progress and task.attempt < task.max_attempts:
        out.append(
            _decision(
                run,
                DecisionKind.RETRY,
                task_id=task.id,
                choice=f"retry as a revision (attempt {task.attempt + 1} of {task.max_attempts})",
                reason="audit failed with attempts left: " + _summary(blockers),
                rule="policy.retry_as_revision",
                finding_ids=fids,
                previous_output=str(task.output) if task.output else None,
            )
        )
        return out

    out.append(
        _decision(
            run,
            DecisionKind.ESCALATE,
            task_id=task.id,
            choice="escalate",
            reason=("no progress" if no_progress else "no attempts left")
            + ": "
            + _summary(blockers),
            rule="policy.escalate",
            finding_ids=fids,
        )
    )

    lineage = _lineage(task)
    own_rank = KIND_ORDER.get(task.output_kind, 99)
    upstream = _earliest_upstream(blockers, own_rank)
    if upstream is not None:
        out.append(
            _upstream_or_ask(
                run,
                task.id,
                upstream,
                blockers,
                past,
                limits,
                tasks or [],
                lineage=lineage,
                gate=None,
            )
        )
        return out

    tried_stronger = task.model == limits.stronger_model or any(
        d.kind is DecisionKind.STRONGER_MODEL and d.task_id in lineage for d in past
    )
    if tried_stronger:
        out.append(
            _ask(
                run,
                task.id,
                f"Task {task.id} ({task.agent.value}) still fails after "
                f"its attempts and the stronger model: {_summary(blockers)}. "
                "How should I proceed?",
                "policy.ask_after_stronger_model",
                fids,
            )
        )
    else:
        out.append(
            _decision(
                run,
                DecisionKind.STRONGER_MODEL,
                task_id=task.id,
                choice=f"one last attempt on {limits.stronger_model}",
                reason="the fault is in the task's own output: " + _summary(blockers),
                rule="policy.stronger_model",
                model=limits.stronger_model,
                finding_ids=fids,
            )
        )
    return out


def _ask(run: Run, task_id: str | None, question: str, rule: str, fids: list[str]) -> Decision:
    inputs: dict[str, Any] = {"question": question, "finding_ids": fids}
    if task_id is not None:
        inputs["task_id"] = task_id
    return Decision(
        run_id=run.id,
        task_id=task_id,
        kind=DecisionKind.ASK_USER,
        inputs=inputs,
        choice="ask the user",
        reason=question,
        made_by="code",
        rule=rule,
    )


def _upstream_or_ask(
    run: Run,
    task_id: str | None,
    upstream: ArtifactRef,
    blockers: list[Finding],
    past: list[Decision],
    limits: PolicyLimits,
    tasks: list[Task],
    *,
    lineage: set[str],
    gate: Phase | None,
) -> Decision:
    fids = _ids(blockers)
    already = [
        d
        for d in past
        if d.kind is DecisionKind.UPSTREAM_RERUN
        and (
            (d.task_id is not None and d.task_id in lineage)
            or (gate and d.inputs.get("gate") == gate.value)
        )
    ]
    where = f"task {task_id}" if task_id else f"the {gate.value if gate else ''} gate"
    if already:
        return _ask(
            run,
            task_id,
            f"{where} still fails after re-running {already[-1].inputs.get('artifact')}: "
            f"{_summary(blockers)}. How should I proceed?",
            "policy.ask_after_upstream_rerun",
            fids,
        )
    if run.upstream_reruns >= limits.max_upstream_reruns:
        return _ask(
            run,
            task_id,
            f"{where} needs {upstream} reopened, but the run already used "
            f"its {limits.max_upstream_reruns} upstream re-runs: {_summary(blockers)}. "
            "How should I proceed?",
            "policy.upstream_cap",
            fids,
        )
    producer = _producer(upstream, tasks)
    return _decision(
        run,
        DecisionKind.UPSTREAM_RERUN,
        task_id=task_id,
        choice=f"reopen {upstream} and update everything built on it",
        reason=f"{where} fails because of {upstream}: {_summary(blockers)}",
        rule="policy.upstream_rerun",
        artifact=str(upstream),
        phase=KIND_PHASE.get(upstream.kind, Phase.PLANNING).value,
        gate=gate.value if gate else None,
        producer_task_id=producer.id if producer else None,
        finding_ids=fids,
    )


# --- gate verdicts -------------------------------------------------------------------


@dataclass
class _Fix:
    task: Task
    findings: list[Finding] = field(default_factory=list)


def decide_gate(
    run: Run,
    phase: Phase,
    verdict: Verdict,
    previous: Verdict | None,
    tasks: list[Task],
    outputs: Mapping[str, Artifact],
    past: list[Decision],
    limits: PolicyLimits,
) -> list[Decision]:
    """Decisions for a phase gate verdict. ``tasks`` are the run's tasks, ``outputs``
    the loaded output artifact of each task by task id, ``previous`` the gate's
    last verdict for this phase."""
    if verdict.passed and not verdict.blockers:
        return []
    blockers = verdict.blockers
    fids = _ids(blockers)
    live = [t for t in tasks if t.status is not TaskStatus.SUPERSEDED]
    phase_tasks = sorted(
        (
            t
            for t in live
            if t.phase is phase and t.agent not in (AgentName.USER, AgentName.AUDITOR)
        ),
        key=lambda t: t.created_at,
    )

    if any(f.rule_id == AUDIT_RULE for f in blockers):
        return [
            _ask(
                run,
                None,
                f"The {phase.value} gate audit failed: {_summary(blockers)}. "
                "Re-run the audit, or how should I proceed?",
                "policy.fail_closed",
                fids,
            )
        ]

    if (
        previous is not None
        and not previous.passed
        and _blocker_keys(previous) == _blocker_keys(verdict)
    ):
        return [
            _decision(
                run,
                DecisionKind.NO_PROGRESS,
                task_id=None,
                choice="stop fixing",
                reason=f"the {phase.value} gate failed the same rules on the same items "
                "after its fixes: " + _summary(blockers),
                rule="policy.gate_no_progress",
                finding_ids=fids,
            ),
            _ask(
                run,
                None,
                f"The {phase.value} gate keeps failing: {_summary(blockers)}. "
                "How should I proceed?",
                "policy.gate_no_progress",
                fids,
            ),
        ]

    gate_rank = min((KIND_ORDER[k] for k, p in KIND_PHASE.items() if p is phase), default=99)
    upstream = _earliest_upstream(blockers, gate_rank)
    if upstream is not None:
        return [
            _upstream_or_ask(
                run, None, upstream, blockers, past, limits, live, lineage=set(), gate=phase
            )
        ]

    fixes: dict[str, _Fix] = {}
    unrouted: list[Finding] = []
    for finding in blockers:
        target = route(finding, phase_tasks, outputs)
        if target is None:
            unrouted.append(finding)
        else:
            fixes.setdefault(target.id, _Fix(target)).findings.append(finding)

    out: list[Decision] = []
    stuck: list[Finding] = list(unrouted)
    for fix in fixes.values():
        if fix.task.attempt >= fix.task.max_attempts:
            stuck += fix.findings
            continue
        after = downstream(fix.task, phase_tasks, set(fixes))
        out.append(
            _decision(
                run,
                DecisionKind.GATE_FIX,
                task_id=fix.task.id,
                choice=f"revise {fix.task.agent.value} task {fix.task.id}"
                + (f", then re-run {len(after)} later tasks" if after else ""),
                reason=f"the {phase.value} gate failed: {_summary(fix.findings)}",
                rule="policy.gate_fix",
                finding_ids=_ids(fix.findings),
                rerun_after=[t.id for t in after],
            )
        )
    if stuck:
        why = (
            "no task produced the item at fault" if unrouted else "its tasks have no attempts left"
        )
        out.append(
            _ask(
                run,
                None,
                f"The {phase.value} gate failed and {why}: {_summary(stuck)}. "
                "How should I proceed?",
                "policy.gate_cap",
                _ids(stuck),
            )
        )
    return out


def route(
    finding: Finding, phase_tasks: list[Task], outputs: Mapping[str, Artifact]
) -> Task | None:
    """The task in the phase that produced the item at fault."""
    engineers = [t for t in phase_tasks if t.agent is AgentName.ENGINEER]

    def wrote(path: str) -> Task | None:
        """The latest Engineer task whose code change wrote ``path``."""
        hits = [
            t
            for t in engineers
            if isinstance(outputs.get(t.id), CodeChange) and path in outputs[t.id].files_written  # type: ignore[union-attr]
        ]
        return hits[-1] if hits else None

    ref = finding.artifact
    item = (finding.item or "").strip()
    # QA's blocking issues and failed criteria become Engineer fixes.
    if ref is not None and ref.kind == "qa_report" and finding.rule_id == "R7":
        return (wrote(item) if item else None) or (engineers[-1] if engineers else None)
    if ref is not None:
        producer = _producer(ref, phase_tasks)
        if producer is not None:
            return producer
    if item:
        writer = wrote(item.split(":")[0])
        if writer is not None:
            return writer
    if engineers and any(p is Phase.IMPLEMENTATION for p in {t.phase for t in phase_tasks}):
        return engineers[-1]
    producers = [t for t in phase_tasks if t.output is not None]
    return producers[-1] if producers else (phase_tasks[-1] if phase_tasks else None)


def downstream(task: Task, phase_tasks: list[Task], exclude: set[str]) -> list[Task]:
    """Tasks in the phase that build on ``task``, directly or not, plus a fresh QA
    review after any Engineer fix."""
    found: dict[str, Task] = {}
    frontier = {task.id}
    while frontier:
        nxt = {
            t.id
            for t in phase_tasks
            if t.id not in found and t.id != task.id and frontier & set(t.depends_on)
        }
        for t in phase_tasks:
            if t.id in nxt:
                found[t.id] = t
        frontier = nxt
    if task.agent is AgentName.ENGINEER:
        for t in phase_tasks:
            if t.agent is AgentName.QA:
                found.setdefault(t.id, t)
    return [t for t in phase_tasks if t.id in found and t.id not in exclude]


# --- the VerdictPolicy implementation ------------------------------------------------


class CodeVerdictPolicy:
    """Implements ``VerdictPolicy`` over the stores (read only)."""

    def __init__(
        self,
        *,
        tasks: TaskStore,
        artifacts: ArtifactStore,
        decisions: DecisionLog,
        limits: PolicyLimits | None = None,
    ) -> None:
        self.tasks = tasks
        self.artifacts = artifacts
        self.decisions = decisions
        self.limits = limits or PolicyLimits()
        self._gates: dict[tuple[str, Phase], list[Verdict]] = {}

    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        """Decisions for a task verdict; see ``decide_task``."""
        past = await self.decisions.list(run.id)
        tasks = await self.tasks.list_tasks(run.id)
        return decide_task(run, task, verdict, history, past, self.limits, tasks)

    async def on_gate_verdict(self, run: Run, phase: Phase, verdict: Verdict) -> list[Decision]:
        """Decisions for a gate verdict; see ``decide_gate``. Remembers each gate's last
        verdict in memory, for the no-progress check."""
        tasks = await self.tasks.list_tasks(run.id)
        outputs: dict[str, Artifact] = {}
        for t in tasks:
            if t.phase is phase and t.output is not None and t.status is not TaskStatus.SUPERSEDED:
                outputs[t.id] = await self.artifacts.load(run.id, t.output)
        past = await self.decisions.list(run.id)
        seen = self._gates.setdefault((run.id, phase), [])
        previous = seen[-1] if seen else None
        seen.append(verdict)
        return decide_gate(run, phase, verdict, previous, tasks, outputs, past, self.limits)
