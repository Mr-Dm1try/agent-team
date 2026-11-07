"""The verdict policy: turns a verdict into Decisions for the orchestrator to apply.

Code makes every choice here, so each Decision has ``made_by="code"`` and names the
policy branch in ``rule``. The branches follow docs/DESIGN.md, "What a verdict
triggers". Every check runs at the end of a task (there are no phase gates), so the
policy first works out whose output each blocker is about:

1. Pass: no decisions; the orchestrator marks the task done.
2. A blocker on an earlier phase's artifact reopens the earliest one at fault
   (UPSTREAM_RERUN) right away, since retrying this task can't fix it. In
   Implementation, a blocker on the PRD or spec about something missing that the
   current spec already plans (a requirement id it maps, a file or an interface) is the
   code's fault instead (``planned_in``): it goes to the Engineer that builds it.
3. A blocker on another task's output in the same phase revises that task (FIX), and
   the tasks after it re-run too. QA's blocking issues and failed criteria become
   Engineer fixes for the files named, then a fresh QA review.
4. Otherwise the fault is the task's own. With attempts left: RETRY as a revision of
   the last output, findings attached. With none left: ESCALATE, with one
   STRONGER_MODEL attempt before the user is asked.
5. A failure that repeats the same rules on the same items as the verdict before it,
   on this task or the one it was reopened from, is NO_PROGRESS: no more retries or
   fixes; the user is asked.
6. Still failing after an upstream re-run, or past a cap: ASK_USER.

An attempt whose own output failed validation (``OWN_OUTPUT_ERRORS``) climbs the
same ladder once the dispatcher escalates it, for no attempts left or for the same
failure as the attempt before (no progress): one STRONGER_MODEL attempt, then
ASK_USER (``decide_error``).

Two more come from the auditor section: a dangerous agent action (S5, S6, S7) STOPs
the run and asks the user, since it already happened, and this wins over everything
else, including a fail-closed verdict that still carries such a helper hit; and a
fail-closed verdict (an ``AUDIT`` finding) escalates and asks, since the expert isn't
at fault. A helper finding marked ``already_happened`` (such as a secret in a tool
call that already ran) escalates and asks at once too, since no retry can clear it;
the user may keep the work.

Extra ``inputs`` keys beyond the contract's: ``finding_ids`` on most kinds,
``previous_output`` on RETRY, ``phase``/``producer_task_id`` on UPSTREAM_RERUN,
``rerun_after`` (task ids) on FIX and ``attempt`` (the first attempt on the stronger
model) on STRONGER_MODEL.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_team.audit.evidence import KIND_ORDER, KIND_PHASE
from agent_team.core.artifacts import Artifact, ArtifactRef, CodeChange, Spec, plain_kind
from agent_team.core.errors import is_own_output_error, repeats_error
from agent_team.core.interfaces import ArtifactStore, DecisionLog, TaskStore, VerdictStore
from agent_team.core.models import (
    AUDIT_RULE,
    DANGEROUS_ACTION_RULES,
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
from agent_team.orchestrator.lineage import ancestry
from agent_team.orchestrator.planning import package_of


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
    """A code-made Decision with task_id always in its inputs."""
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
    """A short one-line list of findings for reasons and questions."""
    items = [f"{f.rule_id}" + (f" {f.item}" if f.item else "") + f": {f.message}" for f in findings]
    more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return "; ".join(items[:limit]) + more


def _ids(findings: Iterable[Finding]) -> list[str]:
    """The ids of these findings."""
    return [f.id for f in findings]


def _lineage(task: Task) -> set[str]:
    """The task and the task it came from, for history lookups."""
    return {task.id} | ({task.parent_task_id} if task.parent_task_id else set())


def _producer(ref: ArtifactRef, tasks: Iterable[Task]) -> Task | None:
    """The task that produced an artifact version, else the latest of that kind."""
    exact = [t for t in tasks if t.output == ref]
    if exact:
        return exact[-1]
    same_kind = [t for t in tasks if t.output is not None and t.output.kind == ref.kind]
    return max(same_kind, key=lambda t: t.created_at) if same_kind else None


# --- whose fault a blocker is ------------------------------------------------------


def _is_upstream(ref: ArtifactRef, task: Task) -> bool:
    """Whether ``ref`` comes before ``task``'s own output: a kind ranked below it from
    an earlier phase, or the plan, which every other task builds on. Work of the same
    phase is fixed in place instead (see ``route``). A Researcher asked for research
    during a later phase writes a brief, which is never upstream of itself."""
    rank = KIND_ORDER.get(ref.kind, 99)
    if rank >= KIND_ORDER.get(task.output_kind, 99):
        return False
    return ref.kind == "plan" or KIND_PHASE.get(ref.kind) is not task.phase


def _earliest_upstream(findings: Iterable[Finding], task: Task) -> ArtifactRef | None:
    """The most upstream artifact a finding blames, if any is upstream of ``task``."""
    upstream = [f.artifact for f in findings if f.artifact and _is_upstream(f.artifact, task)]
    if not upstream:
        return None
    return min(upstream, key=lambda r: (KIND_ORDER.get(r.kind, 99), r.version))


def route(
    finding: Finding, task: Task, phase_tasks: list[Task], outputs: Mapping[str, Artifact]
) -> Task | None:
    """The other task in ``task``'s phase whose output a finding is about, or None
    when the fault is the task's own (or upstream)."""
    others = [t for t in phase_tasks if t.id != task.id]
    engineers = [t for t in others if t.agent is AgentName.ENGINEER]

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
    by_file = wrote(item.split(":")[0]) if item else None
    if task.agent is AgentName.QA:
        # QA reviews the whole solution: the blocking issues in its report become
        # Engineer fixes, and so does an unattributed finding on a file an Engineer
        # wrote. Anything else unattributed (no report, a bad report) is QA's own.
        if ref is not None and ref.kind == "qa_report" and finding.rule_id == "R7":
            return by_file or (engineers[-1] if engineers else None)
        if ref is None:
            return by_file
    if ref is None or ref == task.output or KIND_PHASE.get(ref.kind) is not task.phase:
        return None
    return _producer(ref, others)


#: Words a finding uses for a file or interface that isn't in the code.
MISSING = re.compile(
    r"\b(missing|not implemented|unimplemented|not registered|not found|absent"
    r"|doesn't exist|does not exist|no such)\b",
    re.I,
)
#: Words that say a requirement isn't built, as opposed to the PRD or spec lacking
#: something about it, such as "REQ-3 is missing an acceptance criterion".
NOT_BUILT = re.compile(
    r"\b(not implemented|unimplemented|isn't implemented|not registered|not built)\b", re.I
)
#: Upstream kinds whose blame is checked against the current spec.
PLANNING_KINDS = frozenset({"prd", "spec"})


def _names(text: str, names: Iterable[str]) -> bool:
    """Whether ``text`` names any of ``names`` as a whole word, id or path, so REQ-1
    doesn't match REQ-10 and recent.py doesn't match recent.pyc."""
    return any(re.search(rf"(?<![\w./-]){re.escape(n)}(?![\w-])", text) for n in names)


def planned_in(spec: Spec, finding: Finding) -> list[str]:
    """The components of ``spec`` that already plan what a finding says the code
    lacks: a file they plan or an interface they expose that is missing, or a
    requirement id they map that isn't built (R5, planned files, counts as either).
    When there are any, the code lacks it, not the spec."""
    text = f"{finding.item or ''} {finding.message} {finding.evidence or ''}"
    files_missing = finding.rule_id == "R5" or MISSING.search(text) is not None
    req_missing = finding.rule_id == "R5" or NOT_BUILT.search(text) is not None
    if not (files_missing or req_missing):
        return []
    # A planned file is also named by its file name, when no other planned file has it.
    basenames = Counter(f.rsplit("/", 1)[-1] for c in spec.components for f in c.files)
    found: list[str] = []
    for c in spec.components:
        things = [i.name for i in spec.interfaces if i.component == c.name]
        for path in c.files:
            base = path.rsplit("/", 1)[-1]
            things += [path, base] if basenames[base] == 1 and base != "__init__.py" else [path]
        if (files_missing and _names(text, things)) or (
            req_missing and _names(text, c.requirements)
        ):
            found.append(c.name)
    return found


def _builder(
    components: list[str], task: Task, spec: Spec, engineers: list[Task], tasks: list[Task]
) -> Task | None:
    """The Engineer task that builds these components: the one for their work package,
    else the last Engineer. An Engineer's own check names only itself, and only when
    the components are in its package: another package's are built by its own task."""
    packages = {p.name for p in spec.work_packages if set(p.components) & set(components)}
    if task.agent is AgentName.ENGINEER:
        mine = package_of(ancestry(task, tasks)[0])
        return task if not spec.work_packages or mine in packages else None
    for t in reversed(engineers):
        if package_of(ancestry(t, tasks)[0]) in packages:
            return t
    return engineers[-1] if engineers else None


def needs_spec(task: Task, blockers: Iterable[Finding]) -> bool:
    """Whether deciding these blockers needs the current spec: an Implementation task's
    check that blames the PRD or spec."""
    return task.phase is Phase.IMPLEMENTATION and any(
        f.artifact is not None and f.artifact.kind in PLANNING_KINDS for f in blockers
    )


def blame_code(
    blockers: list[Finding],
    task: Task,
    spec: Spec | None,
    phase_tasks: list[Task],
    tasks: list[Task],
) -> tuple[list[Finding], list[str]]:
    """``blockers`` with each PRD or spec blame about something the current spec
    already plans moved to the code of the Engineer that builds it, and the ids of the
    findings moved. Only in Implementation, where the code is built from that spec."""
    if spec is None or not needs_spec(task, blockers):
        return blockers, []
    engineers = [t for t in phase_tasks if t.agent is AgentName.ENGINEER and t.output is not None]
    out: list[Finding] = []
    moved: list[str] = []
    for f in blockers:
        components = planned_in(spec, f) if f.artifact and f.artifact.kind in PLANNING_KINDS else []
        builder = _builder(components, task, spec, engineers, tasks) if components else None
        target = builder.output if builder is not None else None
        if target is None:
            out.append(f)
            continue
        note = f"spec@v{spec.version} plans it in {', '.join(components)}, so the code lacks it"
        evidence = f"{f.evidence}\n{note}" if f.evidence else note
        out.append(f.model_copy(update={"artifact": target, "evidence": evidence}))
        moved.append(f.id)
    return out, moved


@dataclass
class _Fix:
    """The findings routed to one other task in the phase."""

    task: Task
    findings: list[Finding] = field(default_factory=list)


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


# --- task verdicts -------------------------------------------------------------------


def decide_task(
    run: Run,
    task: Task,
    verdict: Verdict,
    history: list[Verdict],
    past: list[Decision],
    limits: PolicyLimits,
    tasks: list[Task] | None = None,
    outputs: Mapping[str, Artifact] | None = None,
    spec: Spec | None = None,
) -> list[Decision]:
    """Decisions for one task's end check. ``history`` is the earlier verdicts of the
    task and of the tasks it was reopened from, oldest first; ``past`` is the run's
    decision log so far; ``tasks`` the run's tasks, ``outputs`` the loaded output of
    each live task in the phase, by task id (for routing fixes), and ``spec`` the
    current spec, to tell missing code from a missing plan. The first decision names
    the findings moved from the PRD or spec to the code, in ``blamed_code``."""
    if verdict.passed and not verdict.blockers:
        return []
    live = [t for t in tasks or [] if t.status is not TaskStatus.SUPERSEDED]
    phase_tasks = sorted(
        (
            t
            for t in live
            if t.phase is task.phase and t.agent not in (AgentName.USER, AgentName.AUDITOR)
        ),
        key=lambda t: t.created_at,
    )
    blockers, moved = blame_code(verdict.blockers, task, spec, phase_tasks, tasks or [])
    out = _decide(run, task, verdict, blockers, history, past, limits, live, phase_tasks, outputs)
    if out and moved:
        out[0].inputs["blamed_code"] = moved
    return out


def _decide(
    run: Run,
    task: Task,
    verdict: Verdict,
    blockers: list[Finding],
    history: list[Verdict],
    past: list[Decision],
    limits: PolicyLimits,
    live: list[Task],
    phase_tasks: list[Task],
    outputs: Mapping[str, Artifact] | None,
) -> list[Decision]:
    """The decisions for ``blockers``, the failed verdict's blockers with any blame
    already moved to the code; ``live`` are the run's live tasks and ``phase_tasks``
    the work in the task's phase, oldest first."""
    fids = _ids(blockers)

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
                f"The {_who(task)} did something unsafe: "
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
                f"The check of the {_who(task)}'s work couldn't be completed: "
                f"{_summary(blockers)}. Re-run the check, or how should I proceed?",
                "policy.fail_closed",
                fids,
            ),
        ]

    logged = [f for f in blockers if f.already_happened]
    if logged:
        reason = "a finding is about a tool call that already ran, so a retry can't clear it"
        return [
            _decision(
                run,
                DecisionKind.ESCALATE,
                task_id=task.id,
                choice="escalate",
                reason=f"{reason}: {_summary(logged)}",
                rule="policy.already_happened",
                finding_ids=fids,
            ),
            _ask(
                run,
                task.id,
                f"The {_who(task)}'s check found something in what it already did, which "
                f"a retry can't undo: {_summary(blockers)}. How should I proceed?",
                "policy.already_happened",
                fids,
            ),
        ]

    lineage = _lineage(task)
    earlier = [v for v in history if v.id != verdict.id]
    no_progress = bool(
        earlier and not earlier[-1].passed and _blocker_keys(earlier[-1]) == _blocker_keys(verdict)
    )
    upstream = _earliest_upstream(blockers, task)
    fixes: dict[str, _Fix] = {}
    if upstream is None:
        for finding in blockers:
            target = route(finding, task, phase_tasks, outputs or {})
            if target is not None:
                fixes.setdefault(target.id, _Fix(target)).findings.append(finding)

    out: list[Decision] = []
    if no_progress:
        out.append(
            _decision(
                run,
                DecisionKind.NO_PROGRESS,
                task_id=task.id,
                choice="escalate now",
                reason="it failed the same rules on the same items as before: "
                + _summary(blockers),
                rule="policy.no_progress",
                finding_ids=fids,
            )
        )
        if fixes:  # fixing the same items again won't help
            out.append(
                _ask(
                    run,
                    task.id,
                    f"The {_who(task)}'s check keeps failing after fixes: "
                    f"{_summary(blockers)}. How should I proceed?",
                    "policy.no_progress",
                    fids,
                )
            )
            return out

    if upstream is not None:
        out.append(
            _decision(
                run,
                DecisionKind.ESCALATE,
                task_id=task.id,
                choice="escalate",
                reason=f"the fault is upstream, in {upstream}: " + _summary(blockers),
                rule="policy.escalate",
                finding_ids=fids,
            )
        )
        out.append(
            _upstream_or_ask(run, task.id, upstream, blockers, past, limits, live, lineage=lineage)
        )
        return out

    if fixes:
        routed = {f.id for fix in fixes.values() for f in fix.findings}
        own = [f for f in blockers if f.id not in routed]
        return _fix_decisions(run, task, fixes, phase_tasks, own)

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
    if _tried_stronger(task, past, limits, lineage):
        out.append(
            _ask(
                run,
                task.id,
                f"The {_who(task)} still fails its check after its attempts and the "
                f"stronger model: {_summary(blockers)}. How should I proceed?",
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
                attempt=task.attempt + 1,
                finding_ids=fids,
            )
        )
    return out


def _tried_stronger(
    task: Task, past: list[Decision], limits: PolicyLimits, lineage: set[str]
) -> bool:
    """Whether the task already ran on the stronger model, or was sent to it before."""
    return task.model == limits.stronger_model or any(
        d.kind is DecisionKind.STRONGER_MODEL and d.task_id in lineage for d in past
    )


def decide_error(
    run: Run, task: Task, past: list[Decision], limits: PolicyLimits
) -> list[Decision]:
    """Decisions for a task the dispatcher escalated after an attempt whose own output
    failed validation: no attempts left, or the same failure as the attempt before (no
    progress). One last attempt on the stronger model, then the user is asked. Any
    other error gets none, and the orchestrator asks the user."""
    error = task.last_error
    if task.status is not TaskStatus.ESCALATED or not is_own_output_error(error):
        return []
    stuck = repeats_error(error, task.previous_error)
    out = [
        _decision(
            run,
            DecisionKind.NO_PROGRESS if stuck else DecisionKind.ESCALATE,
            task_id=task.id,
            choice="escalate now" if stuck else "escalate",
            reason=("the same failure as the attempt before" if stuck else "no attempts left")
            + f": {error}",
            rule="policy.error_no_progress" if stuck else "policy.error_escalate",
            error=error,
        )
    ]
    if _tried_stronger(task, past, limits, _lineage(task)):
        out.append(
            _ask(
                run,
                task.id,
                f"The {_who(task)} still can't produce valid output after its attempts and "
                f"the stronger model ({error}). How should I proceed?",
                "policy.ask_after_stronger_model",
                [],
            )
        )
    else:
        out.append(
            _decision(
                run,
                DecisionKind.STRONGER_MODEL,
                task_id=task.id,
                choice=f"one last attempt on {limits.stronger_model}",
                reason=f"the task's own output failed: {error}",
                rule="policy.stronger_model",
                model=limits.stronger_model,
                attempt=task.attempt + 1,
            )
        )
    return out


def _fix_decisions(
    run: Run,
    task: Task,
    fixes: Mapping[str, _Fix],
    phase_tasks: list[Task],
    own: list[Finding],
) -> list[Decision]:
    """A FIX for each other task at fault, re-running the tasks after it and the task
    whose check found it (a fresh check once the fix is in). The checking task's own
    blockers ride on the first FIX, as feedback for its re-run. When any task at fault
    has no attempts left, nothing is fixed and the user is asked, since the check
    can't pass without that work."""
    stuck = [
        f
        for fix in fixes.values()
        if fix.task.attempt >= fix.task.max_attempts
        for f in fix.findings
    ]
    if stuck:
        return [
            _ask(
                run,
                task.id,
                f"The {_who(task)}'s check failed and the work at fault has no attempts "
                f"left: {_summary(stuck)}. How should I proceed?",
                "policy.fix_cap",
                _ids(stuck),
            )
        ]
    out: list[Decision] = []
    for fix in fixes.values():
        after = downstream(fix.task, phase_tasks, set(fixes))
        if all(t.id != task.id for t in after):
            after.append(task)
        out.append(
            _decision(
                run,
                DecisionKind.FIX,
                task_id=fix.task.id,
                choice=f"revise the {fix.task.agent.value} task {fix.task.id}"
                + (f", then re-run {len(after)} later tasks" if after else ""),
                reason=f"the {_who(task)}'s check found: {_summary(fix.findings)}",
                rule="policy.fix",
                finding_ids=_ids(fix.findings),
                rerun_after=[t.id for t in after],
                **(
                    {"checker": task.id, "checker_finding_ids": _ids(own)}
                    if own and not out
                    else {}
                ),
            )
        )
    return out


def _who(task: Task) -> str:
    """The task's agent in plain words, for questions to the user."""
    return {
        AgentName.ORCHESTRATOR: "planner",
        AgentName.PRODUCT: "product manager",
        AgentName.QA: "QA reviewer",
    }.get(task.agent, task.agent.value)


def _ask(run: Run, task_id: str | None, question: str, rule: str, fids: list[str]) -> Decision:
    """An ASK_USER decision with the question in its inputs."""
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
    task_id: str,
    upstream: ArtifactRef,
    blockers: list[Finding],
    past: list[Decision],
    limits: PolicyLimits,
    tasks: list[Task],
    *,
    lineage: set[str],
) -> Decision:
    """Reopen the upstream artifact, or ask the user when a cap or a repeat applies."""
    fids = _ids(blockers)
    already = [
        d
        for d in past
        if d.kind is DecisionKind.UPSTREAM_RERUN and d.task_id is not None and d.task_id in lineage
    ]
    name = plain_kind(upstream.kind)
    if already:
        return _ask(
            run,
            task_id,
            f"The work still fails its check after redoing {name}: "
            f"{_summary(blockers)}. How should I proceed?",
            "policy.ask_after_upstream_rerun",
            fids,
        )
    if run.upstream_reruns >= limits.max_upstream_reruns:
        return _ask(
            run,
            task_id,
            f"The work needs {name} redone, but the run already used its "
            f"{limits.max_upstream_reruns} redos of earlier work: {_summary(blockers)}. "
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
        reason=f"task {task_id} fails because of {upstream}: {_summary(blockers)}",
        rule="policy.upstream_rerun",
        artifact=str(upstream),
        phase=KIND_PHASE.get(upstream.kind, Phase.PLANNING).value,
        producer_task_id=producer.id if producer else None,
        finding_ids=fids,
    )


# --- the VerdictPolicy implementation ------------------------------------------------


class CodeVerdictPolicy:
    """Implements ``VerdictPolicy`` over the stores (read only)."""

    def __init__(
        self,
        *,
        tasks: TaskStore,
        artifacts: ArtifactStore,
        decisions: DecisionLog,
        verdicts: VerdictStore,
        limits: PolicyLimits | None = None,
    ) -> None:
        """Wire the policy to the stores it reads."""
        self.tasks = tasks
        self.artifacts = artifacts
        self.decisions = decisions
        self.verdicts = verdicts
        self.limits = limits or PolicyLimits()

    async def on_task_verdict(
        self, run: Run, task: Task, verdict: Verdict, history: list[Verdict]
    ) -> list[Decision]:
        """Decisions for a task verdict; see ``decide_task``. The outputs of the live
        tasks in the task's phase are loaded so fixes can be routed by file, and the
        current spec when a blocker blames it or the PRD in Implementation."""
        if verdict.passed and not verdict.blockers:
            return []
        past = await self.decisions.list(run.id)
        tasks = await self.tasks.list_tasks(run.id)
        spec = await self._spec(run, tasks) if needs_spec(task, verdict.blockers) else None
        outputs: dict[str, Artifact] = {}
        for t in tasks:
            if (
                t.phase is task.phase
                and t.output is not None
                and t.status is not TaskStatus.SUPERSEDED
            ):
                outputs[t.id] = await self.artifacts.load(run.id, t.output)
        return decide_task(
            run,
            task,
            verdict,
            history,
            past,
            self.limits,
            tasks,
            outputs,
            spec=spec,
        )

    async def _spec(self, run: Run, tasks: list[Task]) -> Spec | None:
        """The current spec: the newest live Architect task's output."""
        done = [
            t
            for t in tasks
            if t.agent is AgentName.ARCHITECT
            and t.output is not None
            and t.status is not TaskStatus.SUPERSEDED
        ]
        if not done:
            return None
        latest = max(done, key=lambda t: t.created_at)
        assert latest.output is not None
        spec = await self.artifacts.load(run.id, latest.output)
        return spec if isinstance(spec, Spec) else None

    async def on_task_error(self, run: Run, task: Task) -> list[Decision]:
        """Decisions for a task escalated after a failed attempt; see ``decide_error``."""
        return decide_error(run, task, await self.decisions.list(run.id), self.limits)
