"""Evidence helpers: a few lines of Python per rule that gather exact answers.

Each helper reads an ``AuditContext`` and returns a ``HelperResult``: findings for
exact hits (``from_helper=True``) and notes the auditor model reads as evidence. A
helper hit on a blocker rule fails the step even if the model misses it. Helpers never
run code and never write; they read artifacts, the task store, the tool-call log and
files in the run's workspace.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from agent_team.audit import patterns
from agent_team.audit.toollog import ToolCall
from agent_team.core.artifacts import (
    ARTIFACT_TYPES,
    PRD,
    Artifact,
    ArtifactBase,
    ArtifactRef,
    CodeChange,
    Plan,
    QAReport,
    ResearchAnswer,
    ResearchBrief,
    Spec,
)
from agent_team.core.models import (
    AgentName,
    Decision,
    Finding,
    Phase,
    Rule,
    RuleScope,
    Run,
    Severity,
    Task,
    TaskStatus,
    Verdict,
)

#: Upstream order of artifact kinds; used to find the earliest artifact at fault.
KIND_ORDER: dict[str, int] = {
    "plan": 0,
    "research_brief": 1,
    "research_answer": 1,
    "prd": 2,
    "spec": 3,
    "code_change": 4,
    "qa_report": 5,
}
KIND_PHASE: dict[str, Phase] = {
    "plan": Phase.PLANNING,
    "research_brief": Phase.PLANNING,
    "research_answer": Phase.PLANNING,
    "prd": Phase.PRD,
    "spec": Phase.SPEC,
    "code_change": Phase.IMPLEMENTATION,
    "qa_report": Phase.IMPLEMENTATION,
}

MAX_FILE_BYTES = 200_000
READ_TOOLS = frozenset({"read_file", "list_dir", "grep", "web_fetch", "fetch", "open_url"})
OUTGOING_TOOLS = frozenset({"web_fetch", "web_search", "fetch", "http_request", "open_url"})


_KIND_OF_TYPE: dict[type[ArtifactBase], str] = {cls: kind for kind, cls in ARTIFACT_TYPES.items()}


def kind_of(artifact: ArtifactBase) -> str:
    """The artifact's kind, such as ``prd``, from its class."""
    return _KIND_OF_TYPE[type(artifact)]


def ref_of(artifact: ArtifactBase) -> ArtifactRef:
    """The ``kind@vN`` reference of an artifact."""
    return ArtifactRef(kind=kind_of(artifact), version=max(artifact.version, 1))


@dataclass
class AuditContext:
    """What one sweep can see. Built by the auditor from the stores."""

    run: Run
    scope: RuleScope
    phase: Phase | None = None
    task: Task | None = None
    artifact: Artifact | None = None  # task sweep: the task's output
    previous: Artifact | None = None  # task sweep: the version before it, on revisions
    outputs: list[tuple[Task, Artifact]] = field(default_factory=list)  # in scope
    latest: dict[str, Artifact] = field(default_factory=dict)  # latest of every kind
    tasks: list[Task] = field(default_factory=list)  # every task in the run
    tool_calls: list[ToolCall] = field(default_factory=list)  # in scope
    decisions: list[Decision] = field(default_factory=list)  # run sweep
    verdicts: dict[str, list[Verdict]] = field(default_factory=dict)  # run sweep, by task

    # --- lookups ---

    def find(self, kind: str) -> Artifact | None:
        """The newest artifact of ``kind`` in scope, else the latest in the run."""
        mine = [a for _, a in self.outputs if kind_of(a) == kind]
        if mine:
            return max(mine, key=lambda a: a.version)
        return self.latest.get(kind)

    def of_type[T: ArtifactBase](self, cls: type[T]) -> list[tuple[Task, T]]:
        """The in-scope outputs of one artifact class, with the task that made each."""
        return [(t, a) for t, a in self.outputs if isinstance(a, cls)]

    def read_workspace(self, rel: str) -> str | None:
        """A text file in the run's workspace, or None if it is outside, missing or large."""
        if not patterns.inside(rel, self.run.workspace):
            return None
        path = Path(rel) if Path(rel).is_absolute() else Path(self.run.workspace) / rel
        try:
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def code_files(self) -> list[tuple[Task, CodeChange, str, str]]:
        """(task, code_change, path, text) for each file the in-scope code changes wrote."""
        seen: set[str] = set()
        out: list[tuple[Task, CodeChange, str, str]] = []
        for task, change in reversed(self.of_type(CodeChange)):
            for path in change.files_written:
                if path in seen:
                    continue
                seen.add(path)
                text = self.read_workspace(path)
                if text is not None:
                    out.append((task, change, path, text))
        return out

    @property
    def allowed_hosts(self) -> list[str]:
        """Network hosts the latest spec allows the code to reach."""
        spec = self.find("spec")
        return list(spec.allowed_hosts) if isinstance(spec, Spec) else []

    def ran(self) -> list[ToolCall]:
        """Tool calls that actually ran; the pre-check stopped the others."""
        return [c for c in self.tool_calls if not c.blocked]


@dataclass
class HelperResult:
    """What one helper found: exact hits, and notes the model reads as evidence."""

    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


Helper = Callable[[AuditContext, Rule], HelperResult]
HELPERS: dict[str, Helper] = {}


def helper(name: str) -> Callable[[Helper], Helper]:
    """Register an evidence helper under the name rules.yaml uses for it."""

    def register(fn: Helper) -> Helper:
        HELPERS[name] = fn
        return fn

    return register


def hit(
    rule: Rule,
    message: str,
    *,
    item: str | None = None,
    artifact: ArtifactBase | ArtifactRef | None = None,
    evidence: str | None = None,
    fix: str | None = None,
    severity: Severity | None = None,
) -> Finding:
    """A helper finding for ``rule``, marked ``from_helper``; severity defaults to the rule's."""
    ref = ref_of(artifact) if isinstance(artifact, ArtifactBase) else artifact
    return Finding(
        rule_id=rule.id,
        severity=severity or rule.severity,
        artifact=ref,
        item=item,
        message=message,
        evidence=evidence,
        suggested_fix=fix,
        from_helper=True,
    )


def gather(ctx: AuditContext, rules: Iterable[Rule]) -> dict[str, HelperResult]:
    """Run the helper of each rule. A crashing helper becomes a note, never a pass."""
    out: dict[str, HelperResult] = {}
    for rule in rules:
        fn = HELPERS.get(rule.helper or "")
        if fn is None:
            continue
        try:
            out[rule.id] = fn(ctx, rule)
        except Exception as exc:  # a helper bug must not wave a step through
            out[rule.id] = HelperResult(
                findings=[
                    hit(
                        rule,
                        f"evidence helper '{rule.helper}' failed: {type(exc).__name__}: {exc}",
                        severity=Severity.BLOCKER,
                    )
                ]
                if rule.severity is Severity.BLOCKER
                else [],
                notes=[f"helper error: {type(exc).__name__}: {exc}"],
            )
    return out


def _missing(rule: Rule, kind: str) -> HelperResult:
    return HelperResult(findings=[hit(rule, f"no {kind} artifact to check", item=kind)])


# --- planning ----------------------------------------------------------------------

_ASK_ID = re.compile(r"^ASK-[1-9][0-9]*$")


@helper("plan_check")
def plan_check(ctx: AuditContext, rule: Rule) -> HelperResult:
    """Shape checks on the plan. Whether the asks cover the request and invent nothing
    is the model's call, against the request text in the prompt."""
    plan = ctx.find("plan")
    if not isinstance(plan, Plan):
        return _missing(rule, "plan")
    res = HelperResult()
    ids = [a.id for a in plan.asks]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        res.findings.append(hit(rule, f"ask id {dup} is used twice", item=dup, artifact=plan))
    for ask in plan.asks:
        if not _ASK_ID.match(ask.id):
            res.findings.append(
                hit(rule, f"ask id '{ask.id}' is not of the form ASK-n", item=ask.id, artifact=plan)
            )
        if not ask.text.strip():
            res.findings.append(hit(rule, f"{ask.id} has no text", item=ask.id, artifact=plan))
    known = set(ids)
    for i, rq in enumerate(plan.research_questions, start=1):
        if rq.ask not in known:
            res.findings.append(
                hit(rule, f"research question {i} names unknown ask '{rq.ask}'",
                    item=rq.question[:80], artifact=plan)
            )  # fmt: skip
        if not rq.question.strip():
            res.findings.append(
                hit(rule, f"research question {i} is empty", item=f"question {i}", artifact=plan)
            )
    res.notes.append(
        f"request kind {plan.request_kind}; asks {ids}; "
        f"{len(plan.research_questions)} research questions"
    )
    return res


# --- every task ----------------------------------------------------------------------


@helper("schema")
def schema(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R1: the task produced an artifact of the kind it was asked for, and it parses."""
    task, artifact = ctx.task, ctx.artifact
    if task is None:
        return HelperResult()
    if artifact is None:
        return HelperResult(
            findings=[hit(rule, f"no {task.output_kind} artifact was produced", item=task.id)]
        )
    kind = kind_of(artifact)
    if kind != task.output_kind:
        return HelperResult(
            findings=[
                hit(
                    rule,
                    f"expected a {task.output_kind} artifact, got {kind}",
                    item=task.id,
                    artifact=artifact,
                )
            ]
        )
    try:
        ARTIFACT_TYPES[kind].model_validate(artifact.model_dump())
    except ValidationError as exc:
        return HelperResult(
            findings=[hit(rule, "artifact fails its schema", artifact=artifact, evidence=str(exc))]
        )
    return HelperResult(notes=[f"{kind}@v{artifact.version} parses into its schema"])


def _texts(model: BaseModel) -> str:
    return model.model_dump_json(indent=1)


@helper("secrets")
def secrets(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S1: secret-like strings in the outputs, the code files and the tool calls."""
    res = HelperResult()
    sources: list[tuple[str, str, ArtifactBase | None]] = []
    for _, artifact in ctx.outputs:
        sources.append((str(ref_of(artifact)), _texts(artifact), artifact))
    for _, change, path, text in ctx.code_files():
        sources.append((path, text, change))
    for call in ctx.ran():
        sources.append((f"tool call {call.tool}", str(call.args) + "\n" + call.result, None))
    for where, text, source in sources:
        for h in patterns.find_secrets(text):
            res.findings.append(
                hit(
                    rule,
                    f"{h.kind} in {where}",
                    item=f"{where}:{h.line}",
                    artifact=source,
                    evidence=f"{h.redacted} (line {h.line})",
                    fix="read it from an environment variable or a secrets file outside the "
                    "workspace",
                )
            )
    res.notes.append(f"scanned {len(sources)} artifacts, files and tool calls for secrets")
    return res


@helper("dangerous_actions")
def dangerous_actions(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S5: destructive commands, downloads piped to a shell, deletes outside the
    workspace and credential files, in the tool calls that ran."""
    res = HelperResult()
    for call in ctx.ran():
        command = patterns.command_text(call.args)
        problem = patterns.is_destructive(command) if command else None
        if command and not problem and patterns.rm_outside(command, ctx.run.workspace):
            problem = "deleted files outside the workspace"
        if not problem and any(
            patterns.touches_credentials(p) for p in patterns.path_args(call.args)
        ):
            problem = "read a credential file"
        if not problem and command and patterns.touches_credentials(command):
            problem = "command touches a credential file"
        if problem:
            res.findings.append(
                hit(
                    rule,
                    f"{call.agent or 'agent'} {problem}",
                    item=call.tool,
                    evidence=str(call.args),
                )
            )
    res.notes.append(f"checked {len(ctx.tool_calls)} tool calls for dangerous actions")
    return res


@helper("injected_instructions")
def injected_instructions(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S6: text that reads like instructions in what a tool returned, and a hit when
    a later call reuses a URL or command from that text."""
    res = HelperResult()
    calls = ctx.ran()
    for i, call in enumerate(calls):
        if call.tool not in READ_TOOLS:
            continue
        markers = patterns.injection_markers(call.result)
        if not markers:
            continue
        after = calls[i + 1 :]
        res.notes.append(
            f"{call.tool} {call.args} returned text that looks like instructions "
            f"({markers[:3]}); the {len(after)} tool calls after it were: "
            + "; ".join(f"{c.tool} {c.args}" for c in after[:8])
        )
        planted = set(patterns.urls_in(call.result))
        planted |= {
            line.strip().strip("`$ ")
            for line in call.result.splitlines()
            if patterns.is_destructive(line)
        }
        for later in after:
            later_text = str(later.args)
            used = sorted(p for p in planted if p and p in later_text)
            if used:
                res.findings.append(
                    hit(
                        rule,
                        f"after reading injected text from {call.tool}, the agent called "
                        f"{later.tool} with content from it",
                        item=later.tool,
                        evidence=f"injected: {markers[0]!r}; reused: {used[:3]}",
                    )
                )
    return res


@helper("data_leaving")
def data_leaving(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S7: secrets in outgoing web calls, and commands that upload to outside hosts."""
    res = HelperResult()
    allowed = ctx.allowed_hosts
    for call in ctx.ran():
        if call.tool in OUTGOING_TOOLS:
            for h in patterns.find_secrets(str(call.args)):
                res.findings.append(
                    hit(
                        rule,
                        f"{call.tool} sent a {h.kind} out",
                        item=call.tool,
                        evidence=h.redacted,
                    )
                )
        command = patterns.command_text(call.args)
        if command:
            hosts = patterns.uploads_to_external(command, allowed)
            if hosts:
                res.findings.append(
                    hit(
                        rule,
                        f"command sent data to {', '.join(hosts)}",
                        item=call.tool,
                        evidence=command[:300],
                    )
                )
    return res


@helper("minimal_revision")
def minimal_revision(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R12: what a revision changed against the version before it, and code files
    changed outside the spec's ``affected_files``."""
    task, new, old = ctx.task, ctx.artifact, ctx.previous
    if task is None or new is None or old is None:
        return HelperResult()
    res = HelperResult()
    changed = _changed_sections(old, new)
    res.notes.append(
        f"revision of {ref_of(old)} -> {ref_of(new)} ({task.origin.value}: {task.reason or ''}); "
        f"changed: {changed or 'nothing'}"
    )
    spec = ctx.latest.get("spec")
    if isinstance(new, CodeChange) and isinstance(spec, Spec) and spec.affected_files:
        for path in new.files_written:
            if path not in spec.affected_files:
                res.findings.append(
                    hit(
                        rule,
                        f"{path} changed but the spec revision doesn't list it as affected",
                        item=path,
                        artifact=new,
                    )
                )
    return res


def _changed_sections(old: ArtifactBase, new: ArtifactBase) -> list[str]:
    skip = {"run_id", "task_id", "version", "built_from"}
    a, b = old.model_dump(), new.model_dump()
    out: list[str] = []
    for key in sorted(set(a) | set(b)):
        if key in skip or a.get(key) == b.get(key):
            continue
        before, after = a.get(key), b.get(key)
        if isinstance(before, list) and isinstance(after, list):
            out += [f"{key}[{i}]" for i in _changed_items(before, after)]
        else:
            out.append(key)
    return out


def _changed_items(before: list[Any], after: list[Any]) -> list[str]:
    def key(x: Any, i: int) -> str:
        if isinstance(x, dict):
            return str(x.get("id") or x.get("name") or x.get("key") or i)
        return str(i)

    old = {key(x, i): x for i, x in enumerate(before)}
    new = {key(x, i): x for i, x in enumerate(after)}
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


# --- research ------------------------------------------------------------------------


@helper("sources_opened")
def sources_opened(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R2: every cited source was opened, exactly, by a read tool in this task."""
    artifact = ctx.artifact
    cited: list[str] = []
    if isinstance(artifact, ResearchBrief):
        cited = [s.ref for f in artifact.findings for s in f.sources]
        cited += [s.ref for a in artifact.answers for s in a.sources]
        cited += [s.ref for s in artifact.file_excerpts]
    elif isinstance(artifact, ResearchAnswer):
        cited = [s.ref for s in artifact.sources]
    else:
        return HelperResult()
    ws = ctx.run.workspace
    opened: set[str] = set()
    for call in ctx.ran():
        if call.tool not in READ_TOOLS or call.is_error:
            continue
        for key in ("url", "path", "file", "file_path"):
            value = call.args.get(key)
            if isinstance(value, str):
                opened.add(patterns.normalize_source(value, ws))
    res = HelperResult(notes=[f"opened in this task: {sorted(opened)}"])
    for ref in sorted(set(cited)):
        norm = patterns.normalize_source(ref, ws)
        if norm not in opened:  # exact: fetching a site root doesn't open its pages
            res.findings.append(
                hit(
                    rule,
                    f"cites {ref} but never opened it",
                    item=ref,
                    artifact=artifact,
                    fix="open the source with read_file or web_fetch, or drop the claim",
                )
            )
    return res


@helper("brief_complete")
def brief_complete(ctx: AuditContext, rule: Rule) -> HelperResult:
    """P2: every question in the brief has an answer or a reason it couldn't."""
    brief = ctx.artifact
    if not isinstance(brief, ResearchBrief):
        return HelperResult()
    res = HelperResult()
    plan = ctx.latest.get("plan")
    if isinstance(plan, Plan) and plan.research_questions:
        questions = [f"{q.question} ({q.ask})" for q in plan.research_questions]
        res.notes.append(f"research questions in the plan: {questions}")
    if ctx.task is not None:
        res.notes.append(f"task objective: {ctx.task.objective}")
    if not brief.answers:
        res.findings.append(hit(rule, "the brief answers no questions", artifact=brief))
    for i, ans in enumerate(brief.answers, start=1):
        if not (ans.answer or "").strip() and not (ans.gap or "").strip():
            res.findings.append(
                hit(
                    rule,
                    f"question {i} has neither an answer nor a reason it couldn't be answered",
                    item=ans.question[:80],
                    artifact=brief,
                )
            )
    return res


# --- PRD and spec gates --------------------------------------------------------------


@helper("testable_prd")
def testable_prd(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R3: every requirement has at least one acceptance criterion."""
    prd = ctx.find("prd")
    if not isinstance(prd, PRD):
        return _missing(rule, "prd")
    res = HelperResult(notes=[f"{len(prd.requirements)} requirements"])
    if not prd.requirements:
        res.findings.append(hit(rule, "the PRD has no requirements", artifact=prd))
    for req in prd.requirements:
        if not req.acceptance_criteria:
            res.findings.append(
                hit(rule, f"{req.id} has no acceptance criterion", item=req.id, artifact=prd)
            )
    return res


@helper("spec_coverage")
def spec_coverage(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R4: every requirement maps to a component and every criterion to a check;
    work packages, when set, partition the components exactly."""
    spec, prd = ctx.find("spec"), ctx.find("prd")
    if not isinstance(spec, Spec):
        return _missing(rule, "spec")
    if not isinstance(prd, PRD):
        return _missing(rule, "prd")
    res = HelperResult()
    mapped = {r for c in spec.components for r in c.requirements}
    checked = {ac for chk in spec.checks for ac in chk.criteria}
    for req in prd.requirements:
        if req.id not in mapped:
            res.findings.append(
                hit(rule, f"{req.id} maps to no component", item=req.id, artifact=spec)
            )
        for ac in req.acceptance_criteria:
            if ac.id not in checked:
                res.findings.append(
                    hit(rule, f"{ac.id} has no planned check", item=ac.id, artifact=spec)
                )
    if spec.work_packages:
        names = {c.name for c in spec.components}
        owners: dict[str, list[str]] = {}
        packages = [p.name for p in spec.work_packages]
        for dup in sorted({n for n in packages if packages.count(n) > 1}):
            res.findings.append(
                hit(rule, f"work package {dup} is defined twice", item=dup, artifact=spec)
            )
        for pkg in spec.work_packages:
            if not pkg.components:
                res.findings.append(
                    hit(rule, f"work package {pkg.name} has no components", item=pkg.name,
                        artifact=spec)
                )  # fmt: skip
            for comp in pkg.components:
                if comp not in names:
                    res.findings.append(
                        hit(rule, f"work package {pkg.name} names unknown component {comp}",
                            item=pkg.name, artifact=spec)
                    )  # fmt: skip
                else:
                    owners.setdefault(comp, []).append(pkg.name)
        for comp in sorted(names):
            owned = owners.get(comp, [])
            if len(owned) != 1:
                where = f"in {len(owned)} packages ({owned})" if owned else "in no work package"
                res.findings.append(
                    hit(rule, f"component {comp} is {where}; it must be in exactly one",
                        item=comp, artifact=spec)
                )  # fmt: skip
    return res


@helper("coverage_scope")
def coverage_scope(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R9: at the PRD gate every ask has a requirement (blocker) and requirements
    cite only real asks (warning); at the spec gate components serve real requirements."""
    res = HelperResult()
    plan, prd, spec = ctx.find("plan"), ctx.find("prd"), ctx.find("spec")
    if ctx.phase is Phase.PRD:
        if not isinstance(prd, PRD):
            return _missing(rule, "prd")
        if not isinstance(plan, Plan):
            return _missing(rule, "plan")
        asks = {a.id for a in plan.asks}
        cited = {a for r in prd.requirements for a in r.asks}
        for ask in plan.asks:
            if ask.id not in cited:
                res.findings.append(
                    hit(rule, f"{ask.id} maps to no requirement", item=ask.id, artifact=prd)
                )
        for req in prd.requirements:
            unknown = [a for a in req.asks if a not in asks]
            if unknown:
                res.findings.append(
                    hit(
                        rule,
                        f"{req.id} cites asks the request doesn't have: {unknown}",
                        item=req.id,
                        artifact=prd,
                        severity=Severity.WARNING,
                    )
                )
    elif ctx.phase is Phase.SPEC:
        if not isinstance(spec, Spec):
            return _missing(rule, "spec")
        reqs = {r.id for r in prd.requirements} if isinstance(prd, PRD) else set()
        for comp in spec.components:
            extra = [r for r in comp.requirements if reqs and r not in reqs]
            if not comp.requirements or extra:
                res.findings.append(
                    hit(
                        rule,
                        f"component {comp.name} serves no PRD requirement"
                        + (f" (unknown: {extra})" if extra else ""),
                        item=comp.name,
                        artifact=spec,
                        severity=Severity.WARNING,
                    )
                )
    return res


@helper("open_questions")
def open_questions(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R11: questions in the phase's outputs that are neither answered nor assumed."""
    res = HelperResult()
    for _, artifact in ctx.outputs:
        for q in artifact.questions:
            if not (q.answer or "").strip() and not (q.assumption or "").strip():
                res.findings.append(
                    hit(
                        rule,
                        f"question {q.id} is neither answered nor recorded as an assumption",
                        item=q.id,
                        artifact=artifact,
                        evidence=q.text,
                        fix="ask the user, or record an assumption",
                    )
                )
    return res


# --- implementation gate -------------------------------------------------------------


@helper("files_exist")
def files_exist(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R5: planned files exist, and nothing was written outside the workspace."""
    res = HelperResult()
    spec = ctx.find("spec")
    if not isinstance(spec, Spec):
        return _missing(rule, "spec")
    planned = sorted({f for c in spec.components for f in c.files})
    for path in planned:
        full = Path(path) if Path(path).is_absolute() else Path(ctx.run.workspace) / path
        if not patterns.inside(path, ctx.run.workspace) or not full.exists():
            res.findings.append(
                hit(rule, f"planned file {path} doesn't exist in the workspace", item=path)
            )
    for _task, change in ctx.of_type(CodeChange):
        for path in change.files_written:
            if not patterns.inside(path, ctx.run.workspace):
                res.findings.append(
                    hit(rule, f"{path} is outside the workspace", item=path, artifact=change)
                )
    for call in ctx.ran():
        if call.tool in ("write_file", "edit_file", "append_file"):
            for path in patterns.path_args(call.args):
                if not patterns.inside(path, ctx.run.workspace):
                    res.findings.append(
                        hit(rule, f"{call.tool} wrote outside the workspace: {path}", item=path)
                    )
    res.notes.append(f"planned files: {planned}")
    return res


@helper("tests_ran")
def tests_ran(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R6: each code change reports a passing test run that the tool log backs up."""
    res = HelperResult()
    changes = ctx.of_type(CodeChange)
    if not changes:
        return _missing(rule, "code_change")
    for task, change in changes:
        tests = change.tests
        if tests is None:
            res.findings.append(
                hit(rule, "no unit test run is reported", item=task.id, artifact=change)
            )
            continue
        if tests.exit_code != 0 or tests.failed > 0 or tests.passed == 0:
            res.findings.append(
                hit(
                    rule,
                    f"tests didn't pass: {tests.passed} passed, {tests.failed} failed, "
                    f"exit {tests.exit_code}",
                    item=task.id,
                    artifact=change,
                    evidence=tests.output_tail[-500:],
                )
            )
        ran = [
            c
            for c in ctx.ran()
            if c.task_id == task.id
            and tests.command.strip()
            and tests.command.strip() in patterns.command_text(c.args)
        ]
        if not ran:
            res.findings.append(
                hit(
                    rule,
                    f"the tool log has no run of '{tests.command}' for this task",
                    item=task.id,
                    artifact=change,
                    fix="run the tests with run_command and report that result",
                )
            )
    return res


@helper("qa_clean")
def qa_clean(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R7: QA's blocking issues and failed acceptance criteria."""
    reports = ctx.of_type(QAReport)
    report = reports[-1][1] if reports else ctx.find("qa_report")
    if not isinstance(report, QAReport):
        return _missing(rule, "qa_report")
    res = HelperResult(notes=[f"QA ran end to end: {report.ran_end_to_end}"])
    for issue in report.issues:
        if issue.severity != "blocker":
            continue
        paths: list[str | None] = [*issue.files] or [None]
        for path in paths:
            res.findings.append(hit(rule, f"QA blocker: {issue.text}", item=path, artifact=report))
    for result in report.results:
        if result.result == "fail":
            res.findings.append(
                hit(
                    rule,
                    f"{result.criterion} failed in QA",
                    item=result.criterion,
                    artifact=report,
                    evidence=result.evidence,
                )
            )
        elif result.result == "not_checked":
            res.notes.append(f"{result.criterion} was not checked")
    return res


@helper("interface_names")
def interface_names(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R10: spec interfaces (route paths, function names) missing from the code."""
    spec = ctx.find("spec")
    if not isinstance(spec, Spec):
        return HelperResult()
    code = "\n".join(text for *_, text in ctx.code_files())
    res = HelperResult()
    for iface in spec.interfaces:
        needle = _interface_needle(iface.name)
        if not needle:
            continue
        pattern = re.escape(needle) if "/" in needle else rf"\b{re.escape(needle)}\b"
        if not re.search(pattern, code):
            res.findings.append(
                hit(
                    rule,
                    f"interface {iface.name} from the spec isn't in the code",
                    item=iface.name,
                    artifact=spec,
                )
            )
    return res


def _interface_needle(name: str) -> str:
    route = re.search(r"/[^\s{]*", name)
    if route:
        return route.group(0).rstrip("/") or "/"
    words = re.findall(r"[A-Za-z_]\w*", name)
    return words[-1] if words else ""


@helper("dangerous_code")
def dangerous_code(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S2: eval, exec, shell commands, unsafe deserialization, disabled TLS and
    SQL built from strings, line by line in the code files."""
    res = HelperResult()
    files = ctx.code_files()
    for _, change, path, text in files:
        for h in patterns.find_dangerous_code(text):
            res.findings.append(
                hit(
                    rule,
                    f"{h.what} in {path}",
                    item=path,
                    artifact=change,
                    evidence=f"line {h.line}: {h.text}",
                )
            )
    res.notes.append(f"scanned {len(files)} files")
    return res


@helper("unexpected_reach")
def unexpected_reach(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S3: hosts the spec doesn't allow, and absolute paths outside the workspace."""
    res = HelperResult(notes=[f"allowed hosts: {ctx.allowed_hosts}"])
    for _, change, path, text in ctx.code_files():
        for host in patterns.external_hosts(text, ctx.allowed_hosts):
            res.findings.append(
                hit(
                    rule,
                    f"{path} reaches {host}, which the spec doesn't allow",
                    item=path,
                    artifact=change,
                )
            )
        for h in patterns.find_outside_paths(text):
            res.findings.append(
                hit(
                    rule,
                    f"{path} uses a path outside the workspace: {h.text}",
                    item=path,
                    artifact=change,
                    evidence=f"line {h.line}",
                )
            )
    return res


@helper("risky_dependencies")
def risky_dependencies(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S4: unpinned and lookalike dependencies in the spec and requirements.txt."""
    res = HelperResult()
    spec = ctx.find("spec")
    deps: list[tuple[str, ArtifactBase | None]] = []
    if isinstance(spec, Spec):
        deps += [(d, spec) for d in spec.dependencies]
    reqs = ctx.read_workspace("requirements.txt")
    if reqs:
        deps += [
            (line.split("#")[0].strip(), None)
            for line in reqs.splitlines()
            if line.split("#")[0].strip() and not line.strip().startswith("-")
        ]
    for dep, source in deps:
        for issue in patterns.check_requirement(dep):
            res.findings.append(hit(rule, f"{dep}: {issue.problem}", item=dep, artifact=source))
    res.notes.append(f"dependencies: {[d for d, _ in deps]}")
    return res


# --- whole run -----------------------------------------------------------------------


@helper("process_trail")
def process_trail(ctx: AuditContext, rule: Rule) -> HelperResult:
    """R8: every live task finished with times and a verdict within its attempts,
    and every decision has a reason."""
    res = HelperResult()
    for task in ctx.tasks:
        if task.status is TaskStatus.SUPERSEDED or task.agent in (
            AgentName.USER,
            AgentName.AUDITOR,
        ):
            continue
        if task.status is not TaskStatus.DONE:
            res.findings.append(
                hit(rule, f"task ended as {task.status.value}, not done", item=task.id)
            )
            continue
        if task.started_at is None or task.finished_at is None:
            res.findings.append(hit(rule, "task has no start or end time", item=task.id))
        if not ctx.verdicts.get(task.id):
            res.findings.append(hit(rule, "task has no audit verdict", item=task.id))
        # max_attempts plus the one attempt the policy may grant on the stronger model
        if task.attempt > task.max_attempts + 1:
            res.findings.append(
                hit(
                    rule,
                    f"task ran {task.attempt} attempts, over its limit of {task.max_attempts}",
                    item=task.id,
                )
            )
    for decision in ctx.decisions:
        if not decision.reason.strip():
            res.findings.append(
                hit(rule, f"decision {decision.kind.value} has no reason", item=decision.id)
            )
    res.notes.append(f"{len(ctx.tasks)} tasks, {len(ctx.decisions)} decisions")
    return res
