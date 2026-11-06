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
    ResearchQuestion,
    Spec,
    TestRun,
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
    TaskOrigin,
    TaskStatus,
    Verdict,
    attempt_text,
    stronger_attempts,
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
#: Tools whose results bring outside text in; S6 looks for instructions planted in it.
READ_TOOLS = frozenset(
    {"read_file", "list_dir", "grep", "web_fetch", "fetch", "open_url", "memory_search"}
)
OUTGOING_TOOLS = frozenset({"web_fetch", "web_search", "fetch", "http_request", "open_url"})
#: Tools whose results are text from outside, such as a web page. S1 doesn't scan
#: those results, since a tutorial's sample token isn't ours to leak and S6 already
#: checks them for planted instructions; it still scans their args.
UNTRUSTED_RESULT_TOOLS = OUTGOING_TOOLS


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
    #: Task sweep of a research answer: the briefs it was written from.
    briefs: list[ResearchBrief] = field(default_factory=list)
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
        """Store the helper under its name and return it unchanged."""
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
    already_happened: bool = False,
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
        already_happened=already_happened,
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
    """A hit for a rule whose required artifact doesn't exist."""
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
    """An artifact as JSON text, for pattern scans."""
    return model.model_dump_json(indent=1)


@helper("secrets")
def secrets(ctx: AuditContext, rule: Rule) -> HelperResult:
    """S1: secret-like strings in the outputs, the code files and the tool calls. The
    results of tools that bring outside text in are left out (see
    ``UNTRUSTED_RESULT_TOOLS``). A hit in a tool call is marked ``already_happened``:
    the call is in the log, so a retry can't clear it."""
    res = HelperResult()
    # (where, text, artifact at fault, whether it is a logged tool call)
    sources: list[tuple[str, str, ArtifactBase | None, bool]] = []
    for _, artifact in ctx.outputs:
        sources.append((str(ref_of(artifact)), _texts(artifact), artifact, False))
    for _, change, path, text in ctx.code_files():
        sources.append((path, text, change, False))
    for call in ctx.ran():
        text = str(call.args)
        if call.tool not in UNTRUSTED_RESULT_TOOLS:
            text += "\n" + call.result
        sources.append((f"tool call {call.tool}", text, None, True))
    for where, text, source, logged in sources:
        for h in patterns.find_secrets(text):
            res.findings.append(
                hit(
                    rule,
                    f"{h.kind} in {where}",
                    item=f"{where}:{h.line}",
                    artifact=source,
                    evidence=f"{h.redacted} (line {h.line})",
                    fix="the call already ran, so tell the user where the secret came from"
                    if logged
                    else "read it from an environment variable or a secrets file outside the "
                    "workspace",
                    already_happened=logged,
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
    """Top-level fields, and list items by id, that differ between two versions."""
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
    """Ids of list items that were added, removed or changed."""

    def key(x: Any, i: int) -> str:
        """An item's id, name or key, else its position."""
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
        final = call.data.get("final_url")  # where web_fetch ended up after redirects
        if isinstance(final, str):
            opened.add(patterns.normalize_source(final, ws))
    res = HelperResult(notes=[f"opened in this task: {sorted(opened)}"])
    res.findings += _unsourced_pins(ctx, rule, artifact)
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


#: A version pin in research text, written as a requirement: a package name, ==, ===
#: or ~= with no spaces, and a version with at least one dot (``mcp==2.3.0``). Prose
#: such as ``status_code == 200`` or ``timeout>=30`` isn't one.
_PIN = re.compile(r"(?<![\w.-])([A-Za-z][A-Za-z0-9._-]*)(===|==|~=)(\d+(?:\.\w+)+)(?!\w|\.\w)")
#: Names a pin can't be about: the interpreter, not a package.
_NOT_PACKAGES = frozenset({"python", "py"})
#: Most characters read from one saved tool result when looking for a version.
_MAX_SOURCE_CHARS = 2_000_000


def _package(name: str) -> str:
    """A package name in its normal form, so ``Py_JWT`` and ``py-jwt`` match."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _source_texts(ctx: AuditContext) -> list[str]:
    """The text of every source the task opened: each successful read tool's result,
    in full when it was saved to a file for being long."""
    texts: list[str] = []
    for call in ctx.ran():
        if call.tool not in READ_TOOLS or call.is_error:
            continue
        text = call.result
        if call.stored_at:
            try:
                with open(call.stored_at, encoding="utf-8", errors="replace") as f:
                    text = f.read(_MAX_SOURCE_CHARS)
            except OSError:
                pass
        texts.append(text)
    return texts


def _research_text(artifact: ResearchBrief | ResearchAnswer) -> str:
    """The claims of a brief or answer, where a version pin would be stated."""
    if isinstance(artifact, ResearchAnswer):
        return "\n".join([artifact.summary, artifact.answer])
    parts = [artifact.summary, *artifact.constraints]
    parts += [f.claim for f in artifact.findings]
    parts += [a.answer or "" for a in artifact.answers]
    return "\n".join(parts)


def _unsourced_pins(
    ctx: AuditContext, rule: Rule, artifact: ResearchBrief | ResearchAnswer
) -> list[Finding]:
    """R2 for version claims: every pin the research states (``mcp==2.3.0``) shows up
    in a source the task opened, with the package's name in the same source. A pin
    the user's request already gives needs no source."""
    pins = {
        (_package(m[1]), m[1], m[2], m[3])
        for m in _PIN.finditer(_research_text(artifact))
        if m[1].lower() not in _NOT_PACKAGES
    }
    if not pins:
        return []
    texts = _source_texts(ctx)
    request = ctx.run.request.lower()
    out: list[Finding] = []
    for package, name, op, version in sorted(pins):
        if f"{name}{op}{version}".lower() in request.replace(" ", ""):
            continue
        # A release tag may read v2.3.0, and ==2.0 is the release 2.0.0.
        shown = re.compile(rf"(?<![\w.])v?{re.escape(version)}(?:\.0)*(?![\w]|\.\d)")
        spelled = "[-_.]+".join(re.escape(part) for part in package.split("-"))
        named = re.compile(rf"(?<![\w.-]){spelled}(?![\w-])", re.IGNORECASE)
        if any(shown.search(t) and named.search(t) for t in texts):
            continue
        out.append(
            hit(
                rule,
                f"states {name}{op}{version} but no source opened in this task shows that version",
                item=f"{name}{op}{version}",
                artifact=artifact,
                fix=f"fetch https://pypi.org/pypi/{name}/json, which lists every released "
                "version, and cite it; or drop the pin",
            )
        )
    return out


@helper("brief_complete")
def brief_complete(ctx: AuditContext, rule: Rule) -> HelperResult:
    """P2: every question in a brief has an answer or a reason it couldn't; a final
    answer is written from its briefs and pins no version its briefs contradict."""
    if isinstance(ctx.artifact, ResearchAnswer):
        return _answer_matches_briefs(ctx, rule, ctx.artifact)
    brief = ctx.artifact
    if not isinstance(brief, ResearchBrief):
        return HelperResult()
    res = HelperResult()
    if ctx.task is not None:
        res.notes.append(f"the questions this brief was given: {own_questions(ctx)}")
        res.notes.append(
            "judge the brief only on these: the plan's other research questions are "
            "other tasks' work"
        )
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


def own_questions(ctx: AuditContext) -> list[str]:
    """The questions a research brief's task was given: its objective, with the plan's
    ask it serves for planned research, or why an agent asked for research it requested.
    Never the plan's other research questions, which other tasks answer."""
    task = ctx.task
    if task is None:
        return []
    question = task.objective
    plan = ctx.latest.get("plan")
    request = _request_of(task, ctx.tasks)
    if request is not None:
        question += f" ({request.reason})" if request.reason else ""
    elif isinstance(plan, Plan):
        asks = [q.ask for q in own_plan_questions(plan, task)]
        if asks:
            question += f" (for {', '.join(asks)})"
    return [question]


def _request_of(task: Task, tasks: list[Task]) -> Task | None:
    """The task made from a research request that ``task`` is, or was reopened from
    (a fix or revision of requested research); None for planned research."""
    by_id = {t.id: t for t in tasks}
    current: Task | None = task
    seen: set[str] = set()
    while current is not None and current.id not in seen:
        if current.origin is TaskOrigin.RESEARCH_REQUEST:
            return current
        if current.origin is TaskOrigin.PLAN:
            return None
        seen.add(current.id)
        current = by_id.get(current.parent_task_id) if current.parent_task_id else None
    return None


def own_plan_questions(plan: Plan, task: Task) -> list[ResearchQuestion]:
    """The plan's research questions a research task was made from: the one its
    objective is; none for research an agent asked for."""
    return [q for q in plan.research_questions if q.question == task.objective]


def _pins(text: str) -> dict[str, set[str]]:
    """The releases each package is pinned to in ``text``, by normal package name,
    with trailing zero parts dropped, so ``==2.3`` and ``~=2.3.0`` are one release."""
    found: dict[str, set[str]] = {}
    for m in _PIN.finditer(text):
        if m[1].lower() not in _NOT_PACKAGES:
            found.setdefault(_package(m[1]), set()).add(re.sub(r"(\.0)+$", "", m[3]))
    return found


def _answer_matches_briefs(ctx: AuditContext, rule: Rule, answer: ResearchAnswer) -> HelperResult:
    """P2 for a final answer: it pins no package at a version its briefs don't, when
    the briefs pin that package. The briefs' answers go in as notes, so the model can
    judge the rest of the answer against them."""
    res = HelperResult()
    if not ctx.briefs:
        res.notes.append("the answer was written from no research brief")
        return res
    brief_pins: dict[str, set[str]] = {}
    for brief in ctx.briefs:
        for a in brief.answers:
            said = a.answer if a.answer else f"(no answer: {a.gap or 'no reason given'})"
            res.notes.append(f"{ref_of(brief)} on {a.question[:80]!r}: {said[:300]}")
        for package, versions in _pins(_research_text(brief)).items():
            brief_pins.setdefault(package, set()).update(versions)
    for package, versions in sorted(_pins(_research_text(answer)).items()):
        known = brief_pins.get(package)
        if not known or versions <= known:
            continue
        stated = ", ".join(sorted(versions - known))
        res.findings.append(
            hit(
                rule,
                f"pins {package} at {stated}, but its briefs found {', '.join(sorted(known))}",
                item=package,
                artifact=answer,
                fix="answer from the briefs, or say why the answer departs from them",
            )
        )
    return res


# --- PRD and spec ------------------------------------------------------------------


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
    """R9: at the end of the PRD every ask has a requirement (blocker) and requirements
    cite only real asks (warning); at the end of the spec components serve real
    requirements."""
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
    """R11: questions in the checked outputs that are neither answered nor assumed."""
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


# --- implementation -----------------------------------------------------------------


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
    """R6: each code change reports a passing test run that the tool log backs up.

    The reported command matches a logged run after both are normalised: an
    ``sh -c``/``bash -c`` wrapper unwrapped, leading ``VAR=value`` assignments and
    redirections dropped, whitespace collapsed. The logged exit code is evidence only
    when nothing after the tests can mask it (see ``patterns.exit_code_masked``);
    otherwise the last pytest summary line in the output decides."""
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
        finding = _test_run_evidence(ctx, rule, task, change, tests, res.notes)
        if finding is not None:
            res.findings.append(finding)
    return res


def _commands_of(ctx: AuditContext, task: Task) -> list[tuple[ToolCall, patterns.CommandShape]]:
    """The task's commands that ran, each with its normalised shape. Only a call whose
    args hold a command counts: install_deps also logs an argv (pip's), but its args
    are packages, so ``pytest`` in a package list is never taken for a test run."""
    out: list[tuple[ToolCall, patterns.CommandShape]] = []
    for call in ctx.ran():
        if call.task_id != task.id:
            continue
        argv = call.args.get("argv")
        if isinstance(argv, list) and argv:
            out.append((call, patterns.command_shape([str(a) for a in argv])))
        elif patterns.command_text(call.args):
            out.append((call, patterns.command_shape(patterns.command_text(call.args))))
    return out


def _test_run_evidence(
    ctx: AuditContext,
    rule: Rule,
    task: Task,
    change: CodeChange,
    tests: TestRun,
    notes: list[str],
) -> Finding | None:
    """The R6 finding on whether the log backs up the reported test run, or None."""
    declared = patterns.command_shape(tests.command)
    commands = _commands_of(ctx, task)
    ran = [
        (c, shape) for c, shape in commands if patterns.contains_command(shape.text, declared.text)
    ]
    if not ran:
        seen = list(dict.fromkeys(patterns.command_text(c.args) for c, _ in commands))[-5:]
        listed = "; ".join(f"`{cmd[:160]}`" for cmd in seen) or "none"
        return hit(
            rule,
            f"tests.command '{tests.command[:200]}' matches no command this task ran "
            f"(commands that ran: {listed})",
            item=task.id,
            artifact=change,
            fix="set tests.command to the exact argv of the test run you report, joined by "
            "spaces, such as `python -m pytest -q`, with no notes; run the tests as plain "
            'argv, such as ["python", "-m", "pytest", "-q"]',
        )
    # The run the report describes: the last one of exactly that command, else the last
    # that contains it. A later narrower run (one file, -k) doesn't replace it.
    exact = [(c, shape) for c, shape in ran if shape.text == declared.text]
    last, shape = (exact or ran)[-1]
    notes.append(
        f"runs of '{tests.command}' in the tool log: "
        + "; ".join(f"{patterns.command_text(c.args)} -> {c.data.get('exit_code')}" for c, _ in ran)
    )
    target = declared.steps[-1].text if declared.steps else declared.text
    index = max(
        (i for i, step in enumerate(shape.steps) if patterns.contains_command(step.text, target)),
        default=len(shape.steps) - 1,
    )
    masked = patterns.exit_code_masked(shape, index)
    logged = last.data.get("exit_code")
    if masked is None:
        if isinstance(logged, int) and logged != tests.exit_code:
            return hit(
                rule,
                f"the last run of '{tests.command}' in the tool log exited {logged}, "
                f"but the change reports exit {tests.exit_code}",
                item=task.id,
                artifact=change,
                evidence=last.result[-500:],
                fix="fix the code until the tests pass, run them again and report that run",
            )
        return None
    summary = patterns.pytest_summary(last.result)
    plain = 'run the tests as plain argv, such as ["python", "-m", "pytest", "-q"]'
    if summary is None:
        return hit(
            rule,
            f"the exit code of the last run of '{tests.command}' says nothing: {masked}, "
            "and its output has no pytest summary line",
            item=task.id,
            artifact=change,
            evidence=last.result[-500:],
            fix=f"{plain}, so its exit code is the tests' own",
        )
    failed = summary.get("failed", 0) + summary.get("errors", 0)
    notes.append(f"exit code masked ({masked}); pytest summary: {summary}")
    if failed or not summary.get("passed"):
        return hit(
            rule,
            f"the last run of '{tests.command}' shows {summary.get('passed', 0)} passed, "
            f"{summary.get('failed', 0)} failed and {summary.get('errors', 0)} errors in its "
            f"pytest summary, whatever its exit code ({masked})",
            item=task.id,
            artifact=change,
            evidence=last.result[-500:],
            fix=f"fix the code until the tests pass, then {plain}",
        )
    return None


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
    declared = [
        line
        for _, out in ctx.outputs
        if isinstance(out, CodeChange)
        for line in out.spec_deviations
    ]
    if declared:
        res.notes.append(f"departures from the spec the Engineer declared: {declared}")
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
    """What to search the code for: a route's static path or a function's name."""
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
    stronger = stronger_attempts(ctx.decisions)
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
        first = stronger.get(task.id)
        if first is not None and task.attempt >= first:
            res.notes.append(
                f"{task.id}: attempt {first} ran on the stronger model, which the policy "
                f"grants past the configured limit by raising it; the task ended at "
                f"{attempt_text(task.attempt, task.max_attempts)}"
            )
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
