"""The memory writer: the one hook that saves memories (docs/DESIGN.md, "Memory").

No agent writes to memory. Agents propose decisions and research facts in their output,
and this hook saves them only when that output's check passes. It also builds the
memories nobody proposes in an output:

- a lesson when a task that failed its check now passes: the rule and finding that
  caught it, and the fix that worked (what changed between the two outputs),
- a preference when the orchestrator records a ``remember`` decision for a user's
  message,
- a run summary when the run finishes, with one model call.

Every memory keeps the run and task it came from. Text that reads like instructions is
never saved, so a fetched page can't plant orders for a later run. A memory close to
one already saved is merged into it. When a task that was given memories passes on its
first try, those memories gain usefulness.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Awaitable, Callable, Sequence

from agent_team.agents.prompts import RUN_SUMMARY
from agent_team.audit import patterns
from agent_team.core.artifacts import Artifact, ArtifactRef, ResearchAnswer, ResearchBrief, Spec
from agent_team.core.base import Contract
from agent_team.core.config import MemoryConfig
from agent_team.core.interfaces import (
    AfterHook,
    ArtifactStore,
    EventBus,
    MemoryIndex,
    TaskStore,
    VerdictStore,
)
from agent_team.core.models import (
    AUDIT_RULE,
    AgentName,
    DecisionKind,
    Event,
    EventName,
    Finding,
    Memory,
    MemoryKind,
    MemorySource,
    RuleScope,
    Task,
    TaskStatus,
    Verdict,
)
from agent_team.core.tracing import Tracer
from agent_team.llm.structured import parse_output
from agent_team.llm.types import LLMClient, LLMRequest, Message
from agent_team.memory.embed import Embeddings
from agent_team.orchestrator.lineage import ancestry

log = logging.getLogger(__name__)

#: Longest memory text kept; longer text is cut.
MAX_TEXT = 400
#: Lessons built from one passing verdict.
MAX_LESSONS = 3
#: Artifact fields that say nothing about the fix.
_NOT_THE_FIX = {
    "run_id",
    "task_id",
    "version",
    "built_from",
    "summary",
    "questions",
    "research_requests",
    "memory_proposals",
}


class _Summary(Contract):
    """The run-summary model's reply."""

    summary: str


def failed_closed(verdict: Verdict) -> bool:
    """A check that failed closed (the auditor gave no verdict): it judged nothing."""
    blockers = verdict.blockers
    return bool(blockers) and all(f.rule_id == AUDIT_RULE for f in blockers)


def clip(text: str, limit: int = MAX_TEXT) -> str:
    """The text on one line, cut to ``limit`` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def cited_sources(artifact: Artifact, workspace: str) -> set[str]:
    """Every source the artifact cites, normalized as rule R2 compares them."""
    if isinstance(artifact, ResearchBrief):
        refs = [s.ref for f in artifact.findings for s in f.sources]
        refs += [s.ref for a in artifact.answers for s in a.sources]
        refs += [s.ref for s in artifact.file_excerpts]
    elif isinstance(artifact, ResearchAnswer):
        refs = [s.ref for s in artifact.sources]
    else:
        return set()
    return {patterns.normalize_source(r, workspace) for r in refs}


def changed_fields(old: Artifact, new: Artifact) -> list[str]:
    """The fields of ``new`` that differ from ``old``, leaving out the bookkeeping."""
    a = old.model_dump(mode="json", exclude=_NOT_THE_FIX)
    b = new.model_dump(mode="json", exclude=_NOT_THE_FIX)
    return sorted(k for k in b if a.get(k) != b[k])


def lesson_text(task: Task, finding: Finding, old: Artifact | None, new: Artifact) -> str:
    """A lesson: who hit what rule, the finding, and the fix that worked: the passing
    output's summary and the fields it changed, or the auditor's suggested fix when
    the failed output can't be compared."""
    if old is not None and old.kind == new.kind:
        fields = changed_fields(old, new)
        fix = new.summary.strip().rstrip(".")
        fix += f" (changed {', '.join(fields)})" if fields else ""
    else:
        fix = (finding.suggested_fix or new.summary).strip().rstrip(".")
    who = f"{task.agent.value.capitalize()}, {task.phase.value}"
    return clip(f"{who}: rule {finding.rule_id} caught {finding.message!r}. What fixed it: {fix}.")


class MemoryWriter:
    """Saves memories from passing checks, recorded preferences and finished runs."""

    def __init__(
        self,
        *,
        index: MemoryIndex,
        embeddings: Embeddings,
        tasks: TaskStore,
        artifacts: ArtifactStore,
        verdicts: VerdictStore,
        llm: LLMClient,
        bus: EventBus,
        config: MemoryConfig,
    ) -> None:
        """Keep the stores it reads and the index it writes."""
        self.index = index
        self.embeddings = embeddings
        self.tasks = tasks
        self.artifacts = artifacts
        self.verdicts = verdicts
        self.llm = llm
        self.bus = bus
        self.config = config

    def attach(self, bus: EventBus) -> None:
        """Run on every verdict, decision and finished run, each guarded so a memory
        failure is an error event and never reaches the run."""
        bus.on_after(EventName.AUDIT_VERDICT, self.guarded(self.on_verdict))
        bus.on_after(EventName.DECISION, self.guarded(self.on_decision))
        bus.on_after(EventName.RUN_FINISHED, self.guarded(self.on_run_finished))

    def guarded(self, hook: Callable[[Event], Awaitable[None]]) -> AfterHook:
        """``hook``, with any exception logged as an error event instead of raised."""

        async def run(event: Event) -> None:
            """Call the hook; report what it raised."""
            try:
                await hook(event)
            except Exception as exc:
                log.exception("memory writer failed on %s", event.name)
                await self.bus.emit(
                    Event(
                        name=EventName.ERROR,
                        run_id=event.run_id,
                        task_id=event.task_id,
                        level="error",
                        data={
                            "error_type": type(exc).__name__,
                            "error": str(exc) or type(exc).__name__,
                            "traceback": traceback.format_exc(),
                            "hook": f"memory writer on {event.name.value}",
                        },
                    )
                )

        return run

    # --- hooks -------------------------------------------------------------------------

    async def on_verdict(self, event: Event) -> None:
        """A task's check passed: save its proposals and spec decisions, a lesson when
        it fixes a failed check, and credit the memories it was given on a first-try
        pass. A failed check, a run check or a task reopened meanwhile saves nothing.

        The pass fixes a failure when the check before it, on this task or the tasks it
        was reopened from, failed; a check that failed closed judged nothing and is
        skipped. So a revision of work that already passed is no fix."""
        data = event.data
        if not data.get("passed") or data.get("scope") != RuleScope.TASK.value:
            return
        if event.task_id is None:
            return
        task = await self.tasks.get_task(event.task_id)
        if task.status is TaskStatus.SUPERSEDED or task.output is None:
            return
        artifact = await self.artifacts.load(task.run_id, task.output)
        history = [
            v
            for t in ancestry(task, await self.tasks.list_tasks(task.run_id))
            for v in await self.verdicts.for_task(t.id)
            if v.id != data.get("verdict_id")
        ]
        judged = [v for v in history if not failed_closed(v)]
        fixed = judged[-1] if judged and not judged[-1].passed else None
        workspace = (await self.tasks.get_run(task.run_id)).workspace
        candidates = self._proposed(task, artifact, workspace)
        if fixed is not None:
            candidates += await self._lessons(task, fixed, artifact)
        await self._save_all(task.run_id, candidates)
        own_failures = [v for v in judged if v.task_id == task.id and not v.passed]
        if task.attempt == 1 and fixed is None and not own_failures:
            await self.index.credit(task.id)

    async def on_decision(self, event: Event) -> None:
        """A ``remember`` decision: save the user's preference."""
        if event.data.get("kind") != DecisionKind.REMEMBER.value:
            return
        text = event.data.get("inputs", {}).get("preference")
        if isinstance(text, str) and text.strip():
            source = MemorySource(run_id=event.run_id, task_id=event.task_id)
            await self._save_all(event.run_id, [(MemoryKind.PREFERENCE, text, source)])

    async def on_run_finished(self, event: Event) -> None:
        """Save one summary per run, written by the model, or built from the run's
        record when the model can't. A summary merged into another run's still counts."""
        summaries = await self.index.memories([MemoryKind.RUN_SUMMARY])
        if any(src.run_id == event.run_id for s in summaries for src in s.sources):
            return
        text = await self._summary(event.run_id)
        source = MemorySource(run_id=event.run_id)
        await self._save_all(event.run_id, [(MemoryKind.RUN_SUMMARY, text, source)])

    # --- candidates --------------------------------------------------------------------

    def _proposed(
        self, task: Task, artifact: Artifact, workspace: str
    ) -> list[tuple[MemoryKind, str, MemorySource]]:
        """The output's decisions and the research facts whose source the output
        cites (so R2 checked it was opened)."""
        out: list[tuple[MemoryKind, str, MemorySource]] = []
        here = MemorySource(run_id=task.run_id, task_id=task.id)
        if isinstance(artifact, Spec):
            out += [(MemoryKind.DECISION, d, here) for d in artifact.decisions]
        cited = cited_sources(artifact, workspace)
        for proposal in artifact.memory_proposals:
            if proposal.kind == "decision":
                out.append((MemoryKind.DECISION, proposal.text, here))
            elif proposal.kind != "research_fact":
                log.info("%s proposal left out: checks and the user make those", proposal.kind)
            elif proposal.source and patterns.normalize_source(proposal.source, workspace) in cited:
                ref = here.model_copy(update={"ref": proposal.source})
                out.append((MemoryKind.RESEARCH_FACT, proposal.text, ref))
            else:
                log.info("research fact without a cited source not saved: %s", proposal.text)
        return out

    async def _lessons(
        self, task: Task, failed: Verdict, artifact: Artifact
    ) -> list[tuple[MemoryKind, str, MemorySource]]:
        """Lessons from the failed check's blockers on this task's own output, each with
        the fix that worked. A finding that blamed an earlier artifact was fixed there,
        not here."""
        findings = [
            f
            for f in failed.blockers
            if f.rule_id != AUDIT_RULE and (f.artifact is None or f.artifact.kind == artifact.kind)
        ][:MAX_LESSONS]
        here = MemorySource(run_id=task.run_id, task_id=task.id)
        out: list[tuple[MemoryKind, str, MemorySource]] = []
        for finding in findings:
            old = await self._failed_output(task, failed, finding, artifact)
            out.append((MemoryKind.LESSON, lesson_text(task, finding, old, artifact), here))
        return out

    async def _failed_output(
        self, task: Task, failed: Verdict, finding: Finding, artifact: Artifact
    ) -> Artifact | None:
        """The output the failed check judged: the finding's artifact when it is the
        same kind, else the version before the passing one."""
        ref = finding.artifact
        if ref is None or ref.kind != artifact.kind or ref.version >= artifact.version:
            if artifact.version <= 1:
                return None
            ref = ArtifactRef(kind=artifact.kind, version=artifact.version - 1)
        try:
            return await self.artifacts.load(task.run_id, ref)
        except KeyError:
            return None

    async def _summary(self, run_id: str) -> str:
        """One model call over the run's request, tasks and artifacts."""
        run = await self.tasks.get_run(run_id)
        tasks = [t for t in await self.tasks.list_tasks(run_id) if t.agent is not AgentName.USER]
        lines = [
            f"- {t.agent.value} {t.output_kind}: {t.status.value}, {t.attempt} attempt(s)"
            for t in tasks
            if t.agent is not AgentName.AUDITOR
        ]
        summaries: dict[str, str] = {}
        for t in tasks:
            if t.output is not None and t.status is TaskStatus.DONE:
                art = await self.artifacts.load(run_id, t.output)
                summaries[str(t.output)] = clip(art.summary, 300)
        prompt = (
            f"Request:\n{run.request}\n\nTasks:\n"
            + "\n".join(lines)
            + "\n\nArtifacts:\n"
            + "\n".join(f"- {ref}: {text}" for ref, text in summaries.items())
        )
        try:
            # Under the run's root span, which is closed by now, so the trace stays one tree.
            async with Tracer(self.bus).in_run(
                run, "run summary", "other", agent=AgentName.ORCHESTRATOR
            ):
                reply = await self.llm.complete(
                    LLMRequest(
                        agent=AgentName.ORCHESTRATOR,
                        system=RUN_SUMMARY,
                        messages=[Message.user(prompt)],
                        max_tokens=2000,
                        run_id=run_id,
                    )
                )
            parsed, _ = parse_output(reply.text, _Summary)
            if parsed is not None and parsed.summary.strip():
                return parsed.summary
        except Exception as exc:
            log.warning("run summary model call failed: %s", exc)
        retried = sum(1 for t in tasks if t.attempt > 1)
        return f"{run.title}: {len(summaries)} artifacts, {retried} task(s) retried."

    # --- saving ------------------------------------------------------------------------

    async def _save_all(
        self, run_id: str, candidates: Sequence[tuple[MemoryKind, str, MemorySource]]
    ) -> None:
        """Embed and save each candidate (merging near-duplicates) and log it; text that
        reads like instructions is logged and dropped."""
        kept: list[tuple[MemoryKind, str, MemorySource]] = []
        for kind, raw, source in candidates:
            text = clip(raw)
            markers = patterns.injection_markers(text)
            if not text or markers:
                await self._emit(run_id, source, {"kind": kind.value, "skipped": markers})
                continue
            kept.append((kind, text, source))
        if not kept:
            return
        vectors, error = await self.embeddings.vectors([t for _, t, _ in kept], query=False)
        for i, (kind, text, source) in enumerate(kept):
            vector = vectors[i] if vectors else None
            memory, merged = await self.index.save(
                Memory(kind=kind, text=text, sources=[source]),
                vector,
                merge_at=self.config.merge_similarity,
            )
            await self._emit(
                run_id,
                source,
                {
                    "memory_id": memory.id,
                    "kind": kind.value,
                    "merged": merged,
                    "embedded": vector is not None,
                    "embedder_error": error,
                    "text": text,
                },
            )

    async def _emit(self, run_id: str, source: MemorySource, data: dict[str, object]) -> None:
        """Log one save (or skip) in the run's trace."""
        await self.bus.emit(
            Event(
                name=EventName.MEMORY_SAVED,
                run_id=run_id,
                task_id=source.task_id,
                agent=None,
                level="warning" if "skipped" in data else "info",
                data=data,
            )
        )
