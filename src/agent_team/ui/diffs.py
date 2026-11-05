"""What a change request changed: a section diff for each artifact that got a new
version, and the git diff of the workspace between the code versions' tags.

A snapshot is taken of the latest version of every artifact kind when a run reaches
done; after a change re-runs and the run is done again, ``change_report`` compares
the two, and names the kinds that are new since. Artifacts are compared field by
field, and list items that carry an ``id`` or ``name`` (requirements, components,
checks) are matched by it, so a reordered list doesn't read as a rewrite.
"""

from __future__ import annotations

import difflib
import json
from typing import Any

from agent_team.core.artifacts import Artifact, ArtifactRef
from agent_team.core.errors import SandboxError
from agent_team.core.interfaces import ArtifactStore, TaskStore
from agent_team.core.models import Run
from agent_team.sandbox.git import WorkspaceGit

#: Fields that change with every version and say nothing about its content.
META_FIELDS = frozenset({"run_id", "task_id", "version", "built_from"})
#: Artifact kinds shown as a section diff; code changes show the workspace diff.
SECTION_KINDS = ("plan", "prd", "spec", "qa_report", "research_answer")
MAX_DIFF_LINES = 400

Snapshot = dict[str, int]


async def snapshot(run_id: str, tasks: TaskStore) -> Snapshot:
    """The newest version of each artifact kind the run's tasks have written."""
    versions: Snapshot = {}
    for task in await tasks.list_tasks(run_id):
        if task.output is not None:
            ref = task.output
            versions[ref.kind] = max(versions.get(ref.kind, 0), ref.version)
    return versions


def _key(item: Any, index: int) -> str:
    """How a list item is matched between versions: its id or name, else its place."""
    if isinstance(item, dict):
        for field in ("id", "name", "criterion", "question"):
            if isinstance(item.get(field), str):
                return str(item[field])
    return f"#{index + 1}"


def _matched(old: list[Any], new: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Both lists keyed for matching: by ``_key``, or by place when a key repeats in
    either list, so no item is hidden behind another with the same key."""
    old_keys = [_key(x, i) for i, x in enumerate(old)]
    new_keys = [_key(x, i) for i, x in enumerate(new)]
    if len(set(old_keys)) < len(old_keys) or len(set(new_keys)) < len(new_keys):
        old_keys = [f"#{i + 1}" for i in range(len(old))]
        new_keys = [f"#{i + 1}" for i in range(len(new))]
    return dict(zip(old_keys, old, strict=True)), dict(zip(new_keys, new, strict=True))


def _text(value: Any) -> list[str]:
    """A value as stable, readable lines."""
    if isinstance(value, str):
        return value.splitlines() or [""]
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).splitlines()


def _unified(old: Any, new: Any, label: str) -> list[str]:
    """A unified diff of two values, labelled with the section it belongs to."""
    lines = difflib.unified_diff(
        _text(old), _text(new), f"{label} (before)", f"{label} (after)", lineterm="", n=1
    )
    return list(lines)


def section_diff(old: Artifact, new: Artifact) -> list[str]:
    """The sections that differ between two versions of an artifact, as diff lines."""
    before = old.model_dump(mode="json", exclude=set(META_FIELDS))
    after = new.model_dump(mode="json", exclude=set(META_FIELDS))
    out: list[str] = []
    for field in sorted(set(before) | set(after)):
        a, b = before.get(field), after.get(field)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list):
            old_items, new_items = _matched(a, b)
            for key in sorted(set(old_items) | set(new_items)):
                x, y = old_items.get(key), new_items.get(key)
                if x == y:
                    continue
                label = f"{field}[{key}]"
                if x is None:
                    out.append(f"+ added {label}")
                elif y is None:
                    out.append(f"- removed {label}")
                out += _unified(x if x is not None else "", y if y is not None else "", label)
        else:
            out += _unified(a if a is not None else "", b if b is not None else "", field)
    return out


async def change_report(
    run: Run, before: Snapshot, artifacts: ArtifactStore, tasks: TaskStore
) -> list[str]:
    """Lines that show what changed since ``before``: each artifact with a newer
    version, then the code diff between the code versions' tags."""
    after = await snapshot(run.id, tasks)
    out: list[str] = []
    for kind in SECTION_KINDS:
        old_v, new_v = before.get(kind), after.get(kind)
        if new_v is None or (old_v is not None and new_v <= old_v):
            continue
        if old_v is None:
            out.append(f"{kind} v{new_v}: new")
            continue
        old = await artifacts.load(run.id, ArtifactRef(kind=kind, version=old_v))
        new = await artifacts.load(run.id, ArtifactRef(kind=kind, version=new_v))
        changed = section_diff(old, new)
        out.append(f"{kind} v{old_v} → v{new_v}:")
        out += changed or ["  (no change in content)"]
    old_code, new_code = before.get("code_change"), after.get("code_change")
    if new_code is not None and old_code is None:
        out.append(f"code: code_change v{new_code}: new")
    elif old_code is not None and new_code is not None and new_code > old_code:
        out.append(f"code: code_change v{old_code} → v{new_code}:")
        out += await code_diff(run, old_code, new_code)
    if not out:
        return ["Nothing changed since the last version you saw."]
    if len(out) > MAX_DIFF_LINES:
        cut = len(out) - MAX_DIFF_LINES
        out = [*out[:MAX_DIFF_LINES], f"[... {cut} more lines]"]
    return out


async def code_diff(run: Run, old: int, new: int) -> list[str]:
    """The workspace diff between two code versions' tags, or why there is none."""
    try:
        text = await WorkspaceGit(run.workspace).diff(
            ArtifactRef(kind="code_change", version=old),
            ArtifactRef(kind="code_change", version=new),
        )
    except SandboxError as exc:
        return [f"  (no code diff: {exc})"]
    return text.splitlines() or ["  (no change in the files)"]
