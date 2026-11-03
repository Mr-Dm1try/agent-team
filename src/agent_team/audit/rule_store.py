"""SQLite rule store: the catalog seeded into a ``rules`` table, filtered per sweep."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from agent_team.core.models import AgentName, Phase, Rule, RuleScope

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    data TEXT NOT NULL
)
"""


class SqliteRuleStore:
    """Implements ``RuleStore``. Use ``":memory:"`` in tests.

    Re-seeding updates each rule's definition from the catalog but keeps the
    ``enabled`` flag already stored, so a rule switched off stays off.
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        """Open the database, turn on WAL for files and create the table."""
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self._db.commit()
        self._lock = asyncio.Lock()

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    async def seed(self, rules: list[Rule]) -> None:
        """Make the table match the catalog: add and update its rules, keep each stored
        ``enabled`` flag, and delete rules the catalog no longer has."""
        async with self._lock:
            ids = [r.id for r in rules]
            marks = ",".join("?" * len(ids))
            self._db.execute(f"DELETE FROM rules WHERE id NOT IN ({marks})", ids)
            for rule in rules:
                row = self._db.execute("SELECT enabled FROM rules WHERE id = ?", (rule.id,))
                existing = row.fetchone()
                enabled = bool(existing[0]) if existing else rule.enabled
                stored = rule.model_copy(update={"enabled": enabled})
                self._db.execute(
                    "INSERT OR REPLACE INTO rules (id, scope, enabled, data) VALUES (?, ?, ?, ?)",
                    (rule.id, rule.scope.value, int(enabled), stored.model_dump_json()),
                )
            self._db.commit()

    async def set_enabled(self, rule_id: str, enabled: bool) -> None:
        """Switch one rule on or off. Raises ``KeyError`` for an unknown id."""
        async with self._lock:
            row = self._db.execute("SELECT data FROM rules WHERE id = ?", (rule_id,)).fetchone()
            if row is None:
                raise KeyError(rule_id)
            rule = Rule.model_validate_json(row[0]).model_copy(update={"enabled": enabled})
            self._db.execute(
                "UPDATE rules SET enabled = ?, data = ? WHERE id = ?",
                (int(enabled), rule.model_dump_json(), rule_id),
            )
            self._db.commit()

    async def all(self) -> list[Rule]:
        """Every stored rule, enabled or not, by id."""
        async with self._lock:
            rows = self._db.execute("SELECT data FROM rules ORDER BY id").fetchall()
        return [Rule.model_validate_json(r[0]) for r in rows]

    async def for_scope(
        self, scope: str, *, phase: Phase | None, agent: AgentName | None
    ) -> list[Rule]:
        """Enabled rules for one scope. ``phase=None`` or ``agent=None`` skips that filter."""
        wanted = RuleScope(scope)
        async with self._lock:
            rows = self._db.execute(
                "SELECT data FROM rules WHERE scope = ? AND enabled = 1 ORDER BY id",
                (wanted.value,),
            ).fetchall()
        rules = [Rule.model_validate_json(r[0]) for r in rows]
        return [r for r in rules if matches(r, phase=phase, agent=agent)]


def matches(rule: Rule, *, phase: Phase | None, agent: AgentName | None) -> bool:
    """Whether a rule applies to this phase and agent; None skips that filter."""
    if phase is not None and rule.phases and phase not in rule.phases:
        return False
    return not (agent is not None and rule.agents and agent not in rule.agents)
