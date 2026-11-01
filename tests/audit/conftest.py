"""Shared fixtures for the audit tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from agent_team.audit.catalog import load_catalog
from agent_team.audit.rule_store import SqliteRuleStore
from agent_team.audit.toollog import InMemoryToolCallLog
from agent_team.core.models import Run

from .fakes import make_run


@pytest.fixture
def run(tmp_path: Path) -> Run:
    return make_run(tmp_path / "ws")


@pytest.fixture
def rules() -> list[Any]:
    return load_catalog()


@pytest.fixture
async def rule_store() -> AsyncIterator[SqliteRuleStore]:
    store = SqliteRuleStore()
    await store.seed(load_catalog())
    yield store
    store.close()


@pytest.fixture
def tool_log() -> InMemoryToolCallLog:
    return InMemoryToolCallLog()
