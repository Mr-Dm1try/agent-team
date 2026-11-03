"""The auditor package: rule catalog and store, evidence helpers, the auditor, the tool pre-check
and the verdict policy."""

from agent_team.audit.auditor import LLMAuditor
from agent_team.audit.catalog import CATALOG_PATH, load_catalog
from agent_team.audit.policy import CodeVerdictPolicy, PolicyLimits, decide_task
from agent_team.audit.precheck import ToolPrecheck, make_precheck_hook
from agent_team.audit.rule_store import SqliteRuleStore
from agent_team.audit.toollog import (
    EventFileToolCallLog,
    InMemoryToolCallLog,
    ToolCall,
    ToolCallLog,
)
from agent_team.core.models import AUDIT_RULE

__all__ = [
    "AUDIT_RULE",
    "CATALOG_PATH",
    "CodeVerdictPolicy",
    "EventFileToolCallLog",
    "InMemoryToolCallLog",
    "LLMAuditor",
    "PolicyLimits",
    "SqliteRuleStore",
    "ToolCall",
    "ToolCallLog",
    "ToolPrecheck",
    "decide_task",
    "load_catalog",
    "make_precheck_hook",
]
