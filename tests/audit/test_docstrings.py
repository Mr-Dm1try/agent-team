"""Every class and function in the audit package, private ones included, says what it
is for in a docstring. Ruff's pydocstyle rules skip private names, so this walks the
AST instead."""

from __future__ import annotations

import ast
from pathlib import Path

import agent_team.audit

PACKAGE = Path(agent_team.audit.__file__).parent


def test_every_def_and_class_has_a_docstring() -> None:
    missing = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_def = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            if is_def and not ast.get_docstring(node):
                missing.append(f"{path.name}:{node.lineno} {node.name}")
    assert missing == []
