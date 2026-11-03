"""Every function and class under src/ has a non-empty docstring, private ones too."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def missing_docstrings(root: Path) -> list[str]:
    """``file:line name`` for each def or class under ``root`` without a docstring."""
    gaps: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, DEFS) and not ast.get_docstring(node):
                gaps.append((str(path.relative_to(root.parent)), node.lineno, node.name))
    return [f"{file}:{line} {name}" for file, line, name in sorted(gaps)]


def test_every_function_and_class_has_a_docstring() -> None:
    gaps = missing_docstrings(SRC)
    assert not gaps, "missing docstrings:\n" + "\n".join(gaps)


def test_the_check_catches_nested_and_private_defs(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text(
        'class A:\n    """Doc."""\n\n    def _p(self):\n        def inner():\n'
        '            """Doc."""\n\n        return inner\n\n\nasync def f():\n    pass\n'
        '\n\ndef g():\n    """ """\n'
    )
    assert missing_docstrings(src) == ["src/m.py:4 _p", "src/m.py:11 f", "src/m.py:15 g"]
