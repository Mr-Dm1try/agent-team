from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_team.agents.jsonrepair import extract_json, parse_output


class Out(BaseModel):
    name: str
    n: int


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "a", "n": 1}',
        'Here you go:\n```json\n{"name": "a", "n": 1}\n```',
        'Sure! {"name": "a", "n": 1} hope that helps',
        '{"name": "a", "n": 1,}',
        "{“name”: “a”, “n”: 1}",
    ],
)
def test_repairs_common_wrappers(text: str) -> None:
    value, errors = parse_output(text, Out)
    assert errors == [] and value == Out(name="a", n=1)


def test_braces_inside_strings() -> None:
    assert extract_json('x {"name": "a}b", "n": 2} y') == {"name": "a}b", "n": 2}


def test_reports_schema_errors() -> None:
    value, errors = parse_output('{"name": "a", "n": "many"}', Out)
    assert value is None and errors and errors[0].startswith("n:")


def test_reports_missing_json() -> None:
    value, errors = parse_output("no json here", Out)
    assert value is None and "JSON" in errors[0]
    _, errors = parse_output("[1, 2]", Out)
    assert "one JSON object" in errors[0]
