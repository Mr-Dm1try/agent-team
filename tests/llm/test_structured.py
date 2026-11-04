from __future__ import annotations

import pytest
from pydantic import BaseModel

from agent_team.llm.structured import extract_json, parse_output


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
        "{\u201cname\u201d: \u201ca\u201d, \u201cn\u201d: 1}",
    ],
)
def test_repairs_common_wrappers(text: str) -> None:
    value, errors = parse_output(text, Out)
    assert errors == [] and value == Out(name="a", n=1)


def test_extract_finds_fences_prose_and_lists() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert extract_json("[1, 2]") == [1, 2]


def test_braces_inside_strings() -> None:
    assert extract_json('x {"name": "a}b", "n": 2} y') == {"name": "a}b", "n": 2}


def test_backticks_inside_a_fenced_string() -> None:
    text = 'ok:\n```json\n{"name": "```x```", "n": 3}\n```'
    value, errors = parse_output(text, Out)
    assert errors == [] and value == Out(name="```x```", n=3)


def test_a_later_candidate_that_validates_wins() -> None:
    text = 'Example: {"other": 1}\n```json\n{"name": "a", "n": 1}\n```'
    value, _ = parse_output(text, Out)
    assert value == Out(name="a", n=1)


def test_reports_schema_errors() -> None:
    value, errors = parse_output('{"name": "a", "n": "many"}', Out)
    assert value is None and errors and errors[0].startswith("n:")


def test_reports_missing_or_broken_json() -> None:
    value, errors = parse_output("no json here", Out)
    assert value is None and "JSON" in errors[0]
    _, errors = parse_output("[1, 2]", Out)
    assert "one JSON object" in errors[0]
    _, errors = parse_output("", Out)
    assert "empty" in errors[0]
    with pytest.raises(ValueError, match="invalid JSON"):
        extract_json('{"chaos": "cut off')
