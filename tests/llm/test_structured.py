import pytest
from pydantic import BaseModel

from agent_team.core.errors import OutputValidationError
from agent_team.llm.structured import extract_json, parse_output


class Out(BaseModel):
    title: str
    count: int


def test_extract_from_fence_and_prose() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert extract_json("[1, 2]") == [1, 2]


def test_bad_json_raises_with_errors() -> None:
    with pytest.raises(OutputValidationError) as info:
        extract_json('{"chaos": "cut off')
    assert info.value.errors and "invalid JSON" in info.value.errors[0]


def test_schema_errors_are_readable() -> None:
    assert parse_output('{"title": "x", "count": 2}', Out) == Out(title="x", count=2)
    with pytest.raises(OutputValidationError) as info:
        parse_output('{"title": "x", "count": "many"}', Out)
    assert any(e.startswith("count:") for e in info.value.errors)
