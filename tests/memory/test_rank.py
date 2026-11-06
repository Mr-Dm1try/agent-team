"""Ranking memories: the order on a fixed fixture is pinned, weights come from the
config, weak and other-run matches are filtered, and the block fits its caps."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent_team.core.artifacts import PRD, Spec
from agent_team.core.config import MemoryConfig, MemoryWeights
from agent_team.core.models import (
    AgentName,
    Memory,
    MemoryKind,
    MemoryMatch,
    MemorySource,
    Phase,
    Task,
)
from agent_team.memory.rank import build_query, fit, lesson_agents, line, rank

NOW = datetime(2026, 11, 4, tzinfo=UTC)
RUN = "run_now"
K = MemoryKind


def match(
    id: str,
    kind: MemoryKind,
    cosine: float | None,
    bm25: float,
    *,
    age_days: float = 0,
    used: int = 0,
    run: str = RUN,
    words: float = 0.0,
    covered: float = 0.0,
) -> MemoryMatch:
    """A search match with fixed scores."""
    memory = Memory(
        id=id,
        kind=kind,
        text=f"text of {id}",
        sources=[MemorySource(run_id=run)],
        created_at=NOW - timedelta(days=age_days),
        usefulness=used,
    )
    return MemoryMatch(memory=memory, cosine=cosine, bm25=bm25, words=words, covered=covered)


#: Scores by hand with the default weights (cosine .5, bm25 .3, recency .1,
#: usefulness .1), BM25 squashed to s / (1 + s):
FIXTURE = [
    match("mem_a", K.LESSON, 0.80, 3.0),  # relevance .781, score .725
    match("mem_b", K.LESSON, 0.70, 1.0, age_days=60, used=4),  # .625, .592
    match("mem_c", K.PREFERENCE, 0.60, 0.0),  # .375, .400
    # Other run: strong on the memory's words the query has too (.6625)
    match("mem_d", K.DECISION, 0.70, 1.0, age_days=30, run="run_old", covered=0.6),  # .625, .550
    match("mem_e", K.DECISION, 0.60, 0.5, run="run_old"),  # .500: other run, not strong
    match("mem_f", K.LESSON, 0.30, 0.0),  # .188: too weak
    match("mem_g", K.RESEARCH_FACT, 0.50, 0.0),  # .3125, .350: this run, kept
    match("mem_h", K.LESSON, None, 4.0),  # keyword only: .800, .740
    match("mem_bb", K.PREFERENCE, 0.60, 0.0),  # ties mem_c; the lower id goes first
]


def test_the_ranking_order_is_pinned() -> None:
    ranked = rank(FIXTURE, MemoryConfig(), now=NOW, run_id=RUN)
    assert [r.memory.id for r in ranked] == [
        "mem_h",
        "mem_a",
        "mem_b",
        "mem_d",
        "mem_bb",
        "mem_c",
        "mem_g",
    ]
    assert round(ranked[1].score, 3) == 0.725 and round(ranked[1].relevance, 3) == 0.781
    # The same input in another order ranks the same.
    assert rank(FIXTURE[::-1], MemoryConfig(), now=NOW, run_id=RUN) == ranked


def test_the_weights_come_from_the_config() -> None:
    heavy_use = MemoryConfig(weights=MemoryWeights(usefulness=0.5))
    ranked = rank(FIXTURE, heavy_use, now=NOW, run_id=RUN)
    assert ranked[0].memory.id == "mem_b"  # the memory that helped four times


def test_other_runs_need_a_strong_match_unless_no_run_asks() -> None:
    ids = [r.memory.id for r in rank(FIXTURE, MemoryConfig(), now=NOW, run_id=RUN)]
    assert "mem_d" in ids and "mem_e" not in ids
    ids = [r.memory.id for r in rank(FIXTURE, MemoryConfig(), now=NOW, run_id=None)]
    assert "mem_e" in ids and "mem_f" not in ids


def test_another_runs_decision_needs_its_words_in_the_query_not_just_bm25() -> None:
    # One rare shared word gives BM25 a strong squashed score on its own.
    one_word = match("mem_x", K.DECISION, None, 1.94, run="run_old", covered=0.33)
    vector = match("mem_y", K.RESEARCH_FACT, 0.55, 1.94, run="run_old", covered=0.17)
    # Most of what it says is about this task, found by meaning or by keywords alone.
    same_topic = match("mem_z", K.DECISION, 0.75, 0.0, run="run_old", covered=0.83)
    keywords = match("mem_w", K.DECISION, None, 2.5, run="run_old", covered=0.83)
    ranked = rank([one_word, vector, same_topic, keywords], MemoryConfig(), now=NOW, run_id=RUN)
    assert sorted(r.memory.id for r in ranked) == ["mem_w", "mem_z"]
    # The same match from this run is kept: only other runs need a strong match.
    here = match("mem_x", K.DECISION, None, 1.94, covered=0.33)
    assert [r.memory.id for r in rank([here], MemoryConfig(), now=NOW, run_id=RUN)] == ["mem_x"]


def test_fit_stops_at_the_item_and_token_caps() -> None:
    ranked = rank(FIXTURE, MemoryConfig(), now=NOW, run_id=RUN)
    assert [r.memory.id for r, _ in fit(ranked, run_id=RUN, max_items=3, max_tokens=500)] == [
        "mem_h",
        "mem_a",
        "mem_b",
    ]
    one = fit(ranked, run_id=RUN, max_items=10, max_tokens=12)
    assert len(one) == 1 and one[0][1] == 'Lesson: "text of mem_h" (this run)'
    assert fit(ranked, run_id=RUN, max_items=0, max_tokens=500) == []


def test_a_line_quotes_the_text_and_names_its_source() -> None:
    fact = Memory(
        kind=K.RESEARCH_FACT,
        text='FastAPI 0.115 needs\n"Python 3.9+"',
        sources=[
            MemorySource(run_id="run_7", task_id="task_1", ref="https://fastapi.tiangolo.com"),
            MemorySource(run_id="run_8", ref="https://fastapi.tiangolo.com"),
        ],
    )
    assert line(fact, "run_9") == (
        'Research fact: "FastAPI 0.115 needs \\"Python 3.9+\\"" '
        "(run run_7, source https://fastapi.tiangolo.com)"
    )
    said = Memory(kind=K.PREFERENCE, text="type hints", sources=[MemorySource(run_id="session")])
    assert line(said, "run_9") == 'Preference: "type hints" (your messages)'


def test_the_query_is_fixed_fields_only() -> None:
    task = Task(
        run_id=RUN,
        phase=Phase.IMPLEMENTATION,
        agent=AgentName.ENGINEER,
        objective="Build the books endpoints",
        output_kind="code_change",
    )
    prd = PRD(summary="Books API. More words.", title="Books API", requirements=[])
    spec = Spec(summary="FastAPI over SQLite. Two components.", components=[], checks=[])
    assert build_query(task, [prd, spec]) == (
        "engineer, implementation, Build the books endpoints, Books API, FastAPI over SQLite"
    )


def test_words_in_common_count_when_bm25_cannot_weigh_terms() -> None:
    # With two memories BM25 gives every shared term no weight; the word share stands in.
    few = MemoryMatch(memory=FIXTURE[0].memory, cosine=None, bm25=0.000004, words=0.5)
    [ranked] = rank([few], MemoryConfig(), now=NOW, run_id=RUN)
    assert ranked.relevance == 0.5


def test_a_line_too_long_is_skipped_for_the_next_one() -> None:
    long = match("mem_long", K.LESSON, 0.99, 9.0)
    long.memory.text = "word " * 400
    ranked = rank([long, FIXTURE[0]], MemoryConfig(), now=NOW, run_id=RUN)
    assert ranked[0].memory.id == "mem_long"
    kept = fit(ranked, run_id=RUN, max_items=10, max_tokens=100)
    assert [r.memory.id for r, _ in kept] == ["mem_a"]


def test_a_line_names_two_sources_at_most() -> None:
    refs = [MemorySource(run_id=f"run_{i}", ref=f"https://s{i}.dev") for i in range(5)]
    fact = Memory(kind=K.RESEARCH_FACT, text="a fact", sources=refs)
    assert line(fact, None).endswith("source https://s0.dev, https://s1.dev and 3 more)")


def test_a_lesson_goes_only_to_the_agent_that_learned_it() -> None:
    def lesson(id: str, text: str, agent: AgentName | None) -> MemoryMatch:
        """A strong lesson match with the given source agent."""
        memory = Memory(
            id=id,
            kind=K.LESSON,
            text=text,
            sources=[MemorySource(run_id=RUN, agent=agent)],
            created_at=NOW,
        )
        return MemoryMatch(memory=memory, cosine=0.9, bm25=3.0)

    matches = [
        lesson("mem_eng", "Engineer, implementation: rule R6 caught ...", AgentName.ENGINEER),
        lesson("mem_res", "Researcher, planning: rule R2 caught ...", AgentName.RESEARCHER),
        # Saved before sources named the agent: the text says who learned it.
        lesson("mem_old", "Engineer, implementation: rule R6 caught ...", None),
        lesson("mem_any", "no agent named here", None),
    ]
    assert lesson_agents(matches[2].memory) == {AgentName.ENGINEER}
    assert lesson_agents(matches[3].memory) == set()

    def ids(agent: AgentName | None) -> list[str]:
        """What ``agent`` is given."""
        return sorted(
            r.memory.id for r in rank(matches, MemoryConfig(), now=NOW, run_id=RUN, agent=agent)
        )

    assert ids(AgentName.RESEARCHER) == ["mem_any", "mem_res"]
    assert ids(AgentName.ENGINEER) == ["mem_any", "mem_eng", "mem_old"]
    assert len(ids(None)) == 4

    # A lesson two agents learned (merged) goes to both.
    shared = lesson("mem_two", "Engineer, implementation: rule R6 caught ...", AgentName.ENGINEER)
    shared.memory.sources.append(MemorySource(run_id="run_2", agent=AgentName.QA))
    assert lesson_agents(shared.memory) == {AgentName.ENGINEER, AgentName.QA}
    for agent in (AgentName.ENGINEER, AgentName.QA):
        assert rank([shared], MemoryConfig(), now=NOW, run_id=RUN, agent=agent)
    assert not rank([shared], MemoryConfig(), now=NOW, run_id=RUN, agent=AgentName.RESEARCHER)
