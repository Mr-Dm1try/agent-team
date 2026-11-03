"""The shared token estimate, its per-model calibration and prompt anchors."""

from __future__ import annotations

from agent_team.agents.compaction import context_tokens, estimate_tokens, sent_json
from agent_team.llm.tokens import (
    MAX_RATIO,
    MIN_RATIO,
    SAFETY,
    Anchor,
    Calibration,
    estimate_json_tokens,
)
from agent_team.llm.types import Message, TextBlock, ThinkingBlock


def test_calibration_starts_at_three_and_keeps_the_lowest_recent_ratio() -> None:
    cal = Calibration()
    assert cal.ratio("m") == 3 and cal.ratio(None) == 3
    cal.observe("m", chars=4000, tokens=1000)
    cal.observe("m", chars=3500, tokens=1000)
    cal.observe("m", chars=4500, tokens=1000)
    assert cal.ratio("m") == 3.5 * SAFETY  # the lowest, with a margin, so estimates stay high
    assert cal.ratio("other") == 3  # per model
    cal.observe("other", chars=2800, tokens=1000)
    assert cal.ratio(None) == 2.8 * SAFETY  # the lowest over every model seen


def test_calibration_ignores_missing_and_partial_counts() -> None:
    cal = Calibration(start=2.5, keep=2)
    assert not cal.observe("m", chars=1000, tokens=0)  # no usage reported
    # A count that left most of the prompt out, such as cached tokens, is no ratio.
    assert not cal.observe("m", chars=int(MAX_RATIO * 1000) + 1, tokens=1000)
    assert cal.ratio("m") == 2.5
    assert cal.observe("m", chars=500, tokens=1000)  # odd, but errs high: clamped
    assert cal.ratio("m") == MIN_RATIO
    cal.observe("m", chars=4000, tokens=1000)
    cal.observe("m", chars=4200, tokens=1000)
    assert cal.ratio("m") == 4.0 * SAFETY  # the odd one aged out


def test_estimates_take_a_learned_fractional_ratio() -> None:
    assert estimate_json_tokens("x" * 35, chars_per_token=3.5) == 10
    assert estimate_json_tokens("x" * 36, chars_per_token=3.5) == 11


def test_context_tokens_anchor_on_the_count_and_estimate_only_whats_new() -> None:
    first = [Message.user("a" * 3000)]
    later = [
        *first,
        Message(role="assistant", content=[TextBlock(text="b" * 60)]),
        Message.user("c" * 60),
    ]
    anchor = Anchor(sent=1, tokens=500)
    appended = estimate_json_tokens(
        [m.model_dump(mode="json") for m in later[1:]], chars_per_token=4
    )
    assert context_tokens("sys", later, anchor=anchor, chars_per_token=4) == 500 + appended
    # Without an anchor, or with one for more messages than there are, all is estimated.
    full = estimate_tokens("sys", later, chars_per_token=4)
    assert context_tokens("sys", later, chars_per_token=4) == full
    stale = Anchor(sent=5, tokens=500)
    assert context_tokens("sys", later, anchor=stale, chars_per_token=4) == full


def test_calibration_measures_the_prompt_without_thinking() -> None:
    thought = ThinkingBlock(provider="anthropic", model="m", raw={"thinking": "t" * 500})
    msgs = [Message(role="assistant", content=[thought, TextBlock(text="hi")])]
    assert "t" * 500 in sent_json("s", msgs)  # the estimate counts it at raw size
    assert "t" * 500 not in sent_json("s", msgs, thinking=False)


def test_thinking_the_next_provider_drops_is_not_counted() -> None:
    claude = ThinkingBlock(provider="anthropic", model="m", raw={"thinking": "t" * 4000})
    msgs = [
        Message.user("go"),
        Message(role="assistant", content=[claude, TextBlock(text="hi")]),
        Message.user("next"),
    ]
    everything = estimate_tokens("s", msgs, chars_per_token=4)
    same = estimate_tokens("s", msgs, chars_per_token=4, provider="anthropic")
    other = estimate_tokens("s", msgs, chars_per_token=4, provider="ollama")
    assert same == everything and other < everything - 900  # ~4000 chars left out
    # Messages appended after an anchor are counted the same way.
    anchor = Anchor(sent=1, tokens=10)
    anchored = context_tokens("s", msgs, anchor=anchor, chars_per_token=4, provider="ollama")
    assert anchored < 10 + 100
