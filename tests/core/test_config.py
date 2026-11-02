from pathlib import Path

import pytest

from agent_team.core.config import DEFAULT_CONFIG, deep_merge, load_config
from agent_team.core.control import KillSwitch
from agent_team.core.errors import RunStopped
from agent_team.core.models import AgentName, ReviewMode


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.route_for(AgentName.ENGINEER).model == "claude-haiku-5-5"
    assert cfg.stronger_model is not None
    assert cfg.stronger_model.model == "claude-sonnet-5-5"
    assert cfg.route_for(AgentName.PRODUCT).model == "claude-haiku-5-5"
    assert cfg.context_budget(AgentName.RESEARCHER) == 20_000
    assert cfg.context_budget(AgentName.ENGINEER) == 90_000
    assert cfg.review is ReviewMode.AUTO_ASSUMPTIONS
    assert cfg.retry.delays_s == [1, 2, 4]
    assert cfg.prices["claude-haiku-5-5"].input_per_mtok == 0.10


def test_default_config_turns_thinking_on_with_effort_per_agent() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    claude = cfg.providers.anthropic
    assert claude.thinking == {"type": "adaptive"}
    assert claude.on_history_edit == "drop_block"
    assert cfg.defaults.effort == "low"
    efforts = {agent: cfg.route_for(agent).effort for agent in AgentName}
    assert efforts[AgentName.ENGINEER] == efforts[AgentName.ARCHITECT] == "medium"
    assert efforts[AgentName.PRODUCT] == efforts[AgentName.QA] == "low"
    assert efforts[AgentName.RESEARCHER] == "medium"
    assert cfg.providers.ollama.reasoning_models == ["gpt-oss"]


def test_route_without_effort_takes_the_default(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "defaults: {provider: fake, model: m, effort: high}\n"
        "agents: {qa: {provider: fake, model: n}}\n"
    )
    assert load_config(path).route_for(AgentName.QA).effort == "high"


def test_thinking_in_the_old_per_model_format_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "defaults: {provider: fake, model: m}\n"
        "providers: {anthropic: {thinking: {claude-haiku-5-5: {type: disabled}}}}\n"
    )
    with pytest.raises(ValueError, match="one setting"):
        load_config(path)


def test_bad_effort_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("defaults: {provider: fake, model: m, effort: extreme}\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_ollama_route_must_leave_room_for_the_reply(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "defaults: {provider: fake, model: m}\n"
        "agents: {researcher: {provider: ollama, model: o, context_budget: 32000}}\n"
        "providers: {ollama: {num_ctx: 32768}}\n"
    )
    with pytest.raises(ValueError, match=r"researcher: context_budget 32000"):
        load_config(path)
    # Compaction and Ollama share one estimate, so 24000 leaves room for the reply...
    path.write_text(path.read_text().replace("32000", "24000"))
    assert load_config(path).context_budget(AgentName.RESEARCHER) == 24_000
    # ...unless Ollama is set to count more cautiously than compaction does.
    path.write_text(
        path.read_text().replace("{num_ctx: 32768}", "{num_ctx: 32768, chars_per_token: 2}")
    )
    with pytest.raises(ValueError, match="about 36000 tokens"):
        load_config(path)


def test_ollama_stronger_model_is_checked_against_every_agents_budget(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "defaults: {provider: fake, model: m, context_budget: 90000}\n"
        "stronger_model: {provider: ollama, model: o, context_budget: 20000}\n"
    )
    with pytest.raises(ValueError, match="context_budget 90000"):
        load_config(path)


def test_env_var_and_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("defaults: {provider: fake, model: m}\nbudget: {max_usd_per_run: 2}\n")
    monkeypatch.setenv("AGENT_TEAM_CONFIG", str(path))
    cfg = load_config(overrides={"budget": {"max_minutes_per_run": 5}})
    assert cfg.defaults.provider == "fake"
    assert cfg.budget.max_usd_per_run == 2 and cfg.budget.max_minutes_per_run == 5


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("defaults: {provider: fake, model: m}\nbudgte: {}\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}, "d": 4}) == {
        "a": {"b": 1, "c": 3},
        "d": 4,
    }


def test_kill_switch(tmp_path: Path) -> None:
    ks = KillSwitch(tmp_path)
    ks.check("r1")
    ks.stop("r1", "demo")
    assert ks.is_stopped("r1")
    with pytest.raises(RunStopped, match="demo"):
        ks.check("r1")
    ks.clear("r1")
    ks.check("r1")
