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
