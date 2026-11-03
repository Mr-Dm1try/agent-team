"""The rule catalog: the versioned defaults in ``config/rules.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agent_team.core.models import Rule

#: ``config/rules.yaml`` at the repository root.
CATALOG_PATH = Path(__file__).resolve().parents[3] / "config" / "rules.yaml"


def load_catalog(path: Path | str | None = None) -> list[Rule]:
    """Parse the catalog file into rules. Raises ``ValueError`` on duplicate ids."""
    raw: Any = yaml.safe_load(Path(path or CATALOG_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("rules"), list):
        raise ValueError("rule catalog must be a mapping with a 'rules' list")
    rules = [Rule.model_validate(item) for item in raw["rules"]]
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise ValueError(f"duplicate rule id in catalog: {rule.id}")
        seen.add(rule.id)
    return rules
