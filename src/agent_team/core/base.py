"""The base class every data contract shares."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """Base for every contract model: unknown fields are rejected, not dropped."""

    model_config = ConfigDict(extra="forbid")
