"""Small helpers for Ollama's native HTTP API, shared by the chat provider and the
embedder: where the API lives, a model's context window and the error text a failed
call carries."""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)


def api_root(base_url: str) -> str:
    """The server root for the native ``/api`` routes. A ``base_url`` that still ends
    in the OpenAI-compatible ``/v1`` is taken as its server."""
    root = base_url.rstrip("/")
    return root.removesuffix("/v1")


def tagged(model: str) -> str:
    """The model name with its tag, as Ollama lists it: ``gpt-oss`` is ``gpt-oss:latest``."""
    return model if ":" in model else f"{model}:latest"


def error_text(response: httpx.Response) -> str:
    """The ``error`` Ollama put in a failed response's JSON body, else the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return str(body["error"])
    return response.text.strip()


def trained_context(show: dict[str, Any]) -> int | None:
    """The context length a ``/api/show`` reply gives for the model, if any: the
    ``<arch>.context_length`` entry of ``model_info``, else ``details.context_length``."""
    info = show.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    details = show.get("details") or {}
    value = details.get("context_length")
    return value if isinstance(value, int) and value > 0 else None


async def model_context(client: httpx.AsyncClient, model: str) -> int | None:
    """The model's own context window from ``/api/show``; 0 when Ollama answered but
    gave none, and None when it couldn't answer (it is down, or the model isn't
    pulled), so the caller may ask again. Never raises."""
    try:
        reply = await client.post("/api/show", json={"model": model})
        reply.raise_for_status()
        return trained_context(reply.json()) or 0
    except (httpx.HTTPError, ValueError) as exc:
        log.debug("ollama /api/show %s failed: %s", model, exc)
        return None


async def loaded_context(client: httpx.AsyncClient, model: str) -> int | None:
    """The context the model is loaded with now, from ``/api/ps``, or None when it
    isn't loaded or Ollama doesn't say. Never raises."""
    try:
        reply = await client.get("/api/ps")
        reply.raise_for_status()
        models = reply.json().get("models") or []
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        log.debug("ollama /api/ps failed: %s", exc)
        return None
    wanted = tagged(model)
    for entry in models:
        if not isinstance(entry, dict) or wanted not in (
            tagged(str(entry.get("name", ""))),
            tagged(str(entry.get("model", ""))),
        ):
            continue
        value = entry.get("context_length")
        return value if isinstance(value, int) and value > 0 else None
    return None
