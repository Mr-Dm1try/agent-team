"""Embeddings for memory: nomic-embed-text through Ollama's ``/api/embed``, a
deterministic fake for tests, and the switch that falls back to BM25 alone while the
model is down (docs/DESIGN.md, "Memory")."""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections.abc import Callable, Sequence

import httpx

from agent_team.core.config import EmbedderConfig
from agent_team.core.errors import EmbedderUnavailable
from agent_team.core.interfaces import Embedder
from agent_team.core.words import STOPWORDS, WORD
from agent_team.llm.ollama_api import api_root, error_text, model_context

#: Characters per token planned for when cutting a text to the embedder's window. Lower
#: than for chat prompts, since memories hold paths, URLs and code that pack tokens.
EMBED_CHARS_PER_TOKEN = 2.2


class OllamaEmbedder:
    """Implements ``Embedder`` with Ollama's ``/api/embed``. No retries: a failure falls
    back to BM25 at once instead of holding up a task. Each text is cut to what fits the
    model's window, so Ollama never truncates it itself."""

    def __init__(
        self,
        config: EmbedderConfig,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Talk to the Ollama server at ``base_url`` with the configured model;
        ``transport`` replaces the network in tests."""
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=api_root(config.base_url or base_url),
            timeout=config.timeout_s,
            transport=transport,
        )
        self._max_chars: int | None = None

    async def max_chars(self) -> int:
        """The longest text, prefix included, sent to be embedded: the window (the
        configured cap, or the model's own limit from ``/api/show`` when smaller) at
        a low chars-per-token ratio, so the cut errs short."""
        if self._max_chars is not None:
            return self._max_chars
        cap = self.config.max_input_tokens
        limit = await model_context(self._client, self.config.model)
        chars = int(min(cap, limit or cap) * EMBED_CHARS_PER_TOKEN)
        if limit is not None:
            self._max_chars = chars  # Ollama answered; until it does, ask again next time
        return chars

    async def embed(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        """One vector per text, with nomic's search prefix for queries or documents."""
        if not texts:
            return []
        prefix = self.config.query_prefix if query else self.config.document_prefix
        room = max(await self.max_chars() - len(prefix), 1)
        body = {
            "model": self.config.model,
            "input": [prefix + t[:room] for t in texts],
            # A backstop for text denser than planned; the cut above is the real limit.
            "truncate": True,
        }
        try:
            reply = await self._client.post("/api/embed", json=body)
        except httpx.HTTPError as exc:
            raise EmbedderUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if reply.status_code >= 400:
            raise EmbedderUnavailable(f"HTTP {reply.status_code}: {error_text(reply)}")
        try:
            vectors = reply.json().get("embeddings") or []
        except (ValueError, AttributeError) as exc:
            raise EmbedderUnavailable(f"unreadable reply: {exc}") from exc
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise EmbedderUnavailable(f"{self.config.model} returned {len(vectors)} vectors")
        return [list(map(float, v)) for v in vectors]


class FakeEmbedder:
    """Implements ``Embedder`` without a model: each word but the most common ones is
    hashed into one of ``dims`` buckets, so texts that share words point the same way.
    Deterministic, and ``fail`` makes every call raise as if Ollama were down."""

    def __init__(self, dims: int = 256, *, fail: bool = False) -> None:
        """Vectors of ``dims`` floats; ``fail`` starts it down."""
        self.dims = dims
        self.fail = fail
        self.calls = 0

    async def embed(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        """One unit vector per text, or EmbedderUnavailable while ``fail`` is set."""
        self.calls += 1
        if self.fail:
            raise EmbedderUnavailable("ConnectError: fake embedder is down")
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        """Word counts by hashed bucket, scaled to length 1."""
        out = [0.0] * self.dims
        for word in WORD.findall(text.lower()):
            if word in STOPWORDS:
                continue
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            out[int.from_bytes(digest, "big") % self.dims] += 1.0
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]


class Embeddings:
    """The embedder behind a switch: a call that fails or takes longer than
    ``timeout_s`` gives no vectors, and calls are then skipped for ``retry_after_s``
    while search runs on keywords alone and memories are saved without vectors. Each
    answer says why vectors are missing, so the trace can say so."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        retry_after_s: float,
        timeout_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wrap ``embedder``; ``clock`` is swapped in tests."""
        self.embedder = embedder
        self.retry_after_s = retry_after_s
        self.timeout_s = timeout_s
        self._clock = clock
        self._down_until = 0.0
        self._last_error = ""

    async def vectors(
        self, texts: Sequence[str], *, query: bool
    ) -> tuple[list[list[float]] | None, str | None]:
        """(vectors, None), or (None, why) when the embedder is down or just failed."""
        if self._clock() < self._down_until:
            return None, f"embedder down: {self._last_error}"
        try:
            call = self.embedder.embed(texts, query=query)
            return await asyncio.wait_for(call, timeout=self.timeout_s), None
        except (EmbedderUnavailable, TimeoutError) as exc:
            self._last_error = str(exc) or f"no answer in {self.timeout_s:g}s"
            self._down_until = self._clock() + self.retry_after_s
            return None, f"embedder down: {self._last_error}"
