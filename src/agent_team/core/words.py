"""Words for keyword matching, shared by memory search and research-request matching."""

from __future__ import annotations

import re

#: Words too common to help a keyword match.
_COMMON = (
    "a an and are as at be by for from has have in into is it its of on or that the "
    "this to was were will with"
)
STOPWORDS = frozenset(_COMMON.split())
#: A word, for keyword matching and the fake embedder.
WORD = re.compile(r"[a-z0-9]+")


def terms(text: str) -> set[str]:
    """The text's distinct words, without the common ones and a plural ``s``."""
    words = (w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1)
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words}
