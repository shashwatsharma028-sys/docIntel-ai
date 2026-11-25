"""Deterministic offline fakes. Test-only; never imported by production code.

FakeEmbedder is a hashed bag-of-words embedder. It is NOT a semantic model:
retrieval results obtained with it prove that the pipeline (chunk -> vector ->
FAISS -> id -> chunk/source mapping) is wired correctly, not that Gemini
embeddings retrieve well. Real semantic retrieval quality is out of scope for
the offline suite (see docs/BASELINE.md, "Mocked vs real retrieval").
"""
import hashlib
import re
from types import SimpleNamespace
from unittest import mock

import numpy as np

DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "are", "be",
    "by", "with", "as", "at", "this", "that", "it", "its", "from", "what", "how",
    "which", "do", "does", "any", "all", "may", "will", "must", "not", "no", "per",
}


def embed_text(text: str) -> np.ndarray:
    vec = np.zeros(DIM, dtype="float32")
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeEmbedder:
    """Stands in for client.models.embed_content(model=..., contents=...)."""

    def __init__(self):
        self.calls: list[dict] = []

    def embed_content(self, *, model, contents):
        """One call == one provider request; a list input returns one embedding per item."""
        self.calls.append({"model": model, "contents": contents})
        items = contents if isinstance(contents, list) else [contents]
        self.items = getattr(self, "items", 0) + len(items)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=embed_text(t).tolist()) for t in items])

    @property
    def count(self) -> int:
        return len(self.calls)


class _UnexpectedCall(AssertionError):
    pass


def strict_client(name: str) -> mock.MagicMock:
    """A client mock whose known endpoints fail loudly unless a test configures them."""
    client = mock.MagicMock(name=name)
    err = _UnexpectedCall(f"unexpected external API call on {name}; configure a fake explicitly")
    client.models.embed_content.side_effect = err
    client.models.generate_content.side_effect = err
    client.chat.completions.create.side_effect = err
    return client
