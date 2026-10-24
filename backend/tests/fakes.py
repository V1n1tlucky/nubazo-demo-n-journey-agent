"""Deterministic test doubles for the embedder and the chat model."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from journey_agent.llm.client import Completion, Message

_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "is",
    "are",
    "for",
    "on",
    "with",
    "what",
    "how",
    "do",
    "does",
    "you",
    "your",
    "me",
    "about",
    "tell",
    "it",
    "this",
}


class HashEmbedder:
    """Bag-of-words hashed into a fixed space: shared words => high cosine."""

    name = "fake:hash"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in _STOP:
                continue
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text).reshape(1, -1)


class FixedReranker:
    """Returns the same cross-encoder score for every chunk."""

    name = "fake:fixed"

    def __init__(self, score: float) -> None:
        self.value = score

    async def score(self, query: str, texts: list[str]) -> list[float]:
        return [self.value] * len(texts)


class ScriptedLLM:
    model = "fake-llm"

    def __init__(self, answer: str = "", *, fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> Completion:
        self.calls.append(messages)
        if self.fail:
            raise ConnectionError("model server down")
        return Completion(
            self.answer, input_tokens=123, output_tokens=45, model=self.model, latency_ms=12.0
        )

    async def health(self) -> bool:
        return not self.fail
