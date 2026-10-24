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
    """Answers the main question with `answer`; side prompts get canned JSON.

    Side prompts (company identification, LLM guardrail judges, greeting) are
    routed by their system prompt and recorded in `side_calls`, so `calls` keeps
    meaning "the answer-generation call" in assertions like `llm.calls == []`.
    """

    model = "fake-llm"

    def __init__(self, answer: str = "", *, fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.company = "Acme Loans"
        self.input_verdict = '{"verdict": "allow", "category": "on_topic", "reason": "about loans"}'
        self.output_verdict = '{"verdict": "pass", "reason": "grounded in the context"}'
        self.greeting: str | None = None  # None: answer the greeting prompt with `answer`
        self.calls: list[list[Message]] = []
        self.side_calls: list[list[Message]] = []

    def _side_reply(self, system: str) -> str | None:
        if "You identify the company" in system:
            return f'{{"company_name": "{self.company}"}}'
        if "You screen customer messages" in system:
            return self.input_verdict
        if "You review the answer" in system:
            return self.output_verdict
        if "first message a customer sees" in system and self.greeting is not None:
            return self.greeting
        return None

    async def complete(self, messages: list[Message]) -> Completion:
        system = messages[0].content if messages and messages[0].role == "system" else ""
        side = self._side_reply(system)
        (self.calls if side is None else self.side_calls).append(messages)
        if self.fail:
            raise ConnectionError("model server down")
        return Completion(
            side if side is not None else self.answer,
            input_tokens=123,
            output_tokens=45,
            model=self.model,
            latency_ms=12.0,
        )

    async def health(self) -> bool:
        return not self.fail
