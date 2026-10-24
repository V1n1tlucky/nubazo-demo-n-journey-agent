"""Cross-encoder reranking, the signal behind the off-page relevance gate.

Bi-encoder cosine scores sit in a narrow band for any short question, so on-page
and off-page questions overlap (DECISIONS §5). A cross-encoder reads the question
and chunk *together* and separates them with a wide margin, at ~300 ms on CPU for
the handful of chunks retrieval returns.
"""

from __future__ import annotations

import asyncio
from typing import Protocol


class Reranker(Protocol):
    name: str

    async def score(self, query: str, texts: list[str]) -> list[float]: ...


class FastEmbedReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = f"fastembed:{model_name}"
        self._model = TextCrossEncoder(model_name=model_name)

    async def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return await asyncio.to_thread(lambda: [float(s) for s in self._model.rerank(query, texts)])
