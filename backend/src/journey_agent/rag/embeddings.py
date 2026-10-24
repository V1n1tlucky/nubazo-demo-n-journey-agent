"""Embedding providers behind one small protocol.

Default is fastembed (ONNX, CPU, ~130 MB model, no PyTorch) so retrieval works
even if the LLM server is down, and it keeps the 16 GB machine's unified memory
for the LLM. An OpenAI-compatible provider (e.g. Ollama `nomic-embed-text`) is
available for teams that want one model server.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    name: str

    async def embed_documents(self, texts: list[str]) -> Vectors: ...

    async def embed_query(self, text: str) -> Vectors: ...


def _normalize(vectors: NDArray[np.floating]) -> Vectors:
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normalized: Vectors = arr / np.clip(norms, 1e-12, None)
    return normalized


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)

    async def embed_documents(self, texts: list[str]) -> Vectors:
        vecs = await asyncio.to_thread(lambda: list(self._model.passage_embed(texts)))
        return _normalize(np.vstack(vecs))

    async def embed_query(self, text: str) -> Vectors:
        vecs = await asyncio.to_thread(lambda: list(self._model.query_embed(text)))
        return _normalize(np.vstack(vecs))


class OpenAICompatibleEmbedder:
    def __init__(self, model_name: str, base_url: str, api_key: str) -> None:
        from openai import AsyncOpenAI

        self.name = f"openai:{model_name}"
        self._model = model_name
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def _embed(self, texts: list[str]) -> Vectors:
        resp = await self._client.embeddings.create(model=self._model, input=texts)
        return _normalize(np.array([d.embedding for d in resp.data]))

    async def embed_documents(self, texts: list[str]) -> Vectors:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> Vectors:
        return await self._embed([text])
