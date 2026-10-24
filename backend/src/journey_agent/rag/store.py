"""Per-page FAISS indexes, persisted to disk.

One index per page keeps answers strictly page-scoped: retrieval physically
cannot return another page's content. Pages are small (tens to low thousands
of chunks), so an exact `IndexFlatIP` over normalised vectors (= cosine) is
both the fastest and the most accurate choice — ANN indexes only pay off at
~100k+ vectors.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from journey_agent.ingestion.models import Chunk, SourceType
from journey_agent.rag.embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class PageIndex:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], embedder_name: str) -> None:
        self.index = index
        self.chunks = chunks
        self.embedder_name = embedder_name

    @classmethod
    async def build(cls, chunks: list[Chunk], embedder: Embedder) -> PageIndex:
        texts = [f"{c.heading}\n{c.text}" for c in chunks]
        vectors = await embedder.embed_documents(texts)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, embedder.name)

    def search(
        self, query_vec: np.ndarray, *, k: int, sources: set[SourceType] | None = None
    ) -> list[Hit]:
        if not self.chunks:
            return []
        # Over-fetch then filter by source; exact search makes this cheap.
        n = self.index.ntotal if sources else min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vec.reshape(1, -1), n)
        hits: list[Hit] = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if sources is None or chunk.source in sources:
                hits.append(Hit(chunk, float(score)))
            if len(hits) == k:
                break
        return hits

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta = {"embedder": self.embedder_name, "chunks": [c.model_dump() for c in self.chunks]}
        (directory / "chunks.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: Path) -> PageIndex:
        meta = json.loads((directory / "chunks.json").read_text())
        index = faiss.read_index(str(directory / "index.faiss"))
        return cls(index, [Chunk(**c) for c in meta["chunks"]], meta["embedder"])


class IndexRegistry:
    """Lazy, in-memory cache of page indexes backed by the data directory."""

    def __init__(self, root: Path, embedder: Embedder) -> None:
        self.root = root
        self.embedder = embedder
        self._cache: dict[str, PageIndex] = {}
        self._lock = asyncio.Lock()

    async def put(self, page_id: str, chunks: list[Chunk]) -> PageIndex:
        index = await PageIndex.build(chunks, self.embedder)
        await asyncio.to_thread(index.save, self.root / page_id)
        async with self._lock:
            self._cache[page_id] = index
        return index

    async def get(self, page_id: str) -> PageIndex | None:
        async with self._lock:
            if page_id in self._cache:
                return self._cache[page_id]
            path = self.root / page_id
            if not (path / "index.faiss").exists():
                return None
            index = await asyncio.to_thread(PageIndex.load, path)
            if index.embedder_name != self.embedder.name:
                # Vectors from a different model are meaningless; force re-ingest.
                return None
            self._cache[page_id] = index
            return index
