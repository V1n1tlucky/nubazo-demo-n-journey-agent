"""Relevance-gate calibration: `uv run python -m journey_agent.evals evals/nubazo.json`.

Pass `--page-id <id>` to score against an already-ingested page's stored index
instead of re-fetching (live fetches can return a different variant of the page).

The off-page threshold is the single most site-sensitive number in the system.
This script embeds labelled on-page / off-page questions against a page and
reports how well one threshold separates them (cross-encoder score when a
reranker is configured, else cosine), so it is tuned with data
rather than guessed. Run it whenever you onboard a new site or change models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from journey_agent.config import get_settings
from journey_agent.container import make_embedder, make_reranker
from journey_agent.ingestion.chunker import chunk_document
from journey_agent.ingestion.extractor import extract
from journey_agent.ingestion.fetcher import fetch_page
from journey_agent.ingestion.models import Chunk, RawPage, SourceType
from journey_agent.rag.store import PageIndex


async def _chunks(spec: dict[str, Any], html: str | None) -> list[Chunk]:
    settings = get_settings()
    if html is not None:
        raw = RawPage(url=spec["url"], html=html, fetched_with="upload-html")
    else:
        raw = await fetch_page(
            spec["url"],
            mode=settings.fetch_mode,
            timeout_s=settings.fetch_timeout_s,
            max_bytes=settings.max_page_bytes,
            max_payload_bytes=settings.max_payload_bytes,
            allow_private=settings.allow_private_urls,
        )
    return chunk_document(extract(raw), size=settings.chunk_size, overlap=settings.chunk_overlap)


async def main(spec: dict[str, Any], html: str | None, page_id: str | None) -> None:
    settings = get_settings()
    embedder = make_embedder(settings)
    reranker = make_reranker(settings)
    if page_id is not None:
        index = PageIndex.load(settings.index_dir / page_id)
    else:
        index = await PageIndex.build(await _chunks(spec, html), embedder)

    async def top(q: str) -> float:
        # Mirrors the retrieve node's gate: best cosine, or best rerank of the top-k hits.
        qvec = await embedder.embed_query(q)
        hits = index.search(
            qvec, k=settings.top_k_content, sources={SourceType.CONTENT, SourceType.META}
        ) + index.search(qvec, k=settings.top_k_payload, sources={SourceType.PAYLOAD})
        if not hits:
            return 0.0
        if reranker is None:
            return max(h.score for h in hits)
        return max(await reranker.score(q, [f"{h.chunk.heading}\n{h.chunk.text}" for h in hits]))

    on = [(await top(q), q) for q in spec["on_topic"]]
    off = [(await top(q), q) for q in spec["off_topic"]]
    for label, rows in (("ON ", on), ("OFF", off)):
        for score, q in sorted(rows):
            print(f"{label} {score:.3f}  {q}")

    lo_on, hi_off = min(s for s, _ in on), max(s for s, _ in off)
    gate, t = (
        ("rerank", settings.rerank_threshold)
        if reranker
        else ("cosine", settings.relevance_threshold)
    )
    false_rejects = sum(s < t for s, _ in on)
    false_accepts = sum(s >= t for s, _ in off)
    print(f"\nmin on-topic {lo_on:.3f} · max off-topic {hi_off:.3f} · margin {lo_on - hi_off:+.3f}")
    print(
        f"current {gate} threshold {t}: {false_rejects} false rejects, {false_accepts} false accepts"
    )
    if lo_on > hi_off:
        var = "JA_RERANK_THRESHOLD" if reranker else "JA_RELEVANCE_THRESHOLD"
        print(f"suggested {var}={(lo_on + hi_off) / 2:.3f}")
    else:
        print(
            "distributions overlap: rely on LLM grounding rules + post-guardrails, or use a "
            "stronger embedding model / reranker for this site."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--page-id", help="score against this ingested page's stored index")
    args = parser.parse_args()
    spec_path: Path = args.spec
    spec = json.loads(spec_path.read_text())
    html_file = spec.get("html_file")
    html = (spec_path.parent / html_file).read_text() if html_file else None
    asyncio.run(main(spec, html, args.page_id))
