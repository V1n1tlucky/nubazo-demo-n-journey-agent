"""Composition root: builds every long-lived dependency exactly once."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from journey_agent.config import Settings
from journey_agent.graph.builder import build_graph
from journey_agent.graph.nodes import AgentDeps
from journey_agent.ingestion.service import IngestionService
from journey_agent.llm.client import ChatModel, OpenAICompatibleChat
from journey_agent.persistence.repository import Repository
from journey_agent.rag.embeddings import Embedder, FastEmbedEmbedder, OpenAICompatibleEmbedder
from journey_agent.rag.rerank import FastEmbedReranker, Reranker
from journey_agent.rag.store import IndexRegistry
from journey_agent.welcome import WelcomeService

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    repo: Repository
    embedder: Embedder
    llm: ChatModel
    reranker: Reranker | None
    registry: IndexRegistry
    ingestion: IngestionService
    welcome: WelcomeService
    graph: CompiledStateGraph  # type: ignore[type-arg]


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAICompatibleEmbedder(
            settings.embedding_model, settings.llm_base_url, settings.llm_api_key
        )
    return FastEmbedEmbedder(settings.embedding_model)


def make_reranker(settings: Settings) -> Reranker | None:
    return FastEmbedReranker(settings.reranker_model) if settings.reranker_model else None


def make_llm(settings: Settings) -> ChatModel:
    return OpenAICompatibleChat(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
        concurrency=settings.llm_concurrency,
    )


async def build_container(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: ChatModel | None = None,
    reranker: Reranker | None = None,
) -> Container:
    repo = Repository(settings.db_path)
    await repo.init()
    embedder = embedder or make_embedder(settings)
    llm = llm or make_llm(settings)
    reranker = reranker or make_reranker(settings)
    registry = IndexRegistry(settings.index_dir, embedder)
    ingestion = IngestionService(settings, registry, repo, llm)
    graph = build_graph(AgentDeps(settings, registry, embedder, llm, repo, reranker))
    welcome = WelcomeService(repo, llm, registry, ingestion)
    log.info(
        "event=container_ready llm=%s embedder=%s reranker=%s",
        llm.model,
        embedder.name,
        reranker.name if reranker else None,
    )
    return Container(settings, repo, embedder, llm, reranker, registry, ingestion, welcome, graph)
