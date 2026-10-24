from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from journey_agent.config import Settings
from journey_agent.container import Container, build_container
from journey_agent.persistence.repository import PageRecord
from tests.fakes import HashEmbedder, ScriptedLLM

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def page_html() -> str:
    return (FIXTURES / "page.html").read_text()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # The hash embedder produces lower cosines than a real model; thresholds match it.
    return Settings(
        data_dir=tmp_path,
        relevance_threshold=0.2,
        grounding_threshold=0.2,
        reranker_model="",  # cosine gate unless a test injects a fake reranker
        _env_file=None,
    )  # type: ignore[call-arg]


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM(
        "Salaried employees aged 21 to 60 years with a minimum monthly income of ₹25,000 "
        "are eligible."
    )


@pytest.fixture
async def container(settings: Settings, llm: ScriptedLLM) -> AsyncIterator[Container]:
    yield await build_container(settings, embedder=HashEmbedder(), llm=llm)


@pytest.fixture
async def page(container: Container, page_html: str) -> PageRecord:
    return await container.ingestion.ingest_upload("page.html", page_html.encode(), None)
