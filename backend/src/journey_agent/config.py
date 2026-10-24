"""Typed, environment-driven configuration.

Every tunable lives here so behaviour is reproducible from `.env` alone.
Variables are prefixed with `JA_` (e.g. `JA_LLM_MODEL`).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JA_", env_file=".env", extra="ignore")

    # --- server -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    # Where frontend/ is served from: allowed for CORS and to frame page snapshots.
    frontend_origins: list[str] = Field(
        default_factory=lambda: ["http://127.0.0.1:5173", "http://localhost:5173"]
    )

    # --- LLM (any OpenAI-compatible server: Ollama, mlx_lm.server, LM Studio) --
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # local servers ignore it, the SDK requires one
    llm_model: str = "qwen2.5:7b-instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    llm_timeout_s: float = 120.0
    llm_concurrency: int = 1  # a 7B model on 16 GB serves one request at a time well

    # --- embeddings ---------------------------------------------------------
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- ingestion ----------------------------------------------------------
    fetch_mode: Literal["auto", "static", "browser"] = "auto"
    fetch_timeout_s: float = 20.0
    max_page_bytes: int = 5_000_000
    max_payload_bytes: int = 1_000_000
    allow_private_urls: bool = False  # SSRF guard; enable only for local testing
    min_extracted_chars: int = 200  # below this we assume a JS shell and ask for upload
    chunk_size: int = 900
    chunk_overlap: int = 150

    # --- retrieval ----------------------------------------------------------
    top_k_content: int = 5
    top_k_payload: int = 3
    # Calibrated for bge-small on nubazo.com (see evals/); re-run evals per site.
    relevance_threshold: float = 0.55  # cosine; below => off-page (used only without a reranker)
    # Cross-encoder gate (DECISIONS §5); "" disables it and falls back to the cosine gate.
    reranker_model: str = "BAAI/bge-reranker-base"
    rerank_threshold: float = -4.65  # logit; at or above => on-page. Calibrated with `make eval`
    # Below this the question is off-page whatever the LLM input judge says; between the two
    # thresholds the judge breaks the tie (see the retrieve node).
    rerank_hard_threshold: float = -8.0
    # FAISS returns this many times top_k candidates; the reranker picks the best of them.
    candidate_pool: int = 3
    # Question languages the reranker can judge against the page; others use the cosine gate.
    # bge-reranker-base scores Hindi questions on an English page as off-page (evals/).
    reranker_languages: list[str] = Field(default_factory=lambda: ["English"])
    max_context_chars: int = 9000  # keeps prompt inside small local context windows
    history_turns: int = 4

    # --- guardrails ---------------------------------------------------------
    max_input_chars: int = 1500
    grounding_threshold: float = 0.58  # embedding grounding when the LLM judges are off
    # With LLM judges on, embedding similarity only blocks clearly off-base answers: genuine
    # paraphrases and invented facts both score ~0.55, so the output judge decides above this.
    grounding_floor: float = 0.45
    allow_prompt_override: bool = True  # demo console only; disable for embeds
    # LLM-as-judge checks on the message (pre) and the answer (post); +2 model calls per turn.
    llm_guardrails: bool = True
    # Languages the agent answers in; the visitor's message picks one, the first is the default.
    reply_languages: list[str] = Field(default_factory=lambda: ["English", "Hindi"], min_length=1)

    @field_validator("reply_languages")
    @classmethod
    def _known_languages(cls, names: list[str]) -> list[str]:
        from journey_agent.guardrails.language import parse_language

        for name in names:
            parse_language(name)  # fail at startup, not on the first chat turn
        return names

    @property
    def db_path(self) -> Path:
        return self.data_dir / "journey_agent.sqlite3"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots"


@lru_cache
def get_settings() -> Settings:
    return Settings()
