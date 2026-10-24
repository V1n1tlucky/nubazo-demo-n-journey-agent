"""Request/response DTOs for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    url: str = Field(min_length=3, max_length=2048)


class ChatRequest(BaseModel):
    page_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    session_id: str = Field(min_length=8, max_length=64)
    message: str = Field(max_length=10_000)  # guardrail enforces the real limit
    persona_override: str | None = Field(default=None, max_length=4000)


class ErrorBody(BaseModel):
    code: Literal["needs_upload", "unsafe_url", "bad_request", "not_found"]
    detail: str


class NodeInfo(BaseModel):
    id: str
    label: str
    description: str


class AppConfig(BaseModel):
    nodes: list[NodeInfo]
    default_persona: str
    grounding_rules: str
    allow_prompt_override: bool
    model: str
    embedder: str
    browser_fetch: bool
    guardrails: dict[str, Any]  # rules per check + LLM judge prompt templates (policy.py)


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    llm_reachable: bool
    model: str
    embedder: str
