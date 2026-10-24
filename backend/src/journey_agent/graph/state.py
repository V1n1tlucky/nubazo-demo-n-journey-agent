"""LangGraph state for one agent turn."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Route = Literal[
    "continue",  # keep going through the pipeline
    "blocked",  # pre-guardrail refused the input
    "smalltalk",  # greeting/thanks answered without the LLM
    "off_topic",  # retrieval found nothing relevant on the page
    "fallback",  # post-guardrail rejected the LLM answer
    "error",  # infrastructure failure (LLM down, timeout…)
    "answered",  # terminal: grounded answer delivered
]


class NodeTrace(TypedDict):
    node: str
    status: Literal["ok", "blocked", "error"]
    latency_ms: float
    summary: str
    details: dict[str, Any]


class AgentState(TypedDict, total=False):
    # --- inputs ---------------------------------------------------------------
    run_id: str
    session_id: str
    page_id: str
    page_url: str
    page_title: str
    company_name: str  # LLM-identified (company.py); every reply names this company
    page_description: str
    greeting: str  # the LLM greeting the customer saw; the conversation's opening turn
    topics: list[str]
    user_message: str
    persona_override: str | None
    started_at: float

    # --- produced by nodes ----------------------------------------------------
    route: Route
    clean_message: str
    reply_language: str
    canned_reply: str | None
    history: list[tuple[str, str]]
    judge_relevance: str | None  # LLM input judge: on_topic / unrelated / None
    content_hits: list[dict[str, Any]]
    payload_hits: list[dict[str, Any]]
    top_score: float
    system_prompt: str
    messages: list[dict[str, str]]
    answer: str
    input_tokens: int
    output_tokens: int
    tokens_estimated: bool
    model: str
    grounding_score: float | None
    pre_checks: list[dict[str, Any]]
    post_checks: list[dict[str, Any]]
    final_answer: str
    error: str | None

    # Append-only: every node contributes one entry, streamed live to the UI.
    trace: Annotated[list[NodeTrace], operator.add]
