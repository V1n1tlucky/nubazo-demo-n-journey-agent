"""Graph nodes. Each is a small async function of (state) -> partial state.

Dependencies are injected through `AgentDeps` rather than globals, so the
graph is unit-testable with fake LLMs/embedders.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from journey_agent.config import Settings
from journey_agent.graph import prompts
from journey_agent.graph.state import AgentState, NodeTrace
from journey_agent.guardrails.language import reply_language
from journey_agent.guardrails.post import run_post_guardrails
from journey_agent.guardrails.pre import run_pre_guardrails
from journey_agent.ingestion.models import SourceType
from journey_agent.llm.client import ChatModel, Message, estimate_tokens
from journey_agent.persistence.repository import Repository, RunRecord
from journey_agent.rag.embeddings import Embedder
from journey_agent.rag.rerank import Reranker
from journey_agent.rag.store import Hit, IndexRegistry

log = logging.getLogger(__name__)


class NodeFn(Protocol):
    """LangGraph matches nodes on the parameter *name* `state`, so a bare Callable won't do."""

    __name__: str

    def __call__(self, state: AgentState) -> Awaitable[dict[str, Any]]: ...


@dataclass
class AgentDeps:
    settings: Settings
    registry: IndexRegistry
    embedder: Embedder
    llm: ChatModel
    repo: Repository
    reranker: Reranker | None = None


def traced(
    name: str,
) -> Callable[[Callable[..., Awaitable[tuple[dict[str, Any], str, str, dict[str, Any]]]]], NodeFn]:
    """Wrap a node so it returns its update plus a timed NodeTrace entry.

    Inner functions return (update, status, summary, details).
    """

    def decorator(
        fn: Callable[..., Awaitable[tuple[dict[str, Any], str, str, dict[str, Any]]]],
    ) -> NodeFn:
        async def wrapper(state: AgentState) -> dict[str, Any]:
            started = time.perf_counter()
            update, status, summary, details = await fn(state)
            entry: NodeTrace = {
                "node": name,
                "status": status,  # type: ignore[typeddict-item]
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "summary": summary,
                "details": details,
            }
            return {**update, "trace": [entry]}

        wrapper.__name__ = name
        return wrapper

    return decorator


def _hit_dict(hit: Hit, rerank: float | None = None) -> dict[str, Any]:
    return {
        "id": hit.chunk.id,
        "source": hit.chunk.source.value,
        "heading": hit.chunk.heading,
        "text": hit.chunk.text,
        "origin": hit.chunk.origin,
        "score": round(hit.score, 4),
        "rerank": None if rerank is None else round(rerank, 3),
    }


def make_nodes(deps: AgentDeps) -> dict[str, NodeFn]:
    s = deps.settings

    @traced("pre_guardrail")
    async def pre_guardrail(state: AgentState) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        result = run_pre_guardrails(
            state["user_message"],
            max_chars=s.max_input_chars,
            page_title=state.get("company_name") or state["page_title"],
            topics=state.get("topics", []),
        )
        checks = [c.model_dump() for c in result.checks]
        history = await deps.repo.session_history(
            state["session_id"], state["page_id"], s.history_turns
        )
        language = reply_language(result.text, s.reply_languages)
        update = {
            "clean_message": result.text,
            "reply_language": language,
            "route": result.route,
            "canned_reply": result.blocked_reply,
            "pre_checks": checks,
            "history": history,
        }
        status = "blocked" if result.route == "blocked" else "ok"
        failed = [c.name for c in result.checks if c.verdict != "pass"]
        summary = f"{result.route}" + (
            f" · {', '.join(failed)}" if failed else " · all checks passed"
        )
        details = {"checks": checks, "redacted_input": result.text, "reply_language": language}
        return update, status, summary, details

    @traced("retrieve")
    async def retrieve(state: AgentState) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        index = await deps.registry.get(state["page_id"])
        if index is None:
            return (
                {"route": "error", "error": "page index missing"},
                "error",
                "no index for page — re-ingest",
                {},
            )
        query = state["clean_message"]
        history = state.get("history", [])
        # Short follow-ups ("how much is it?") need the previous turn to retrieve well.
        if history and len(query.split()) < 8:
            query = f"{history[-1][0]} {query}"
        qvec = await deps.embedder.embed_query(query)
        content = index.search(
            qvec, k=s.top_k_content, sources={SourceType.CONTENT, SourceType.META}
        )
        payload = index.search(qvec, k=s.top_k_payload, sources={SourceType.PAYLOAD})
        hits = content + payload
        top = max((h.score for h in hits), default=0.0)
        # Cosine can't separate on- from off-page questions; the cross-encoder can, in the
        # languages it was trained on.
        if deps.reranker is not None and hits and state["reply_language"] in s.reranker_languages:
            texts = [f"{h.chunk.heading}\n{h.chunk.text}" for h in hits]
            scores = await deps.reranker.score(query, texts)
            reranks: list[float | None] = [*scores]
            gate, threshold, gate_name = max(scores), s.rerank_threshold, "rerank"
        else:
            reranks = [None] * len(hits)
            gate, threshold, gate_name = top, s.relevance_threshold, "cosine"
        relevant = gate >= threshold
        dicts = [_hit_dict(h, r) for h, r in zip(hits, reranks, strict=True)]
        update: dict[str, Any] = {
            "content_hits": dicts[: len(content)],
            "payload_hits": dicts[len(content) :],
            "top_score": top,
        }
        if not relevant:
            update["route"] = "off_topic"
            update["canned_reply"] = prompts.off_topic_reply(
                state.get("company_name") or state["page_title"], state.get("topics", [])
            )
        summary = (
            f"{len(content)} content + {len(payload)} payload chunks · {gate_name} {gate:.2f}"
            + ("" if relevant else f" < {threshold} (off-page)")
        )
        details = {
            "query": query,
            "gate": gate_name,
            "threshold": threshold,
            "content_hits": update["content_hits"],
            "payload_hits": update["payload_hits"],
        }
        return update, "ok" if relevant else "blocked", summary, details

    @traced("prompt_builder")
    async def prompt_builder(state: AgentState) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        persona = prompts.DEFAULT_PERSONA
        if s.allow_prompt_override and state.get("persona_override"):
            persona = state["persona_override"] or persona
        system = prompts.build_system_prompt(
            persona, title=state["page_title"], url=state["page_url"]
        )

        budget = s.max_context_chars
        content_parts: list[str] = []
        data_parts: list[str] = []
        used: list[str] = []
        for hit in state.get("content_hits", []):
            block = f"[{len(used) + 1}] {hit['heading']}\n{hit['text']}"
            if len(block) > budget:
                break
            content_parts.append(block)
            used.append(hit["id"])
            budget -= len(block)
        for hit in state.get("payload_hits", []):
            if hit["score"] < s.relevance_threshold:
                continue  # weak payload matches are noise for a small model
            block = f"[{len(used) + 1}] source: {hit['origin']}\n{hit['text']}"
            if len(block) > budget:
                break
            data_parts.append(block)
            used.append(hit["id"])
            budget -= len(block)

        content_block, data_block = "\n\n".join(content_parts), "\n\n".join(data_parts)
        messages = [{"role": "system", "content": system}]
        for user, ai in state.get("history", []):
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": ai}]
        messages.append(
            {
                "role": "user",
                "content": prompts.build_user_turn(
                    state["clean_message"], content_block, data_block, state["reply_language"]
                ),
            }
        )

        est = estimate_tokens("".join(m["content"] for m in messages))
        summary = (
            f"{len(content_parts)} content + {len(data_parts)} payload blocks · "
            f"{len(state.get('history', []))} history turns · ~{est} tokens"
        )
        details = {
            "system_prompt": system,
            "retrieval_context": content_block,
            "payload_inputs": data_block,
            "history_turns": len(state.get("history", [])),
            "reply_language": state["reply_language"],
            "estimated_prompt_tokens": est,
            "chunks_used": used,
        }
        return {"system_prompt": system, "messages": messages}, "ok", summary, details

    @traced("generate")
    async def generate(state: AgentState) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        try:
            result = await deps.llm.complete(
                [Message(m["role"], m["content"]) for m in state["messages"]]
            )
        except Exception as exc:
            log.exception("event=llm_error run_id=%s", state["run_id"])
            return (
                {
                    "route": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "canned_reply": prompts.ERROR_REPLY,
                },
                "error",
                f"LLM call failed: {type(exc).__name__}",
                {"error": str(exc)},
            )
        update = {
            "answer": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "tokens_estimated": result.tokens_estimated,
            "model": result.model,
        }
        summary = f"{result.model} · {result.input_tokens} in / {result.output_tokens} out" + (
            " (est.)" if result.tokens_estimated else ""
        )
        return (
            update,
            "ok",
            summary,
            {"raw_answer": result.text, "llm_latency_ms": round(result.latency_ms, 1)},
        )

    @traced("post_guardrail")
    async def post_guardrail(state: AgentState) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        contexts = [
            h["text"] for h in state.get("content_hits", []) + state.get("payload_hits", [])
        ]
        result, score = await run_post_guardrails(
            state.get("answer", ""),
            contexts=contexts,
            system_prompt=state["system_prompt"],
            page_url=state["page_url"],
            question=state["clean_message"],
            reply_language=state["reply_language"],
            embedder=deps.embedder,
            grounding_threshold=s.grounding_threshold,
        )
        checks = [c.model_dump() for c in result.checks]
        update: dict[str, Any] = {
            "post_checks": checks,
            "grounding_score": score,
            "answer": result.text,
        }
        if result.route == "off_topic":
            update["route"] = "off_topic"
            update["canned_reply"] = prompts.off_topic_reply(
                state.get("company_name") or state["page_title"], state.get("topics", [])
            )
        elif result.blocked:
            update["route"] = "fallback"
            update["canned_reply"] = result.blocked_reply
        failed = [c.name for c in result.checks if c.verdict != "pass"]
        summary = f"grounding {score:.2f}" + (
            f" · {', '.join(failed)}" if failed else " · all checks passed"
        )
        return (
            update,
            "blocked" if result.blocked else "ok",
            summary,
            {"checks": checks, "final_text": result.text},
        )

    @traced("write")
    async def write(state: AgentState) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        route = state.get("route", "continue")
        if route == "continue":
            route, final = "answered", state.get("answer", "")
        else:
            final = state.get("canned_reply") or prompts.ERROR_REPLY

        node_latency = {t["node"]: t["latency_ms"] for t in state.get("trace", [])}
        total = round((time.perf_counter() - state["started_at"]) * 1000, 1)
        record = RunRecord(
            run_id=state["run_id"],
            session_id=state["session_id"],
            page_id=state["page_id"],
            route=route,
            user_message=state.get("clean_message") or "",  # redacted; raw input never stored
            ai_response=final,
            model=state.get("model"),
            system_prompt=state.get("system_prompt"),
            prompt_messages=state.get("messages", []),
            retrieved=state.get("content_hits", []) + state.get("payload_hits", []),
            guardrails={"pre": state.get("pre_checks", []), "post": state.get("post_checks", [])},
            node_latency_ms=node_latency,
            input_tokens=state.get("input_tokens", 0),
            output_tokens=state.get("output_tokens", 0),
            tokens_estimated=state.get("tokens_estimated", False),
            total_latency_ms=total,
            grounding_score=state.get("grounding_score"),
            error=state.get("error"),
        )
        status, note = "ok", "persisted"
        try:
            await deps.repo.insert_run(record)
        except Exception as exc:  # never fail the customer reply on an audit write
            log.exception("event=run_write_failed run_id=%s", state["run_id"])
            status, note = "error", f"write failed: {exc}"
        details = record.model_dump(
            include={
                "run_id",
                "route",
                "model",
                "input_tokens",
                "output_tokens",
                "tokens_estimated",
                "total_latency_ms",
                "node_latency_ms",
                "grounding_score",
                "user_message",
                "ai_response",
            }
        )
        summary = (
            f"{note} · {record.input_tokens}+{record.output_tokens} tokens · {total:.0f} ms total"
        )
        return {"route": route, "final_answer": final}, status, summary, details

    return {
        "pre_guardrail": pre_guardrail,
        "retrieve": retrieve,
        "prompt_builder": prompt_builder,
        "generate": generate,
        "post_guardrail": post_guardrail,
        "write": write,
    }
