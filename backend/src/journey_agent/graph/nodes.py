"""Graph nodes. Each is a small async function of (state) -> partial state.

Dependencies are injected through `AgentDeps` rather than globals, so the
graph is unit-testable with fake LLMs/embedders.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from journey_agent.config import Settings
from journey_agent.graph import prompts
from journey_agent.graph.state import AgentState, NodeTrace
from journey_agent.guardrails.language import reply_language
from journey_agent.guardrails.llm_judge import JudgeResult, judge_input, judge_output
from journey_agent.guardrails.post import FALLBACK, run_post_guardrails
from journey_agent.guardrails.pre import (
    REFUSAL_ABUSE,
    REFUSAL_HARMFUL,
    REFUSAL_INJECTION,
    run_pre_guardrails,
)
from journey_agent.guardrails.types import Check, Verdict
from journey_agent.ingestion.chunker import low_signal
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


def _topics(state: AgentState) -> list[str]:
    # Pages without headings fall back to their title as the only "topic"; never offer that.
    title = state["page_title"].strip().lower()
    return [t for t in state.get("topics", []) if t.strip().lower() != title]


def _judge_check(name: str, judge: JudgeResult, *, flag: bool = False) -> dict[str, Any]:
    verdict = {"block": Verdict.BLOCK, "skipped": Verdict.SKIPPED}.get(judge.verdict, Verdict.PASS)
    detail = f"{judge.category}: {judge.reason}" if judge.category else judge.reason
    if flag:  # judged unrelated: a signal for the retrieval gate, not a refusal
        verdict, detail = Verdict.FLAG, f"{detail} (relevance decided by retrieval)"
    return Check(name=name, verdict=verdict, detail=detail).model_dump()


# Messages that only make sense as a reply to the previous turn.
_FOLLOW_UP = re.compile(
    r"^\s*(yes|yeah|yep|yup|ok(ay)?|sure|please|and|also|but|so|then|what about|how about|"
    r"tell me more|more|go on|continue|it|its|it's|that|this|those|these|they|them|same|why)\b",
    re.IGNORECASE,
)


def _is_follow_up(message: str) -> bool:
    """Short is not the same as follow-up: "how do I apply?" stands alone. Context from the
    previous turn or the greeting is only added for replies ("yes, tell me the process",
    "and if I'm self-employed?") or fragments of three words or fewer."""
    return bool(_FOLLOW_UP.match(message)) or len(message.split()) <= 3


def _previous_assistant(state: AgentState, history: list[tuple[str, str]]) -> str:
    """What the customer may be replying to: the last answer, or the greeting at the start."""
    return history[-1][1] if history else state.get("greeting", "")


def make_nodes(deps: AgentDeps) -> dict[str, NodeFn]:
    s = deps.settings

    @traced("pre_guardrail")
    async def pre_guardrail(state: AgentState) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        result = run_pre_guardrails(
            state["user_message"],
            max_chars=s.max_input_chars,
            company=state["company_name"],
            topics=_topics(state),
        )
        checks = [c.model_dump() for c in result.checks]
        history = await deps.repo.session_history(
            state["session_id"], state["page_id"], s.history_turns
        )
        language = reply_language(result.text, s.reply_languages)
        route, reply = result.route, result.blocked_reply
        judge: JudgeResult | None = None
        relevance: str | None = None  # the judge's on_topic/unrelated call, for the retrieve gate
        if route == "continue" and s.llm_guardrails:  # only what regex already let through
            judge = await judge_input(
                deps.llm,
                company=state["company_name"],
                offering=f"{state['page_title']}. {state.get('page_description', '')}".strip(),
                topics=_topics(state),
                previous=_previous_assistant(state, history),
                message=result.text,
            )
            safety = {
                "prompt_injection": REFUSAL_INJECTION,
                "abusive": REFUSAL_ABUSE,
                "harmful": REFUSAL_HARMFUL,
            }
            if judge.blocked and judge.category in safety:
                route, reply = "blocked", safety[judge.category]
                checks.append(_judge_check("llm_input_judge", judge))
            else:
                # Relevance is not the judge's to refuse alone: two independent gates double
                # the wrong refusals ("tell me about gold loan" on a gold-loan lender's site).
                # Retrieval decides, and uses this call only to break a tie.
                if judge.verdict != "skipped":
                    relevance = "unrelated" if judge.blocked else "on_topic"
                checks.append(_judge_check("llm_input_judge", judge, flag=judge.blocked))
        update = {
            "clean_message": result.text,
            "reply_language": language,
            "route": route,
            "canned_reply": reply,
            "pre_checks": checks,
            "history": history,
            "judge_relevance": relevance,
        }
        status = "blocked" if route in {"blocked", "off_topic"} else "ok"
        failed = [c["name"] for c in checks if c["verdict"] != "pass"]
        summary = f"{route}" + (f" · {', '.join(failed)}" if failed else " · all checks passed")
        details = {
            "checks": checks,
            "redacted_input": result.text,
            "reply_language": language,
            "llm_judge": judge.details() if judge else None,
        }
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
        question = state["clean_message"]
        history = state.get("history", [])
        # Context for short follow-ups ("and if self-employed?", "yes, tell me the process"):
        # the previous question, or the greeting the customer is replying to.
        context = history[-1][0] if history else state.get("greeting", "")
        variants = [question]
        if context and _is_follow_up(question):
            variants.append(f"{context} {question}")
        use_rerank = deps.reranker is not None and state["reply_language"] in s.reranker_languages

        # FAISS proposes a pool (pool x top_k); junk chunks are dropped; the cross-encoder
        # picks the best. Cosine ranking alone buries the right chunk under noise.
        best: dict[str, tuple[Hit, float | None]] = {}
        tried: list[str] = []
        for q in variants:
            tried.append(q)
            qvec = await deps.embedder.embed_query(q)
            pool = index.search(
                qvec,
                k=s.top_k_content * s.candidate_pool,
                sources={SourceType.CONTENT, SourceType.META},
            ) + index.search(
                qvec, k=s.top_k_payload * s.candidate_pool, sources={SourceType.PAYLOAD}
            )
            pool = [h for h in pool if not low_signal(h.chunk.text)]
            scores: list[float | None] = [None] * len(pool)
            if use_rerank and pool and deps.reranker is not None:
                scores = list(
                    await deps.reranker.score(
                        q, [f"{h.chunk.heading}\n{h.chunk.text}" for h in pool]
                    )
                )
            for h, r in zip(pool, scores, strict=True):
                prev = best.get(h.chunk.id)
                if prev is None or (r if r is not None else h.score) > (
                    prev[1] if prev[1] is not None else prev[0].score
                ):
                    best[h.chunk.id] = (h, r)
            gate_now = max(
                ((r if r is not None else h.score) for h, r in best.values()), default=-99.0
            )
            if gate_now >= (s.rerank_threshold if use_rerank else s.relevance_threshold):
                break  # the question as asked is clearly answerable; no need for context

        ranked = sorted(
            best.values(), key=lambda hr: hr[1] if hr[1] is not None else hr[0].score, reverse=True
        )
        content = [hr for hr in ranked if hr[0].chunk.source is not SourceType.PAYLOAD][
            : s.top_k_content
        ]
        payload = [hr for hr in ranked if hr[0].chunk.source is SourceType.PAYLOAD][
            : s.top_k_payload
        ]
        top = max((h.score for h, _ in ranked), default=0.0)
        judge = state.get("judge_relevance")
        if use_rerank:
            gate = max((r for _, r in ranked if r is not None), default=-99.0)
            soft, hard, gate_name = s.rerank_threshold, s.rerank_hard_threshold, "rerank"
            if gate >= soft:
                relevant, why = True, f"≥ {soft}"
            elif gate < hard:
                relevant, why = False, f"< {hard} (off-page)"
            elif judge == "on_topic":
                relevant, why = True, f"grey zone [{hard}, {soft}), LLM judge: on_topic"
            else:
                relevant, why = (
                    False,
                    f"grey zone [{hard}, {soft}), LLM judge: {judge or 'n/a'} (off-page)",
                )
        else:
            gate, soft, hard, gate_name = (
                top,
                s.relevance_threshold,
                s.relevance_threshold,
                "cosine",
            )
            relevant = gate >= soft
            why = f"≥ {soft}" if relevant else f"< {soft} (off-page)"
        update: dict[str, Any] = {
            "content_hits": [_hit_dict(h, r) for h, r in content],
            "payload_hits": [_hit_dict(h, r) for h, r in payload],
            "top_score": top,
        }
        if not relevant:
            update["route"] = "off_topic"
            update["canned_reply"] = prompts.off_topic_reply(state["company_name"], _topics(state))
        summary = (
            f"{len(content)} content + {len(payload)} payload chunks · {gate_name} {gate:.2f} {why}"
        )
        details = {
            "query": tried[-1],
            "queries_tried": tried,
            "gate": gate_name,
            "threshold": soft,
            "hard_threshold": hard if use_rerank else None,
            "judge_relevance": judge,
            "decision": why,
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
            persona, company=state["company_name"], title=state["page_title"], url=state["page_url"]
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
            weak = (
                hit["rerank"] < s.rerank_threshold
                if hit.get("rerank") is not None
                else hit["score"] < s.relevance_threshold
            )
            if weak:
                continue  # weak payload matches are noise for a small model
            block = f"[{len(used) + 1}] source: {hit['origin']}\n{hit['text']}"
            if len(block) > budget:
                break
            data_parts.append(block)
            used.append(hit["id"])
            budget -= len(block)

        content_block, data_block = "\n\n".join(content_parts), "\n\n".join(data_parts)
        messages = [{"role": "system", "content": system}]
        greeting = state.get("greeting", "")
        if greeting and len(state.get("history", [])) < s.history_turns:
            # The conversation's real opening: "yes, please" may be a reply to it.
            messages.append({"role": "assistant", "content": greeting})
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
            grounding_threshold=s.grounding_floor if s.llm_guardrails else s.grounding_threshold,
        )
        checks = [c.model_dump() for c in result.checks]
        update: dict[str, Any] = {
            "post_checks": checks,
            "grounding_score": score,
            "answer": result.text,
        }
        blocked = result.blocked
        if result.route == "off_topic":
            update["route"] = "off_topic"
            update["canned_reply"] = prompts.off_topic_reply(state["company_name"], _topics(state))
        elif result.blocked:
            update["route"] = "fallback"
            update["canned_reply"] = result.blocked_reply
        judge: JudgeResult | None = None
        if not blocked and s.llm_guardrails:  # only answers the deterministic checks released
            judge = await judge_output(
                deps.llm,
                company=state["company_name"],
                question=state["clean_message"],
                answer=result.text,
                contexts=contexts,
            )
            checks.append(_judge_check("llm_output_judge", judge))
            if judge.blocked:
                blocked = True
                update["route"] = "fallback"
                update["canned_reply"] = FALLBACK
        update["post_checks"] = checks
        failed = [c["name"] for c in checks if c["verdict"] != "pass"]
        summary = f"grounding {score:.2f}" + (
            f" · {', '.join(failed)}" if failed else " · all checks passed"
        )
        return (
            update,
            "blocked" if blocked else "ok",
            summary,
            {
                "checks": checks,
                "final_text": update.get("canned_reply") if blocked else result.text,
                "llm_judge": judge.details() if judge else None,
            },
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
