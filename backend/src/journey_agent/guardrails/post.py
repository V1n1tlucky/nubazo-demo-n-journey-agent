"""Output guardrails, run on the full answer before the user sees it.

Because these checks need the complete answer, the agent does not stream
tokens to the customer: streaming first and retracting later would already
have shown the unsafe text. The live UI streams *node progress* instead.
"""

from __future__ import annotations

import re

import numpy as np

from journey_agent.guardrails.language import is_wrong_language
from journey_agent.guardrails.pii import find_pii, redact
from journey_agent.guardrails.types import Check, GuardrailResult, Verdict
from journey_agent.rag.embeddings import Embedder

_URL = re.compile(r"https?://[^\s)\]>\"']+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# An assistant must never ask a customer for credentials or identity numbers —
# that is exactly what phishing looks like, and small models do it unprompted.
_SOLICIT = re.compile(
    r"\b(provide|share|send|enter|give|tell|type|confirm)\b[^.?!\n]{0,60}\b("
    r"PAN|Aadhaar|OTP|one[- ]time password|password|passcode|PIN|CVV|card number|"
    r"account number|net ?banking|UPI PIN|date of birth|bank details)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Honest "not on this page" answers are always grounded by definition.
_ABSTAIN = re.compile(
    r"(don't|do not|couldn't|could not|can't|cannot) (find|see|have)|not (mentioned|covered|available)"
    r"|no information|only (help|answer)"
    r"|नहीं मिल|जानकारी नहीं|उपलब्ध नहीं",  # Hindi: "didn't find", "no information", "not available"
    re.IGNORECASE,
)
FALLBACK = (
    "I couldn't find a reliable answer to that on this page. "
    "Could you rephrase, or ask about something shown on the page?"
)


def abstains(answer: str) -> bool:
    """True when the answer opens by saying the page doesn't cover the question.

    Small models abstain and then answer anyway from world knowledge ("…check
    sports news"), so an abstention replaces the whole answer with the canned
    off-page reply rather than being passed through.
    """
    first = next((s for s in _SENTENCE.split(answer) if s.strip()), "")
    return bool(_ABSTAIN.search(first))


def _shingles(text: str, n: int = 8) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _numbers(text: str) -> set[str]:
    # Single digits are mostly list markers/counts ("1.", "two or 3"); too noisy to police.
    return {n.replace(",", "") for n in _NUMBER.findall(text) if len(n.replace(",", "")) > 1}


def unsupported_numbers(answer: str, sources: str) -> set[str]:
    """Numbers (rates, fees, amounts, limits) in the answer that no source contains.

    Embedding similarity cannot tell "fee is 1.5%" from "fee is 4.5%" — both are
    on-topic. For BFSI pages an invented number is the costliest hallucination,
    so it gets a deterministic check.
    """
    return _numbers(answer) - _numbers(sources)


async def grounding_score(answer: str, contexts: list[str], embedder: Embedder) -> float:
    """Mean over answer sentences of max cosine similarity to any context chunk."""
    sentences = [
        s for s in _SENTENCE.split(answer) if len(s.split()) >= 4 and not _ABSTAIN.search(s)
    ]
    if not sentences or not contexts:
        return 1.0
    sent_vecs = await embedder.embed_documents(sentences)
    ctx_vecs = await embedder.embed_documents(contexts)
    sims = sent_vecs @ ctx_vecs.T
    return float(np.mean(sims.max(axis=1)))


async def run_post_guardrails(
    answer: str,
    *,
    contexts: list[str],
    system_prompt: str,
    page_url: str,
    question: str = "",
    reply_language: str | None = None,
    embedder: Embedder,
    grounding_threshold: float,
) -> tuple[GuardrailResult, float]:
    checks: list[Check] = []
    text = answer.strip()

    if not text:
        checks.append(Check(name="non_empty", verdict=Verdict.BLOCK))
        return GuardrailResult(
            text=FALLBACK, checks=checks, route="fallback", blocked_reply=FALLBACK
        ), 0.0
    checks.append(Check(name="non_empty", verdict=Verdict.PASS))

    if abstains(text):
        checks.append(
            Check(name="abstention", verdict=Verdict.MODIFIED, detail="canned off-page reply used")
        )
        return GuardrailResult(text=text, checks=checks, route="off_topic"), 1.0
    checks.append(Check(name="abstention", verdict=Verdict.PASS))

    if reply_language and is_wrong_language(text, reply_language):
        checks.append(
            Check(name="reply_language", verdict=Verdict.BLOCK, detail=f"not {reply_language}")
        )
        return GuardrailResult(
            text=FALLBACK, checks=checks, route="fallback", blocked_reply=FALLBACK
        ), 0.0
    checks.append(Check(name="reply_language", verdict=Verdict.PASS, detail=reply_language or ""))

    if m := _SOLICIT.search(text):
        checks.append(
            Check(name="sensitive_solicitation", verdict=Verdict.BLOCK, detail=f"'{m.group()}'")
        )
        return GuardrailResult(
            text=FALLBACK, checks=checks, route="fallback", blocked_reply=FALLBACK
        ), 0.0
    checks.append(Check(name="sensitive_solicitation", verdict=Verdict.PASS))

    leaked = _shingles(system_prompt) & _shingles(text)
    if leaked:
        checks.append(
            Check(
                name="system_prompt_leak",
                verdict=Verdict.BLOCK,
                detail=f"{len(leaked)} overlapping 8-grams",
            )
        )
        return GuardrailResult(
            text=FALLBACK, checks=checks, route="fallback", blocked_reply=FALLBACK
        ), 0.0
    checks.append(Check(name="system_prompt_leak", verdict=Verdict.PASS))

    # PII that is published on the page (e.g. a support email) is fine to repeat.
    joined = "\n".join(contexts)
    public = {m.value for m in find_pii(joined)}
    text, labels = redact(text, keep=public)
    checks.append(
        Check(
            name="pii_leak",
            verdict=Verdict.MODIFIED if labels else Verdict.PASS,
            detail=f"redacted {', '.join(labels)}" if labels else "",
        )
    )

    # Links must come from the page; small models love inventing plausible URLs.
    host = re.sub(r"^https?://([^/]+).*", r"\1", page_url)
    invented = [u for u in _URL.findall(text) if u not in joined and host not in u]
    for url in invented:
        text = text.replace(url, "[link removed]")
    checks.append(
        Check(
            name="link_allowlist",
            verdict=Verdict.MODIFIED if invented else Verdict.PASS,
            detail=f"removed {len(invented)} unknown link(s)" if invented else "",
        )
    )

    invented_numbers = unsupported_numbers(text, f"{joined}\n{question}")
    if invented_numbers:
        checks.append(
            Check(
                name="numeric_grounding",
                verdict=Verdict.BLOCK,
                detail=f"not on page: {', '.join(sorted(invented_numbers))}",
            )
        )
        return GuardrailResult(
            text=FALLBACK, checks=checks, route="fallback", blocked_reply=FALLBACK
        ), 0.0
    checks.append(Check(name="numeric_grounding", verdict=Verdict.PASS))

    score = await grounding_score(text, contexts, embedder)
    if score < grounding_threshold:
        checks.append(
            Check(
                name="grounding",
                verdict=Verdict.BLOCK,
                detail=f"score {score:.2f} < {grounding_threshold}",
            )
        )
        return GuardrailResult(
            text=FALLBACK, checks=checks, route="fallback", blocked_reply=FALLBACK
        ), score
    checks.append(Check(name="grounding", verdict=Verdict.PASS, detail=f"score {score:.2f}"))

    return GuardrailResult(text=text, checks=checks), score
