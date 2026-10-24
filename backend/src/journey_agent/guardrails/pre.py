"""Input guardrails, run before any retrieval or LLM call.

All checks are deterministic and sub-millisecond. An LLM-as-judge pre-check
would roughly double latency on a laptop-class model for limited gain; the
page-relevance gate (in the retrieval node) and post-checks cover the rest.
"""

from __future__ import annotations

import re

from journey_agent.guardrails.pii import redact
from journey_agent.guardrails.types import Check, GuardrailResult, Verdict

_INJECTION = re.compile(
    r"|".join(
        [
            r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
            r"disregard (the |your )?(system|previous|above)",
            r"(reveal|show|print|repeat|output) (me )?(your|the) (system |initial |hidden )?(prompt|instructions)",
            r"you are now (?!talking)",
            r"\b(DAN|developer mode|jailbreak)\b",
            r"act as (?!a customer)",
            r"pretend (to be|you are)",
            r"</?(system|instructions?)>",
            r"new instructions\s*:",
        ]
    ),
    re.IGNORECASE,
)
_SMALLTALK = re.compile(
    r"^\s*(hi+|hey+|hello+|hiya|yo|namaste|good (morning|afternoon|evening)|"
    r"thanks?( you)?|thank you|ok(ay)?|cool|great|bye|goodbye)[\s!.?]*$",
    re.IGNORECASE,
)
_ABUSE = re.compile(r"\b(fuck|bitch|bastard|asshole|motherf\w*)\b", re.IGNORECASE)

REFUSAL_INJECTION = (
    "I can only help with questions about this page, and I can't change how I work. "
    "What would you like to know about it?"
)
REFUSAL_ABUSE = "Let's keep things respectful. How can I help you with this page?"


def smalltalk_reply(text: str, page_title: str, topics: list[str]) -> str:
    if re.search(r"thank", text, re.IGNORECASE):
        return "You're welcome! Anything else about this page I can help with?"
    if re.search(r"bye", text, re.IGNORECASE):
        return "Goodbye! Come back any time."
    hint = f" For example: {'; '.join(t.rstrip('.') for t in topics[:3])}." if topics else ""
    return f"Hi! I'm the assistant for **{page_title}**, here to help with this page.{hint}"


def run_pre_guardrails(
    text: str, *, max_chars: int, page_title: str, topics: list[str]
) -> GuardrailResult:
    checks: list[Check] = []
    stripped = text.strip()

    if not stripped:
        checks.append(Check(name="non_empty", verdict=Verdict.BLOCK, detail="empty message"))
        return GuardrailResult(
            text=text,
            checks=checks,
            route="blocked",
            blocked_reply="Please type a question about this page.",
        )
    checks.append(Check(name="non_empty", verdict=Verdict.PASS))

    if len(stripped) > max_chars:
        checks.append(
            Check(
                name="length", verdict=Verdict.BLOCK, detail=f"{len(stripped)} > {max_chars} chars"
            )
        )
        return GuardrailResult(
            text=text,
            checks=checks,
            route="blocked",
            blocked_reply=f"Please keep questions under {max_chars} characters.",
        )
    checks.append(Check(name="length", verdict=Verdict.PASS, detail=f"{len(stripped)} chars"))

    if m := _INJECTION.search(stripped):
        checks.append(
            Check(name="prompt_injection", verdict=Verdict.BLOCK, detail=f"matched '{m.group()}'")
        )
        return GuardrailResult(
            text=text, checks=checks, route="blocked", blocked_reply=REFUSAL_INJECTION
        )
    checks.append(Check(name="prompt_injection", verdict=Verdict.PASS))

    if _ABUSE.search(stripped):
        checks.append(Check(name="abuse", verdict=Verdict.BLOCK, detail="abusive language"))
        return GuardrailResult(
            text=text, checks=checks, route="blocked", blocked_reply=REFUSAL_ABUSE
        )
    checks.append(Check(name="abuse", verdict=Verdict.PASS))

    # Redact before the text reaches the LLM *or* the database.
    redacted, labels = redact(stripped)
    checks.append(
        Check(
            name="pii_redaction",
            verdict=Verdict.MODIFIED if labels else Verdict.PASS,
            detail=f"redacted {', '.join(labels)}" if labels else "",
        )
    )

    if _SMALLTALK.match(redacted):
        checks.append(Check(name="smalltalk", verdict=Verdict.PASS, detail="canned reply"))
        return GuardrailResult(
            text=redacted,
            checks=checks,
            route="smalltalk",
            blocked_reply=smalltalk_reply(redacted, page_title, topics),
        )

    return GuardrailResult(text=redacted, checks=checks)
