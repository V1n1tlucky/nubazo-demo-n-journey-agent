"""Input guardrails, run before any retrieval or LLM call.

These checks are deterministic and sub-millisecond. Messages they let through
then go to the LLM input judge (llm_judge.py, in the pre_guardrail node), which
catches what patterns can't: paraphrased injection, politely worded harmful
requests, unrelated questions. guardrails/policy.py describes every check.
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

# Customer-facing: the assistant speaks for the company and never refers to "the page".
REFUSAL_INJECTION = (
    "I'm sorry, I can't help with that, and I can't change how I work. How can I help you today?"
)
REFUSAL_ABUSE = "Let's keep things respectful. How can I help you today?"
REFUSAL_HARMFUL = "I'm sorry, I can't help with that. Is there something else I can help you with?"


def smalltalk_reply(text: str, company: str, topics: list[str]) -> str:
    if re.search(r"thank", text, re.IGNORECASE):
        return "You're welcome! Is there anything else I can help you with?"
    if re.search(r"bye", text, re.IGNORECASE):
        return f"Goodbye, and thank you for visiting {company}!"
    hint = f" I can help with: {'; '.join(t.rstrip('.') for t in topics[:3])}." if topics else ""
    return f"Hi! I'm the virtual assistant for **{company}**. How can I help you today?{hint}"


def run_pre_guardrails(
    text: str, *, max_chars: int, company: str, topics: list[str]
) -> GuardrailResult:
    checks: list[Check] = []
    stripped = text.strip()

    if not stripped:
        checks.append(Check(name="non_empty", verdict=Verdict.BLOCK, detail="empty message"))
        return GuardrailResult(
            text=text,
            checks=checks,
            route="blocked",
            blocked_reply="Please type your question.",
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
            blocked_reply=smalltalk_reply(redacted, company, topics),
        )

    return GuardrailResult(text=redacted, checks=checks)
