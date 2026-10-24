"""Human-readable guardrail policy, served to the console so every check can be inspected.

One entry per check, in the order the checks run (pre.py, post.py, llm_judge.py).
Keep in sync when a check is added or its behaviour changes.
"""

from __future__ import annotations

from journey_agent.config import Settings
from journey_agent.guardrails.llm_judge import INPUT_JUDGE_SYSTEM, OUTPUT_JUDGE_SYSTEM


def guardrail_policy(settings: Settings) -> dict[str, object]:
    llm = settings.llm_guardrails
    pre = [
        ("non_empty", "deterministic", 'A blank message gets "Please type your question."'),
        (
            "length",
            "deterministic",
            f"Messages over {settings.max_input_chars} characters are refused (JA_MAX_INPUT_CHARS).",
        ),
        (
            "prompt_injection",
            "deterministic",
            'Regex for attempts to override or reveal instructions: "ignore previous instructions", '
            '"disregard the system…", "reveal/print your prompt", "you are now…", DAN / developer '
            'mode / jailbreak, "act as…", "pretend to be…", <system> tags, "new instructions:". '
            "A match is refused without calling the LLM.",
        ),
        (
            "abuse",
            "deterministic",
            'Profanity word list. A match gets a polite "let\'s keep things respectful" reply.',
        ),
        (
            "pii_redaction",
            "deterministic",
            "PAN, Aadhaar, card numbers (Luhn-checked), Indian phone numbers and emails are replaced "
            "with [LABEL] before the text reaches the LLM, the judges or the database.",
        ),
        (
            "smalltalk",
            "deterministic",
            "Greetings, thanks and goodbyes get a canned reply naming the company; no LLM call.",
        ),
        (
            "reply_language",
            "deterministic",
            f"The reply language is picked from {settings.reply_languages} (JA_REPLY_LANGUAGES) and "
            "named explicitly in the prompt.",
        ),
        (
            "llm_input_judge",
            "llm" if llm else "disabled",
            "The LLM classifies the message (prompt below), knowing what the company offers and "
            "what the customer may be replying to. prompt_injection, harmful and abusive are "
            "refused. 'unrelated' is only flagged: Retrieval refuses it when the rerank score "
            f"is in the grey zone [{settings.rerank_hard_threshold}, {settings.rerank_threshold}), "
            "so a strong match on the page always wins. Runs only after the deterministic "
            "checks pass; fails open." + ("" if llm else " Disabled: JA_LLM_GUARDRAILS=false."),
        ),
    ]
    post = [
        ("non_empty", "deterministic", "An empty answer is replaced by the fallback reply."),
        (
            "abstention",
            "deterministic",
            "An answer that opens with \"I don't have / couldn't find…\" is replaced by the canned "
            "off-topic reply, so the model can't add outside knowledge after abstaining.",
        ),
        (
            "reply_language",
            "deterministic",
            "An answer of six or more words clearly not in the reply language falls back.",
        ),
        (
            "sensitive_solicitation",
            "deterministic",
            "Asking the customer to share/enter a PAN, Aadhaar, OTP, password, PIN, CVV, card or "
            "account number, net-banking or bank details falls back (phishing pattern).",
        ),
        (
            "system_prompt_leak",
            "deterministic",
            "Any 8-word overlap with the system prompt falls back.",
        ),
        (
            "pii_leak",
            "deterministic",
            "PII not published in the retrieved text is redacted from the answer.",
        ),
        (
            "link_allowlist",
            "deterministic",
            "Links that are neither in the retrieved text nor on the page's host are removed.",
        ),
        (
            "numeric_grounding",
            "deterministic",
            "Every number of two or more digits must appear in the retrieved text or the question, "
            "else the answer falls back (invented rates and fees).",
        ),
        (
            "grounding",
            "deterministic",
            "Mean best cosine similarity of answer sentences to the retrieved chunks must be at "
            + (
                f"least {settings.grounding_floor} (JA_GROUNDING_FLOOR). Genuine paraphrases and "
                "invented facts both score around 0.55, so above the floor the LLM output judge decides."
                if llm
                else f"least {settings.grounding_threshold} (JA_GROUNDING_THRESHOLD)."
            ),
        ),
        (
            "llm_output_judge",
            "llm" if llm else "disabled",
            "The LLM checks the answer against the reference text for unsupported facts, unrelated "
            "advice, credential requests, instruction leaks and promises (prompt below). A block "
            "falls back. Runs only after the deterministic checks pass; fails open."
            + ("" if llm else " Disabled: JA_LLM_GUARDRAILS=false."),
        ),
    ]

    def as_dicts(rows: list[tuple[str, str, str]]) -> list[dict[str, str]]:
        return [{"name": n, "kind": k, "rule": r} for n, k, r in rows]

    return {
        "llm_guardrails": llm,
        "pre": as_dicts(pre),
        "post": as_dicts(post),
        # COMPANY is filled in per page; the live prompt is in each run's trace.
        "judge_prompts": {"pre": INPUT_JUDGE_SYSTEM, "post": OUTPUT_JUDGE_SYSTEM},
    }
