from __future__ import annotations

import pytest

from journey_agent.config import Settings
from journey_agent.guardrails.language import is_wrong_language, reply_language

SITE = ["English", "Hindi"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Tell me about “Other Loan”", "English"),
        ("मुझे गोल्ड लोन के बारे में बताइए", "Hindi"),
        # Open-set detection calls these Romanian / Afrikaans / Turkish with high confidence.
        ("gold loan ka interest rate kitna hai?", "English"),
        ("ok thanks", "English"),
        ("eligibility?", "English"),
        # Unsupported languages get the site default rather than a guess.
        ("Quais empréstimos vocês oferecem?", "English"),
    ],
)
def test_reply_language_is_chosen_from_site_languages(question: str, expected: str) -> None:
    assert reply_language(question, SITE) == expected


def test_single_language_site_always_uses_it() -> None:
    assert reply_language("मुझे गोल्ड लोन के बारे में बताइए", ["English"]) == "English"


@pytest.mark.parametrize(
    ("answer", "expected", "wrong"),
    [
        (
            "Não encontrei informações específicas sobre Outros Empréstimos na página.",
            "English",
            True,
        ),
        ("- Gold Loan\n- Loan Against Property\n- MSME Loan\n- Machinery Loan", "English", False),
        (
            "मुझे पेज पर जानकारी नहीं मिली। आप Gold Loan या Personal Loan के बारे में पूछ सकते हैं।",
            "Hindi",
            False,
        ),
        ("Sim, é possível.", "English", False),  # too short to judge
        # Low-accuracy detection called this Latvian 1.00 and blocked it.
        (
            "Welcome to Nubazo, your one-stop solution for unified data and AI in banking. "
            "Let's optimize your decisions together.",
            "English",
            False,
        ),
    ],
)
def test_is_wrong_language(answer: str, expected: str, wrong: bool) -> None:
    assert is_wrong_language(answer, expected) is wrong


def test_unknown_reply_language_fails_at_startup() -> None:
    with pytest.raises(ValueError, match="Klingon"):
        Settings(reply_languages=["English", "Klingon"], _env_file=None)  # type: ignore[call-arg]
