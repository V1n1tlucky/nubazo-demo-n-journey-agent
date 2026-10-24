"""Reply-language selection and verification.

Told only to "reply in the visitor's language", qwen2.5-7b answered an English
question in Portuguese 5 times in 8; naming the language ("Reply in English.")
fixed it in every trial. So the language is picked here, deterministically, from
the site's supported set (`JA_REPLY_LANGUAGES`). Open-set detection is unsafe on
short or romanised text: "gold loan ka interest rate kitna hai?" is Romanian at
0.95 confidence. Answers are long enough to verify against the open set.
"""

from __future__ import annotations

from functools import lru_cache

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

# Below this many words an answer ("Yes.", "₹5 Lakh") carries too little signal to judge.
_MIN_WORDS_TO_VERIFY = 6


def parse_language(name: str) -> Language:
    try:
        return Language.from_str(name.strip().lower())
    except ValueError:
        raise ValueError(f"unknown language {name!r}") from None


@lru_cache
def _site_detector(languages: tuple[Language, ...]) -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(*languages).build()


@lru_cache
def _open_detector() -> LanguageDetector:
    # High-accuracy mode (~130 MB). Low-accuracy mode called plain English
    # ("Welcome to Nubazo, your one-stop solution…") Latvian at 1.00.
    return LanguageDetectorBuilder.from_all_languages().build()


def reply_language(text: str, supported: list[str]) -> str:
    """The supported language closest to `text`; the first one when unsure."""
    langs = tuple(parse_language(n) for n in supported)
    if len(langs) == 1:
        return langs[0].name.title()
    detected = _site_detector(langs).detect_language_of(text)
    return (detected or langs[0]).name.title()


def is_wrong_language(answer: str, expected: str) -> bool:
    """True when a substantial answer is clearly not written in `expected`."""
    if len(answer.split()) < _MIN_WORDS_TO_VERIFY:
        return False
    confidence = _open_detector().compute_language_confidence(answer, parse_language(expected))
    return confidence < 0.1
