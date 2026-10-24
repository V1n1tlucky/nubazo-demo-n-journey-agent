"""Deterministic PII detection and redaction (tuned for Indian BFSI users).

Regex + checksum is deliberately chosen over an NER model: it is fast,
explainable to auditors, and adds no memory pressure next to the LLM.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


def _luhn(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass(frozen=True)
class PiiPattern:
    label: str
    regex: re.Pattern[str]
    validate: Callable[[str], bool] = lambda _: True


PATTERNS: list[PiiPattern] = [
    PiiPattern("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    PiiPattern("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _luhn),
    PiiPattern("AADHAAR", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b")),
    PiiPattern("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    PiiPattern("PHONE", re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
]


@dataclass(frozen=True)
class PiiMatch:
    label: str
    value: str


def find_pii(text: str) -> list[PiiMatch]:
    found: list[PiiMatch] = []
    taken: list[tuple[int, int]] = []
    for pattern in PATTERNS:  # order matters: cards before the looser phone/aadhaar
        for m in pattern.regex.finditer(text):
            span = m.span()
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue
            if pattern.validate(m.group()):
                taken.append(span)
                found.append(PiiMatch(pattern.label, m.group()))
    return found


def redact(text: str, *, keep: set[str] | None = None) -> tuple[str, list[str]]:
    """Replace PII with `[LABEL]`. Values in `keep` (e.g. public contact details
    that appear on the page itself) are left untouched."""
    keep = keep or set()
    labels: list[str] = []
    for match in find_pii(text):
        if match.value in keep:
            continue
        text = text.replace(match.value, f"[{match.label}]")
        labels.append(match.label)
    return text, labels
