"""Guardrail result types shared by pre- and post-checks."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Verdict(StrEnum):
    PASS = "pass"  # noqa: S105 - verdict, not a credential
    MODIFIED = "modified"  # text was changed (e.g. PII redacted) but allowed
    BLOCK = "block"
    SKIPPED = "skipped"  # an LLM judge that couldn't run (model down, unparseable reply)
    FLAG = "flag"  # a signal for a later node, not a decision (judge: "unrelated")


class Check(BaseModel):
    name: str
    verdict: Verdict
    detail: str = ""


class GuardrailResult(BaseModel):
    text: str  # possibly-modified text to carry forward
    checks: list[Check] = Field(default_factory=list)
    blocked_reply: str | None = None  # canned reply when blocked / short-circuited
    route: str = "continue"  # "continue" | "blocked" | "smalltalk" | "fallback" | "off_topic"

    @property
    def blocked(self) -> bool:
        return self.route != "continue"
