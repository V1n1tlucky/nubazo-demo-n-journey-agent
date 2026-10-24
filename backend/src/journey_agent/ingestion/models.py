"""Domain models produced by ingestion and consumed by RAG."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    CONTENT = "content"  # visible page text, grouped by heading
    META = "meta"  # <title>, meta description, JSON-LD
    PAYLOAD = "payload"  # JSON responses / embedded state (__NEXT_DATA__ …)


class CapturedPayload(BaseModel):
    """A JSON body observed while loading the page (XHR/fetch or inline state)."""

    url: str
    method: str = "GET"
    status: int = 200
    body: object


class RawPage(BaseModel):
    """What a fetcher returns before extraction."""

    url: str
    html: str
    payloads: list[CapturedPayload] = Field(default_factory=list)
    fetched_with: str  # "static" | "browser" | "upload-html" | "upload-har"


class Section(BaseModel):
    source: SourceType
    heading: str  # breadcrumb, e.g. "Products › Credit decisioning"
    text: str
    origin: str  # page URL or payload URL


class Chunk(BaseModel):
    id: str
    source: SourceType
    heading: str
    text: str
    origin: str


class PageDocument(BaseModel):
    """Normalised, page-scoped knowledge ready for indexing."""

    page_id: str
    url: str
    title: str
    description: str = ""
    sections: list[Section]
    fetched_with: str

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)

    @property
    def topics(self) -> list[str]:
        """Distinct section headings (H2 level) — used to suggest on-page questions."""
        seen: dict[str, None] = {}
        for s in self.sections:
            crumbs = s.heading.split(" › ") if s.heading else []
            if s.source is SourceType.CONTENT and len(crumbs) > 1:
                seen.setdefault(crumbs[1], None)
        return list(seen)[:8] or ([self.title] if self.title else [])
