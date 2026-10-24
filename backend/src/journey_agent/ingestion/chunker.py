"""Split sections into overlapping, boundary-aware chunks.

Marketing pages have many tiny sections (a card = heading + one sentence).
Embedding each alone gives weak, context-free vectors, so consecutive sections
sharing a parent heading are packed together before splitting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from journey_agent.ingestion.models import Chunk, PageDocument, Section, SourceType

_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n")
_SEP = " › "


@dataclass
class _Group:
    source: SourceType
    heading: str
    origin: str
    lines: list[str] = field(default_factory=list)


def _group_key(section: Section) -> str:
    crumbs = section.heading.split(_SEP) if section.heading else []
    if section.source is SourceType.CONTENT:
        return _SEP.join(crumbs[:2])
    return crumbs[0] if crumbs else ""


def _pack(sections: list[Section]) -> list[_Group]:
    groups: list[_Group] = []
    for section in sections:
        key = _group_key(section)
        if not groups or (groups[-1].source, groups[-1].heading) != (section.source, key):
            groups.append(_Group(section.source, key, section.origin))
        leaf = section.heading.split(_SEP)[-1] if section.heading else ""
        if leaf and leaf != key.split(_SEP)[-1]:
            groups[-1].lines.append(f"{leaf}:")
        groups[-1].lines.append(section.text)
    return groups


def _split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    units = [u for u in _BOUNDARY.split(text) if u.strip()]
    chunks: list[str] = []
    buf = ""
    for unit in units:
        # Hard-wrap pathological single units (minified text, long lines).
        while len(unit) > size:
            chunks.append(unit[:size])
            unit = unit[size - overlap :]
        if buf and len(buf) + len(unit) + 1 > size:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf = f"{buf}\n{unit}".strip() if buf else unit
    if buf:
        chunks.append(buf)
    return chunks


def chunk_document(doc: PageDocument, *, size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for group in _pack(doc.sections):
        for piece in _split("\n".join(group.lines), size, overlap):
            out.append(
                Chunk(
                    id=f"{doc.page_id}:{len(out)}",
                    source=group.source,
                    heading=group.heading or doc.title,
                    text=piece,
                    origin=group.origin,
                )
            )
    return out
