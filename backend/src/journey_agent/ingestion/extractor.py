"""Turn raw HTML + JSON payloads into heading-scoped text sections.

Design choices:
* Text is grouped under its heading breadcrumb so retrieved chunks carry
  their context ("Pricing › Enterprise") — this matters far more for answer
  quality on a 7B model than the embedding model choice.
* JSON is flattened to `path: value` lines; small models read that reliably,
  whereas raw nested JSON wastes context and confuses them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from journey_agent.guardrails.pii import redact
from journey_agent.ingestion.models import (
    CapturedPayload,
    PageDocument,
    RawPage,
    Section,
    SourceType,
)

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}
BLOCKS = {
    "p",
    "li",
    "td",
    "th",
    "dd",
    "dt",
    "blockquote",
    "figcaption",
    "summary",
    "label",
    "button",
    "div",
    "section",
    "article",
    "aside",
    "header",
    "footer",
    "main",
    "nav",
    "ul",
    "ol",
    "table",
    "tr",
    "form",
    "details",
    "pre",
    "address",
    *HEADINGS,
}
DROP = ["script", "style", "noscript", "template", "svg", "iframe", "canvas", "object"]
INLINE_STATE_IDS = {"__NEXT_DATA__", "__NUXT_DATA__", "__APOLLO_STATE__"}
_WS = re.compile(r"\s+")
_TOKENISH = re.compile(r"^[A-Za-z0-9+/=_\-]{60,}$")  # hashes, base64, JWTs
MAX_PAYLOAD_LINES = 400


def page_id_for(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def extract(raw: RawPage) -> PageDocument:
    soup = BeautifulSoup(raw.html, "lxml")
    sections: list[Section] = []

    title = _clean(soup.title.get_text()) if soup.title else ""
    description = ""
    meta_lines: list[str] = []
    for meta in soup.find_all("meta"):
        key = meta.get("name") or meta.get("property") or ""
        content = _clean(str(meta.get("content", "")))
        if not content or not isinstance(key, str):
            continue
        if key in {"description", "og:description"} and not description:
            description = content
        if key in {"description", "keywords", "og:title", "og:description", "og:site_name"}:
            meta_lines.append(f"{key}: {content}")
    if title or meta_lines:
        sections.append(
            Section(
                source=SourceType.META,
                heading="Page metadata",
                text="\n".join([f"title: {title}", *meta_lines]),
                origin=raw.url,
            )
        )

    # Structured data embedded in the HTML is often the richest "payload".
    for script in soup.find_all("script"):
        stype = script.get("type", "")
        sid = script.get("id", "")
        if stype == "application/ld+json" or sid in INLINE_STATE_IDS or stype == "application/json":
            try:
                body = json.loads(script.string or "")
            except json.JSONDecodeError:
                continue
            source = SourceType.META if stype == "application/ld+json" else SourceType.PAYLOAD
            sections.extend(
                _payload_sections(
                    CapturedPayload(url=f"{raw.url}#{sid or stype}", body=body), source
                )
            )

    for tag in soup(DROP):
        tag.decompose()
    body = soup.body or soup
    sections.extend(_content_sections(body, raw.url))

    for payload in raw.payloads:
        sections.extend(_payload_sections(payload, SourceType.PAYLOAD))

    return PageDocument(
        page_id=page_id_for(raw.url),
        url=raw.url,
        title=title or raw.url,
        description=description,
        sections=sections,
        fetched_with=raw.fetched_with,
    )


def _has_block_child(tag: Tag) -> bool:
    return tag.find(lambda t: isinstance(t, Tag) and t.name in BLOCKS) is not None


def _content_sections(body: Tag, origin: str) -> list[Section]:
    crumbs: dict[int, str] = {}
    current: list[str] = []
    seen: set[str] = set()
    out: list[Section] = []

    def breadcrumb() -> str:
        return " › ".join(crumbs[k] for k in sorted(crumbs))

    def flush() -> None:
        if current:
            out.append(
                Section(
                    source=SourceType.CONTENT,
                    heading=breadcrumb(),
                    text="\n".join(current),
                    origin=origin,
                )
            )
            current.clear()

    for el in body.descendants:
        if not isinstance(el, Tag) or el.name not in BLOCKS:
            continue
        if el.name in HEADINGS:
            text = _clean(el.get_text(" "))
            if not text:
                continue
            flush()
            level = HEADINGS[el.name]
            crumbs = {k: v for k, v in crumbs.items() if k < level}
            crumbs[level] = text[:120]
            continue
        if any(p.name in HEADINGS for p in el.parents if isinstance(p, Tag)):
            continue
        if _has_block_child(el):
            # Capture loose text in containers, e.g. <div>Intro <div>…</div></div>.
            text = _clean(" ".join(s for s in el.children if isinstance(s, NavigableString)))
        else:
            text = _clean(el.get_text(" "))
        # Navigation and footers repeat; keep the first occurrence only.
        if len(text) < 3 or text in seen:
            continue
        seen.add(text)
        current.append(text)
    flush()
    return out


def _flatten(node: Any, path: str = "") -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, value in enumerate(node[:50]):
            yield from _flatten(value, f"{path}[{i}]")
    elif node is not None and node != "":
        text = _clean(str(node))
        if not _TOKENISH.match(text):
            yield f"{path}: {text[:300]}"


def _redact_payload(lines: list[str]) -> str:
    """Payloads (esp. HARs captured while logged in) can hold one user's account data.
    The index is shared by every visitor, so personal identifiers never enter it."""
    return redact("\n".join(lines))[0]


def _payload_sections(payload: CapturedPayload, source: SourceType) -> list[Section]:
    """One section for top-level scalars, one per nested top-level key."""
    body = payload.body
    groups: list[tuple[str, Any]] = (
        list(body.items()) if isinstance(body, dict) else [("root", body)]
    )
    scalars = {k: v for k, v in groups if not isinstance(v, dict | list)}
    nested = [(k, v) for k, v in groups if isinstance(v, dict | list)]
    label = "Structured data" if source is SourceType.META else f"{payload.method} {payload.url}"

    out: list[Section] = []
    budget = MAX_PAYLOAD_LINES
    for key, value in ([("", scalars)] if scalars else []) + nested:
        lines = list(_flatten(value, str(key)))[:budget]
        budget -= len(lines)
        if lines:
            heading = f"{label} › {key}" if key else label
            out.append(
                Section(
                    source=source, heading=heading, text=_redact_payload(lines), origin=payload.url
                )
            )
        if budget <= 0:
            break
    return out
