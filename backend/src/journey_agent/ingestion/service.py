"""Ingestion orchestration: acquire → extract → chunk → index → snapshot → record."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from journey_agent.company import identify_company
from journey_agent.config import Settings
from journey_agent.ingestion.chunker import chunk_document
from journey_agent.ingestion.extractor import extract
from journey_agent.ingestion.fetcher import browser_available, fetch_page
from journey_agent.ingestion.har import parse_har
from journey_agent.ingestion.models import Chunk, RawPage, SourceType
from journey_agent.ingestion.snapshot import sanitize_snapshot
from journey_agent.ingestion.theme import theme_from_html, theme_from_snapshot
from journey_agent.ingestion.url_safety import normalize_url
from journey_agent.llm.client import ChatModel
from journey_agent.persistence.repository import PageRecord, Repository
from journey_agent.rag.store import IndexRegistry

log = logging.getLogger(__name__)


class NeedsUploadError(RuntimeError):
    """The page could not be read server-side; the user should upload HTML/HAR."""


@dataclass
class IngestionService:
    settings: Settings
    registry: IndexRegistry
    repo: Repository
    llm: ChatModel  # company identification is a required step of ingestion

    def snapshot_path(self, page_id: str) -> Path:
        return self.settings.snapshot_dir / f"{page_id}.html"

    async def ensure_company(self, page: PageRecord) -> PageRecord:
        """Pages ingested before identification was required get it on first use.

        Raises CompanyIdentificationError, like ingestion does.
        """
        if page.company_name:
            return page
        path = self.snapshot_path(page.page_id)
        html = await asyncio.to_thread(path.read_text, "utf-8") if path.is_file() else ""
        index = await self.registry.get(page.page_id)
        chunks = index.chunks if index else []
        company = await identify_company(
            self.llm,
            url=page.url,
            title=page.title,
            description=page.description,
            excerpt=_excerpt(chunks),
            html=html,
            meta_chunks=[c for c in chunks if c.source is SourceType.META],
        )
        await self.repo.set_company(page.page_id, company)
        return page.model_copy(update={"company_name": company, "welcome_message": None})

    async def theme_palette(self, page: PageRecord) -> list[str]:
        """The page's brand palette (primary first), computed from its snapshot once and stored."""
        if page.theme_palette is not None:
            return [c for c in page.theme_palette.split(",") if c]
        path = self.snapshot_path(page.page_id)
        if not path.is_file():
            return []
        html = await asyncio.to_thread(path.read_text, "utf-8")
        palette: list[str] = []
        if browser_available():
            try:
                palette = await theme_from_snapshot(
                    html,
                    timeout_s=self.settings.fetch_timeout_s,
                    allow_private=self.settings.allow_private_urls,
                )
            except Exception as exc:  # Chromium not downloaded, page too slow, …
                log.warning("event=theme_browser_failed page_id=%s error=%s", page.page_id, exc)
        if not palette:
            palette = await asyncio.to_thread(theme_from_html, html)
        await self.repo.set_theme(page.page_id, palette)
        log.info("event=theme_detected page_id=%s palette=%s", page.page_id, palette)
        return palette

    async def ingest_url(self, url: str) -> PageRecord:
        s = self.settings
        raw = await fetch_page(
            normalize_url(url),
            mode=s.fetch_mode,
            timeout_s=s.fetch_timeout_s,
            max_bytes=s.max_page_bytes,
            max_payload_bytes=s.max_payload_bytes,
            allow_private=s.allow_private_urls,
        )
        return await self._ingest(raw, require_content=True)

    async def ingest_upload(self, filename: str, data: bytes, url: str | None) -> PageRecord:
        if filename.lower().endswith(".har"):
            raw = parse_har(data, max_payload_bytes=self.settings.max_payload_bytes)
        else:
            html = data.decode("utf-8", errors="replace")
            page_url = url or _canonical_url(html)
            if not page_url:
                raise ValueError("Please provide the page URL for an HTML upload.")
            raw = RawPage(url=page_url, html=html, fetched_with="upload-html")
        return await self._ingest(raw, require_content=False)

    async def _ingest(self, raw: RawPage, *, require_content: bool) -> PageRecord:
        raw.url = normalize_url(raw.url)  # final URL after redirects is the page identity
        doc = await asyncio.to_thread(extract, raw)
        content_chars = sum(len(x.text) for x in doc.sections if x.source is SourceType.CONTENT)
        if (
            require_content
            and content_chars < self.settings.min_extracted_chars
            and not raw.payloads
        ):
            raise NeedsUploadError(
                f"Only {content_chars} characters of text were readable — the page is probably "
                "rendered by JavaScript or protected against bots. Upload a HAR or HTML export."
            )
        chunks = chunk_document(
            doc, size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )
        if not chunks:
            raise NeedsUploadError("No readable text found on this page.")

        # Before anything is written: no LLM-identified company, no page.
        company = await identify_company(
            self.llm,
            url=doc.url,
            title=doc.title,
            description=doc.description,
            excerpt=_excerpt(chunks),
            html=raw.html,  # the original, with JSON-LD and meta tags the snapshot drops
            meta_chunks=[c for c in chunks if c.source is SourceType.META],
        )

        await self.registry.put(doc.page_id, chunks)
        snapshot = await asyncio.to_thread(sanitize_snapshot, raw.html, raw.url)
        path = self.snapshot_path(doc.page_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, snapshot, "utf-8")

        record = PageRecord(
            page_id=doc.page_id,
            url=doc.url,
            title=doc.title,
            description=doc.description,
            fetched_with=doc.fetched_with,
            chunk_count=len(chunks),
            char_count=doc.char_count,
            payload_count=len(raw.payloads),
            topics=doc.topics,
            company_name=company,
        )
        await self.repo.upsert_page(record)
        log.info(
            "event=page_ingested page_id=%s url=%s chunks=%d payloads=%d via=%s",
            doc.page_id,
            doc.url,
            len(chunks),
            len(raw.payloads),
            doc.fetched_with,
        )
        return await self.repo.get_page(doc.page_id) or record


def _excerpt(chunks: list[Chunk]) -> str:
    return "\n".join(c.text for c in chunks if c.source is SourceType.CONTENT)[:4000]


def _canonical_url(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return str(link["href"])
    og = soup.find("meta", property="og:url")
    if og and og.get("content"):
        return str(og["content"])
    return None
