from __future__ import annotations

import base64
import json

import pytest

from journey_agent.ingestion.chunker import chunk_document
from journey_agent.ingestion.extractor import extract
from journey_agent.ingestion.har import HarError, parse_har
from journey_agent.ingestion.models import CapturedPayload, RawPage, SourceType
from journey_agent.ingestion.snapshot import sanitize_snapshot
from journey_agent.ingestion.url_safety import UnsafeURLError, assert_public_url, normalize_url


class TestUrlSafety:
    def test_normalize_adds_scheme_and_strips_fragment(self) -> None:
        assert normalize_url("Nubazo.com#top") == "https://nubazo.com/"

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/admin",
            "http://localhost:8000",
            "http://10.0.0.5/",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/",
        ],
    )
    def test_blocks_private_targets(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            assert_public_url(url)

    def test_blocks_non_http_schemes(self) -> None:
        with pytest.raises(UnsafeURLError):
            assert_public_url("file:///etc/passwd", allow_private=True)

    def test_private_allowed_when_opted_in(self) -> None:
        assert_public_url("http://localhost:8000", allow_private=True)


class TestExtractor:
    def test_sections_carry_heading_breadcrumbs(self, page_html: str) -> None:
        doc = extract(
            RawPage(url="https://acme.example/loans", html=page_html, fetched_with="static")
        )
        headings = {s.heading for s in doc.sections if s.source is SourceType.CONTENT}
        assert "Personal loans in minutes › Eligibility › Documents" in headings
        assert doc.title.startswith("Acme Loans")
        assert doc.topics == ["Eligibility", "Interest rates"]

    def test_scripts_never_become_content(self, page_html: str) -> None:
        doc = extract(
            RawPage(url="https://acme.example/loans", html=page_html, fetched_with="static")
        )
        assert all("window.track" not in s.text for s in doc.sections)

    def test_embedded_json_is_flattened(self, page_html: str) -> None:
        doc = extract(
            RawPage(url="https://acme.example/loans", html=page_html, fetched_with="static")
        )
        payload = "\n".join(s.text for s in doc.sections if s.source is SourceType.PAYLOAD)
        assert "props.pageProps.maxTenureMonths: 60" in payload
        meta = "\n".join(s.text for s in doc.sections if s.source is SourceType.META)
        assert "interestRate: 10.99%" in meta

    def test_repeated_text_is_deduplicated(self) -> None:
        html = "<body><h1>T</h1><p>Same line</p><p>Same line</p></body>"
        doc = extract(RawPage(url="https://x.example/", html=html, fetched_with="static"))
        assert doc.sections[-1].text == "Same line"


class TestChunker:
    def test_small_sibling_sections_are_packed(self, page_html: str) -> None:
        doc = extract(
            RawPage(url="https://acme.example/loans", html=page_html, fetched_with="static")
        )
        chunks = chunk_document(doc, size=900, overlap=100)
        eligibility = [c for c in chunks if c.heading.endswith("Eligibility")]
        assert len(eligibility) == 1
        assert "Documents:" in eligibility[0].text and "PAN card" in eligibility[0].text

    def test_long_text_respects_size(self) -> None:
        html = "<body><h1>T</h1><p>" + "This is a sentence. " * 300 + "</p></body>"
        doc = extract(RawPage(url="https://x.example/", html=html, fetched_with="static"))
        chunks = chunk_document(doc, size=500, overlap=50)
        assert len(chunks) > 5
        assert max(len(c.text) for c in chunks) <= 500


class TestHar:
    def _har(self, entries: list[dict[str, object]]) -> bytes:
        return json.dumps({"log": {"entries": entries}}).encode()

    def test_extracts_document_and_json_payloads(self) -> None:
        body = base64.b64encode(b'{"plans": [{"name": "Gold", "fee": 499}]}').decode()
        raw = parse_har(
            self._har(
                [
                    {
                        "request": {"url": "https://shop.example/", "method": "GET"},
                        "response": {
                            "status": 200,
                            "content": {"mimeType": "text/html", "text": "<h1>Shop</h1>"},
                        },
                    },
                    {
                        "request": {"url": "https://shop.example/api/plans", "method": "GET"},
                        "response": {
                            "status": 200,
                            "content": {
                                "mimeType": "application/json",
                                "text": body,
                                "encoding": "base64",
                            },
                        },
                    },
                ]
            ),
            max_payload_bytes=10_000,
        )
        assert raw.url == "https://shop.example/"
        assert raw.payloads[0].body == {"plans": [{"name": "Gold", "fee": 499}]}

    def test_rejects_har_without_html(self) -> None:
        with pytest.raises(HarError):
            parse_har(self._har([]), max_payload_bytes=10)

    def test_rejects_garbage(self) -> None:
        with pytest.raises(HarError):
            parse_har(b"not json", max_payload_bytes=10)


class TestSnapshot:
    def test_strips_active_content_and_sets_base(self, page_html: str) -> None:
        out = sanitize_snapshot(page_html, "https://acme.example/loans")
        assert "<script" not in out
        assert "onload" not in out
        assert "javascript:" not in out
        assert "Content-Security-Policy" not in out
        assert '<base href="https://acme.example/loans" target="_blank"/>' in out
        assert ".reveal" in out  # JS-revealed content is forced visible


def test_payload_pii_is_redacted_before_indexing() -> None:
    raw = RawPage(
        url="https://bank.example/",
        html="<body><h1>Bank</h1><p>Welcome to the bank.</p></body>",
        payloads=[
            CapturedPayload(
                url="https://bank.example/api/me",
                body={"user": {"email": "jane@example.com", "pan": "ABCDE1234F", "plan": "Gold"}},
            )
        ],
        fetched_with="upload-har",
    )
    payload = next(s for s in extract(raw).sections if s.source is SourceType.PAYLOAD)
    assert "jane@example.com" not in payload.text and "ABCDE1234F" not in payload.text
    assert "user.plan: Gold" in payload.text
