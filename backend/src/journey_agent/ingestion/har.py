"""Parse a browser HAR export into a RawPage.

A HAR ("Save all as HAR" in DevTools → Network) is the best fallback when a
site blocks bots: unlike "Save page as HTML" it also contains every JSON
response the page received.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from journey_agent.ingestion.models import CapturedPayload, RawPage


class HarError(ValueError):
    pass


def _decode(content: dict[str, Any]) -> str | None:
    text = content.get("text")
    if text is None:
        return None
    if content.get("encoding") == "base64":
        return base64.b64decode(text).decode("utf-8", errors="replace")
    return str(text)


def parse_har(raw: bytes, *, max_payload_bytes: int) -> RawPage:
    try:
        entries = json.loads(raw)["log"]["entries"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HarError("Not a valid HAR file.") from exc

    page_url: str | None = None
    html: str | None = None
    payloads: list[CapturedPayload] = []

    for entry in entries:
        req, resp = entry.get("request", {}), entry.get("response", {})
        content = resp.get("content", {})
        mime = content.get("mimeType", "")
        body = _decode(content)
        if body is None:
            continue
        if html is None and "html" in mime and resp.get("status", 0) < 400:
            page_url, html = req.get("url"), body
        elif "json" in mime and len(body) <= max_payload_bytes:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                continue
            payloads.append(
                CapturedPayload(
                    url=req.get("url", ""),
                    method=req.get("method", "GET"),
                    status=resp.get("status", 200),
                    body=parsed,
                )
            )

    if html is None or page_url is None:
        raise HarError("HAR contains no HTML document response.")
    return RawPage(url=page_url, html=html, payloads=payloads, fetched_with="upload-har")
