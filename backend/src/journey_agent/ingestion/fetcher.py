"""Fetch a page's HTML plus any JSON payloads it loads.

Two strategies:

* **static** — a single httpx GET. Fast, no extra install, but blind to
  content that JavaScript renders or fetches after load.
* **browser** — headless Chromium via Playwright. Renders the DOM and records
  every JSON response the page receives (the "response payload" knowledge).

`auto` prefers the browser when Playwright is installed and falls back to static.
When both fail or produce an empty JS shell, the API asks the user to upload an
HTML or (better) HAR file instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

import httpx

from journey_agent.ingestion.models import CapturedPayload, RawPage
from journey_agent.ingestion.url_safety import assert_public_url

log = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36 NubazoJourneyAgent/0.1"
)


class FetchError(RuntimeError):
    """Raised when a page cannot be fetched; the UI responds by offering upload."""


def browser_available() -> bool:
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


async def fetch_page(
    url: str,
    *,
    mode: Literal["auto", "static", "browser"],
    timeout_s: float,
    max_bytes: int,
    max_payload_bytes: int,
    allow_private: bool,
) -> RawPage:
    await asyncio.to_thread(assert_public_url, url, allow_private=allow_private)

    if mode == "browser" or (mode == "auto" and browser_available()):
        try:
            return await _fetch_browser(
                url,
                timeout_s=timeout_s,
                max_payload_bytes=max_payload_bytes,
                allow_private=allow_private,
            )
        except Exception as exc:  # Playwright raises many types; fall back on any
            if mode == "browser":
                raise FetchError(f"Browser fetch failed: {exc}") from exc
            log.warning("event=browser_fetch_failed url=%s error=%s", url, exc)

    return await _fetch_static(
        url, timeout_s=timeout_s, max_bytes=max_bytes, allow_private=allow_private
    )


async def _fetch_static(
    url: str, *, timeout_s: float, max_bytes: int, allow_private: bool
) -> RawPage:
    async def check_redirect(request: httpx.Request) -> None:
        # Re-validate every hop so a public URL cannot redirect into the LAN.
        await asyncio.to_thread(assert_public_url, str(request.url), allow_private=allow_private)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout_s,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        event_hooks={"request": [check_redirect]},
    ) as client:
        try:
            resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise FetchError(f"Could not reach {url}: {exc}") from exc

    if resp.status_code >= 400:
        raise FetchError(f"{url} returned HTTP {resp.status_code} (possibly bot protection).")
    if len(resp.content) > max_bytes:
        raise FetchError(f"Page is larger than {max_bytes} bytes.")
    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "xml" not in ctype:
        raise FetchError(f"Expected HTML but got '{ctype}'.")
    return RawPage(url=str(resp.url), html=resp.text, fetched_with="static")


async def _fetch_browser(
    url: str, *, timeout_s: float, max_payload_bytes: int, allow_private: bool
) -> RawPage:
    from playwright.async_api import Response, async_playwright

    payloads: list[CapturedPayload] = []
    pending: set[asyncio.Task[None]] = set()

    async def capture(response: Response) -> None:
        ctype = response.headers.get("content-type", "")
        if "json" not in ctype or response.request.resource_type not in {"xhr", "fetch"}:
            return
        try:
            body = await response.body()
            if len(body) > max_payload_bytes:
                return
            payloads.append(
                CapturedPayload(
                    url=response.url,
                    method=response.request.method,
                    status=response.status,
                    body=json.loads(body),
                )
            )
        except Exception as exc:  # body may be gone after navigation
            log.debug("event=payload_skip url=%s error=%s", response.url, exc)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(user_agent=USER_AGENT)

            if not allow_private:

                async def guard(route):  # type: ignore[no-untyped-def]
                    try:
                        await asyncio.to_thread(assert_public_url, route.request.url)
                    except ValueError:
                        await route.abort()
                        return
                    await route.continue_()

                await page.route("**/*", guard)

            page.on("response", lambda r: pending.add(asyncio.ensure_future(capture(r))))
            await page.goto(url, wait_until="networkidle", timeout=timeout_s * 1000)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            html = await page.content()
            final_url = page.url
        finally:
            await browser.close()

    return RawPage(url=final_url, html=html, payloads=payloads, fetched_with="browser")
