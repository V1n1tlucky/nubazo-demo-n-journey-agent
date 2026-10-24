"""The brand colour of an ingested page, so the assistant can wear the page's theme.

The stored snapshot is rendered in headless Chromium (its stylesheets load; its
scripts are already stripped) and the colours a visitor actually sees are
sampled: button/CTA backgrounds weighted by area, header/nav backgrounds, link
and heading colours, and `<meta name="theme-color">`. Greys, near-white and
near-black are ignored and the heaviest hue wins. Without Chromium, a cheaper
pass reads the meta tag and inline CSS instead.
"""

from __future__ import annotations

import asyncio
import colorsys
import contextlib
import logging
import re
from collections import defaultdict

from journey_agent.ingestion.url_safety import assert_public_url

log = logging.getLogger(__name__)

RGB = tuple[int, int, int]

# Runs inside the rendered snapshot. Returns [css colour, weight] pairs.
_SAMPLE_JS = r"""
() => {
  const out = [];
  const vh = innerHeight;
  const seen = (el) => {
    const r = el.getBoundingClientRect();
    return r.width > 4 && r.height > 4 && r.bottom > 0 && r.top < vh * 3;
  };
  const resolve = (c) => {
    const i = document.createElement("i");
    i.style.color = c;
    document.body.appendChild(i);
    const v = getComputedStyle(i).color;
    i.remove();
    return v;
  };
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta && meta.content) out.push([resolve(meta.content), 30]);
  const buttons = 'button, [role=button], input[type=submit], a[class*=btn], a[class*=button], ' +
    '[class*=btn-primary], [class*=cta]';
  for (const el of document.querySelectorAll(buttons)) {
    if (!seen(el)) continue;
    const r = el.getBoundingClientRect();
    out.push([getComputedStyle(el).backgroundColor, 2 + Math.min(r.width * r.height, 40000) / 2000]);
  }
  for (const el of document.querySelectorAll("header, nav, [class*=header], [class*=navbar]")) {
    if (seen(el)) out.push([getComputedStyle(el).backgroundColor, 6]);
  }
  for (const el of [...document.querySelectorAll("a")].slice(0, 300)) {
    if (seen(el)) out.push([getComputedStyle(el).color, 0.25]); // many links mustn't outvote a CTA
  }
  for (const el of [...document.querySelectorAll("h1, h2")].slice(0, 30)) {
    if (seen(el)) out.push([getComputedStyle(el).color, 2]);
  }
  return out;
}
"""

_RGB_FN = re.compile(
    r"rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)"
)
_HEX = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_META = re.compile(
    r"""<meta[^>]+name=["']theme-color["'][^>]+content=["']([^"']+)["']""", re.IGNORECASE
)
_INLINE_CSS = re.compile(r"<style[^>]*>(.*?)</style>|style=\"([^\"]*)\"", re.IGNORECASE | re.DOTALL)


def parse_color(value: str) -> RGB | None:
    """`rgb()/rgba()` (as browsers report computed styles) or `#rgb/#rrggbb`; None if transparent."""
    if m := _RGB_FN.search(value):
        alpha = m.group(4)
        if alpha is not None:
            a = float(alpha.rstrip("%")) / (100 if alpha.endswith("%") else 1)
            if a < 0.5:
                return None
        r, g, b = (round(float(m.group(i))) for i in (1, 2, 3))
        return r, g, b
    if m := _HEX.search(value):
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return None


# Chromium's built-in link colours: seeing them means the site's CSS didn't apply.
_UA_DEFAULTS = {(0, 0, 238), (85, 26, 139)}


def _is_brand_like(rgb: RGB) -> bool:
    if rgb in _UA_DEFAULTS:
        return False
    _, lightness, s = colorsys.rgb_to_hls(*(c / 255 for c in rgb))
    return s >= 0.25 and 0.15 <= lightness <= 0.85


def pick_theme(samples: list[tuple[str, float]]) -> str | None:
    """The heaviest saturated hue among weighted colour samples, as `#rrggbb`."""
    buckets: dict[int, float] = defaultdict(float)
    exact: dict[RGB, float] = defaultdict(float)
    for value, weight in samples:
        rgb = parse_color(value)
        if rgb is None or not _is_brand_like(rgb):
            continue
        hue = colorsys.rgb_to_hls(*(c / 255 for c in rgb))[0]
        buckets[int(hue * 24) % 24] += weight  # 15° hue bins
        exact[rgb] += weight
    if not buckets:
        return None
    best = max(buckets, key=lambda b: buckets[b])
    in_bucket = {
        rgb: w
        for rgb, w in exact.items()
        if int(colorsys.rgb_to_hls(*(c / 255 for c in rgb))[0] * 24) % 24 == best
    }
    r, g, b = max(in_bucket, key=lambda c: in_bucket[c])
    return f"#{r:02x}{g:02x}{b:02x}"


def theme_from_html(html: str) -> str | None:
    """Fallback without a browser: meta theme-color, then colours in inline CSS."""
    samples: list[tuple[str, float]] = []
    if m := _META.search(html):
        samples.append((m.group(1), 30))
    for block in _INLINE_CSS.finditer(html):
        css = block.group(1) or block.group(2) or ""
        samples += [(c.group(0), 1) for c in _HEX.finditer(css)]
        samples += [(c.group(0), 1) for c in _RGB_FN.finditer(css)]
    return pick_theme(samples)


async def theme_from_snapshot(html: str, *, timeout_s: float, allow_private: bool) -> str | None:
    """Render the sanitised snapshot and sample its on-screen colours."""
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1280, "height": 800})
            if not allow_private:

                async def guard(route):  # type: ignore[no-untyped-def]
                    try:
                        await asyncio.to_thread(assert_public_url, route.request.url)
                    except ValueError:
                        await route.abort()
                        return
                    await route.continue_()

                await page.route("**/*", guard)
            await page.set_content(html, wait_until="domcontentloaded", timeout=timeout_s * 1000)
            # Stylesheets matter; a slow font or image shouldn't block the answer.
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("networkidle", timeout=5000)
            samples = await page.evaluate(_SAMPLE_JS)
        finally:
            await browser.close()
    return pick_theme([(str(c), float(w)) for c, w in samples])
