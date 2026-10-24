"""The brand colour of an ingested page, so the assistant can wear the page's theme.

The stored snapshot is rendered in headless Chromium (its stylesheets load; its
scripts are already stripped) and two kinds of evidence are combined:

* CSS colours of what matters most: button/CTA backgrounds weighted by area,
  header/nav backgrounds, link and heading colours, `<meta name="theme-color">`;
* the rendered pixels of the logo(s), where brand palettes live, often in images
  or SVG that CSS sampling can't see. Nothing else is pixel-sampled: hero photos
  would add skin tones, and text in a header band adds anti-aliasing noise.

Greys, near-white and near-black are ignored; the heaviest distinct hues form
the palette. Without Chromium, a cheaper pass reads the meta tag and inline CSS.
"""

from __future__ import annotations

import asyncio
import colorsys
import contextlib
import io
import logging
import re
from collections import defaultdict

import numpy as np
from PIL import Image

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

# Bounding boxes of logo-like elements near the top: anything marked logo/brand, or an
# image/SVG inside the header, nav or home link.
_LOGO_RECTS_JS = r"""
() => {
  const text = (el) => [el.getAttribute("class"), el.id, el.getAttribute("alt"),
                        el.getAttribute("src"), el.getAttribute("aria-label")].join(" ");
  const branded = (el) => /logo|brand/i.test(text(el));
  const inHeader = (el) => !!el.closest('header, nav, a[href="/"], a[href="./"], [class*=brand i], [class*=logo i]');
  const out = [];
  for (const el of document.querySelectorAll("img, svg, [class*=logo i], [id*=logo i]")) {
    const r = el.getBoundingClientRect();
    if (r.width < 16 || r.height < 10 || r.width > 600 || r.top > 300 || r.bottom < 0) continue;
    const media = el.tagName === "IMG" || el.tagName.toLowerCase() === "svg";
    if (!(branded(el) || (media && inHeader(el)))) continue;
    // Clamp to the viewport: off-canvas menus put logos partly off-screen.
    const x = Math.max(0, r.left), y = Math.max(0, r.top);
    const w = Math.min(innerWidth, r.right) - x, h = Math.min(innerHeight, r.bottom) - y;
    if (w < 8 || h < 8) continue;
    out.push({ x, y, width: w, height: h });
    if (out.length >= 4) break;
  }
  return out;
}
"""
LOGO_WEIGHT = 40.0  # total for all logos; a large CTA weighs about 22


def pixel_samples(png: bytes, weight: float) -> list[tuple[str, float]]:
    """Brand-like colours in a screenshot, sharing `weight` by how many pixels each covers.

    Only clearly saturated pixels count, so a small logo on lots of white still speaks
    clearly; pale anti-aliasing blends and dull textured backgrounds don't.
    """
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if img.width > 200:
        img = img.resize((200, max(1, round(200 * img.height / img.width))), Image.Resampling.BOX)
    colors, freq = np.unique(np.asarray(img).reshape(-1, 3), axis=0, return_counts=True)
    counts: dict[RGB, int] = {}
    for (r, g, b), n in zip(colors.tolist(), freq.tolist(), strict=True):
        _, lightness, sat = _hls((r, g, b))
        if _is_brand_like((r, g, b)) and sat >= 0.45 and lightness <= 0.75:
            counts[(r, g, b)] = n
    total = sum(counts.values())
    if not total:
        return []
    return [(f"rgb({r}, {g}, {b})", weight * n / total) for (r, g, b), n in counts.items()]


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


def _hls(rgb: RGB) -> tuple[float, float, float]:
    return colorsys.rgb_to_hls(*(c / 255 for c in rgb))


def _cluster(rgb: RGB) -> tuple[int, int]:
    h, lightness, _ = _hls(rgb)
    return int(h * 24) % 24, min(int(lightness / 0.15), 6)  # 15° hue x 0.15 lightness cells


def _distinct(a: RGB, b: RGB) -> bool:
    """Different palette colours: 20°+ apart in hue (maroon vs orange) or clearly lighter/darker."""
    ha, la, _ = _hls(a)
    hb, lb, _ = _hls(b)
    hue_gap = min(abs(ha - hb), 1 - abs(ha - hb)) * 360
    return hue_gap >= 20 or abs(la - lb) >= 0.2


def pick_palette(samples: list[tuple[str, float]], size: int = 3) -> list[str]:
    """Up to `size` distinct brand colours, heaviest first, as `#rrggbb`.

    Near-identical shades merge into one entry; a colour carrying under 8% of the
    primary's weight is a stray badge, not part of the palette.
    """
    cells: dict[tuple[int, int], float] = defaultdict(float)
    exact: dict[RGB, float] = defaultdict(float)
    for value, weight in samples:
        rgb = parse_color(value)
        if rgb is None or not _is_brand_like(rgb):
            continue
        cells[_cluster(rgb)] += weight
        exact[rgb] += weight
    chosen: list[tuple[RGB, float]] = []
    for cell in sorted(cells, key=lambda c: -cells[c]):
        if len(chosen) == size or (chosen and cells[cell] < 0.08 * chosen[0][1]):
            break
        rep = max((rgb for rgb in exact if _cluster(rgb) == cell), key=lambda c: exact[c])
        if all(_distinct(rep, other) for other, _ in chosen):
            chosen.append((rep, cells[cell]))
    return [f"#{r:02x}{g:02x}{b:02x}" for (r, g, b), _ in chosen]


def pick_theme(samples: list[tuple[str, float]]) -> str | None:
    """The heaviest saturated hue among weighted colour samples, as `#rrggbb`."""
    palette = pick_palette(samples, size=1)
    return palette[0] if palette else None


def theme_from_html(html: str) -> list[str]:
    """Fallback without a browser: meta theme-color, then colours in inline CSS."""
    samples: list[tuple[str, float]] = []
    if m := _META.search(html):
        samples.append((m.group(1), 30))
    for block in _INLINE_CSS.finditer(html):
        css = block.group(1) or block.group(2) or ""
        samples += [(c.group(0), 1) for c in _HEX.finditer(css)]
        samples += [(c.group(0), 1) for c in _RGB_FN.finditer(css)]
    return pick_palette(samples)


async def theme_from_snapshot(html: str, *, timeout_s: float, allow_private: bool) -> list[str]:
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
            samples = [(str(c), float(w)) for c, w in await page.evaluate(_SAMPLE_JS)]
            with contextlib.suppress(Exception):  # logo images must have loaded to be sampled
                await page.wait_for_function(
                    "[...document.images].every((i) => i.complete)", timeout=4000
                )
            logos: list[list[tuple[str, float]]] = []
            for rect in (await page.evaluate(_LOGO_RECTS_JS))[:4]:
                with contextlib.suppress(Exception):  # an element that vanished, etc.
                    shot = await page.screenshot(type="png", clip=rect)
                    if found := pixel_samples(shot, 1.0):
                        logos.append(found)
            for found in logos:  # the logo weight is shared only by logos that had colour
                samples += [(c, w * LOGO_WEIGHT / len(logos)) for c, w in found]
        finally:
            await browser.close()
    return pick_palette(samples)
