"""Produce a safe, frameable static snapshot of the page for the left pane.

Most production sites (nubazo.com included) send `X-Frame-Options: DENY` /
`frame-ancestors 'none'`, so a live iframe is impossible. We instead serve the
captured DOM from our own origin with all scripts removed, and the iframe is
sandboxed without `allow-scripts`.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

# Sites commonly hide content until JS adds a class (e.g. `.reveal.in`).
# Without scripts that content would stay invisible, so force it visible.
_UNHIDE_CSS = """
.reveal,[data-reveal],[data-aos],.fade-in,.fade-up,.animate,.wow,.aos-init{
  opacity:1!important;transform:none!important;visibility:visible!important}
"""


def sanitize_snapshot(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "noscript", "iframe", "object", "embed", "base"]):
        tag.decompose()
    for meta in soup.find_all("meta"):
        if str(meta.get("http-equiv", "")).lower() in {"content-security-policy", "refresh"}:
            meta.decompose()
    for el in soup.find_all(True):
        for attr in list(el.attrs):
            value = str(el.attrs[attr]).strip().lower()
            if attr.lower().startswith("on") or value.startswith("javascript:"):
                del el.attrs[attr]
        if el.name == "form":
            el.attrs.pop("action", None)  # sandbox blocks submission anyway

    head = soup.head or soup.new_tag("head")
    if soup.head is None and soup.html is not None:
        soup.html.insert(0, head)
    base = soup.new_tag("base", href=base_url, target="_blank")
    head.insert(0, base)
    style = soup.new_tag("style")
    style.string = _UNHIDE_CSS
    head.append(style)
    return str(soup)
