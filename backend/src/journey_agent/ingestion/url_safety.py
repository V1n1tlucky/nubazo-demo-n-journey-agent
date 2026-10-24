"""SSRF protection for server-side URL fetching.

The ingest endpoint fetches arbitrary user-supplied URLs from the server, so
without this check anyone with UI access could make the server read
`http://169.254.169.254/` (cloud metadata) or internal admin panels.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit


class UnsafeURLError(ValueError):
    pass


def normalize_url(url: str) -> str:
    """Canonical form used as the page identity (no fragment, lowercase host)."""
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    parts = urlsplit(raw)
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def assert_public_url(url: str, *, allow_private: bool = False) -> None:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        raise UnsafeURLError(f"Only http(s) URLs are allowed, got '{parts.scheme}'.")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("URL has no host.")
    if allow_private:
        return
    try:
        infos = socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Cannot resolve host '{host}'.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise UnsafeURLError(f"Host '{host}' resolves to non-public address {ip}.")
