"""Structured (key=value) logging setup."""

from __future__ import annotations

import logging


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
    )
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpcore", "openai", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
