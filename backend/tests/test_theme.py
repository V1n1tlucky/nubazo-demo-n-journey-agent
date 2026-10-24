from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from journey_agent.container import Container
from journey_agent.ingestion import service
from journey_agent.ingestion.theme import parse_color, pick_palette, pick_theme, theme_from_html
from journey_agent.main import create_app
from journey_agent.persistence.repository import PageRecord, Repository


@pytest.mark.parametrize(
    ("value", "rgb"),
    [
        ("rgb(30, 94, 255)", (30, 94, 255)),
        ("rgba(30, 94, 255, 0.9)", (30, 94, 255)),
        ("rgba(30, 94, 255, 0)", None),  # transparent backgrounds are not a colour
        ("#1e5eff", (30, 94, 255)),
        ("#15f", (17, 85, 255)),
        ("inherit", None),
    ],
)
def test_parse_color(value: str, rgb: tuple[int, int, int] | None) -> None:
    assert parse_color(value) == rgb


def test_pick_theme_prefers_heaviest_saturated_hue() -> None:
    samples = [
        ("rgb(255, 255, 255)", 100),  # white page background: ignored
        ("rgb(40, 40, 40)", 80),  # dark grey text: ignored
        ("rgb(0, 0, 238)", 50),  # browser default link colour: CSS didn't load, ignored
        ("rgb(36, 57, 112)", 12),  # navy CTA
        ("rgb(38, 60, 115)", 4),  # same hue, slightly different shade
        ("rgb(224, 67, 91)", 9),  # a red badge
    ]
    assert pick_theme(samples) == "#243970"


def test_pick_theme_none_when_only_greys() -> None:
    assert pick_theme([("#ffffff", 5), ("#333333", 5), ("rgba(0,0,0,0)", 9)]) is None


def test_theme_from_html_reads_meta_and_inline_css() -> None:
    assert theme_from_html('<meta name="theme-color" content="#0fb48c">') == ["#0fb48c"]
    html = '<style>.btn{background:#d98a00}</style><a style="color:#d98a00">x</a>'
    assert theme_from_html(html) == ["#d98a00"]


def test_palette_keeps_distinct_brand_colours_heaviest_first() -> None:
    samples = [
        ("rgb(255, 98, 0)", 30),  # orange CTA
        ("rgb(250, 105, 10)", 6),  # same orange, another shade: merges
        ("rgb(33, 98, 161)", 12),  # link blue
        ("rgb(174, 40, 46)", 8),  # maroon: ~26° from orange, a different palette colour
        ("rgb(16, 200, 120)", 1),  # stray badge: under 8% of the primary
        ("#ffffff", 90),
    ]
    assert pick_palette(samples) == ["#ff6200", "#2162a1", "#ae282e"]


def test_palette_of_a_single_hue_brand_is_one_colour() -> None:
    assert pick_palette([("#243970", 20), ("#263b72", 10), ("#f4f4f4", 50)]) == ["#243970"]


def test_repository_migrates_old_pages_table(tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite3"
    with sqlite3.connect(db) as conn:  # a pages table from before theme_color existed
        conn.execute(
            "CREATE TABLE pages (page_id TEXT PRIMARY KEY, url TEXT NOT NULL UNIQUE, "
            "title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', fetched_with TEXT NOT NULL, "
            "chunk_count INTEGER NOT NULL, char_count INTEGER NOT NULL, payload_count INTEGER NOT NULL, "
            "topics TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
    Repository(db)._init_sync()
    with sqlite3.connect(db) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(pages)")}
    assert {"theme_color", "theme_palette", "company_name", "welcome_message"} <= cols


async def test_theme_endpoint_computes_once_and_reingest_resets(
    container: Container, page: PageRecord, page_html: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "browser_available", lambda: False)  # no Chromium in unit tests
    path = container.ingestion.snapshot_path(page.page_id)
    path.write_text(path.read_text() + "<style>.cta{background:#0fb48c}</style>")

    with TestClient(create_app(container.settings, container)) as client:
        body = client.get(f"/api/pages/{page.page_id}/theme").json()
        assert body["theme_color"] == "#0fb48c" and body["palette"] == ["#0fb48c"]
        path.write_text("<html></html>")  # stored value is reused, not recomputed
        assert client.get(f"/api/pages/{page.page_id}/theme").json()["palette"] == ["#0fb48c"]

    await container.ingestion.ingest_upload("page.html", page_html.encode(), None)
    stored = await container.repo.get_page(page.page_id)
    assert stored is not None and stored.theme_color is None and stored.theme_palette is None
