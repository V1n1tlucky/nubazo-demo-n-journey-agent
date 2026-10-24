"""SQLite persistence for pages and agent runs (the "write" node's sink).

SQLite (WAL mode) is the right default for a single-machine local agent:
zero ops, ACID, and trivially inspectable (`sqlite3 data/journey_agent.sqlite3`).
The repository interface keeps a later move to Postgres a contained change.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    page_id       TEXT PRIMARY KEY,
    url           TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    fetched_with  TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL,
    char_count    INTEGER NOT NULL,
    payload_count INTEGER NOT NULL,
    topics        TEXT NOT NULL DEFAULT '[]',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    theme_color   TEXT,  -- primary brand colour; NULL = not computed yet, '' = none found
    theme_palette TEXT,  -- comma-separated #rrggbb, primary first (ingestion/theme.py)
    company_name  TEXT,  -- detected brand name (welcome.py)
    welcome_message TEXT  -- LLM-written greeting; NULL until generated
);
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL,
    page_id           TEXT NOT NULL REFERENCES pages(page_id),
    created_at        TEXT NOT NULL,
    route             TEXT NOT NULL,
    user_message      TEXT NOT NULL,
    ai_response       TEXT NOT NULL,
    model             TEXT,
    system_prompt     TEXT,
    prompt_messages   TEXT NOT NULL DEFAULT '[]',
    retrieved         TEXT NOT NULL DEFAULT '[]',
    guardrails        TEXT NOT NULL DEFAULT '{}',
    node_latency_ms   TEXT NOT NULL DEFAULT '{}',
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    tokens_estimated  INTEGER NOT NULL DEFAULT 0,
    total_latency_ms  REAL NOT NULL DEFAULT 0,
    grounding_score   REAL,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_session ON runs(session_id, created_at);
CREATE INDEX IF NOT EXISTS ix_runs_page ON runs(page_id, created_at);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class PageRecord(BaseModel):
    page_id: str
    url: str
    title: str
    description: str = ""
    fetched_with: str
    chunk_count: int
    char_count: int
    payload_count: int
    topics: list[str] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    theme_color: str | None = None  # page's primary brand colour (#rrggbb); ingestion/theme.py
    theme_palette: str | None = None  # "#a,#b,#c", primary first; None = not computed yet
    company_name: str | None = None  # LLM-identified at ingestion; see company.py
    welcome_message: str | None = None  # LLM-written greeting, cached per page


class RunRecord(BaseModel):
    run_id: str
    session_id: str
    page_id: str
    created_at: str = Field(default_factory=utcnow)
    route: str
    user_message: str  # always the PII-redacted text
    ai_response: str
    model: str | None = None
    system_prompt: str | None = None
    prompt_messages: list[dict[str, str]] = Field(default_factory=list)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    guardrails: dict[str, Any] = Field(default_factory=dict)
    node_latency_ms: dict[str, float] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_estimated: bool = False
    total_latency_ms: float = 0.0
    grounding_score: float | None = None
    error: str | None = None


_JSON_FIELDS = {"topics", "prompt_messages", "retrieved", "guardrails", "node_latency_ms"}


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in _JSON_FIELDS & data.keys():
        data[key] = json.loads(data[key])
    return data


class Repository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_sync(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            # Additive migrations for databases created before these columns existed.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(pages)")}
            for col in ("theme_color", "theme_palette", "company_name", "welcome_message"):
                if col not in cols:
                    # Names come from the literal tuple above, never from input.
                    conn.execute(f"ALTER TABLE pages ADD COLUMN {col} TEXT")

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    # --- pages ----------------------------------------------------------------
    def _upsert_page_sync(self, page: PageRecord) -> None:
        now = utcnow()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """INSERT INTO pages (page_id, url, title, description, fetched_with,
                       chunk_count, char_count, payload_count, topics, created_at, updated_at,
                       company_name)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(page_id) DO UPDATE SET
                       title=excluded.title, description=excluded.description,
                       fetched_with=excluded.fetched_with, chunk_count=excluded.chunk_count,
                       char_count=excluded.char_count, payload_count=excluded.payload_count,
                       topics=excluded.topics, updated_at=excluded.updated_at,
                       company_name=excluded.company_name,
                       theme_color=NULL, theme_palette=NULL, welcome_message=NULL
                       -- re-ingested: the page may have been restyled or reworded
                   """,
                (
                    page.page_id,
                    page.url,
                    page.title,
                    page.description,
                    page.fetched_with,
                    page.chunk_count,
                    page.char_count,
                    page.payload_count,
                    json.dumps(page.topics),
                    now,
                    now,
                    page.company_name,
                ),
            )

    async def upsert_page(self, page: PageRecord) -> None:
        await asyncio.to_thread(self._upsert_page_sync, page)

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            return [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]

    async def get_page(self, page_id: str) -> PageRecord | None:
        rows = await asyncio.to_thread(
            self._query, "SELECT * FROM pages WHERE page_id=?", (page_id,)
        )
        return PageRecord(**rows[0]) if rows else None

    async def set_theme(self, page_id: str, palette: list[str]) -> None:
        await asyncio.to_thread(
            self._update_sync,
            page_id,
            theme_color=palette[0] if palette else "",
            theme_palette=",".join(palette),
        )

    def _update_sync(self, page_id: str, **cols: str | None) -> None:
        # Column names come from the keyword arguments below, never from input.
        sets = ", ".join(f"{k}=?" for k in cols)
        with closing(self._connect()) as conn, conn:
            conn.execute(f"UPDATE pages SET {sets} WHERE page_id=?", (*cols.values(), page_id))  # noqa: S608

    async def set_company(self, page_id: str, company: str) -> None:
        """A new company name also invalidates the greeting that named the old one."""
        await asyncio.to_thread(
            self._update_sync, page_id, company_name=company, welcome_message=None
        )

    async def set_welcome(self, page_id: str, message: str) -> None:
        await asyncio.to_thread(self._update_sync, page_id, welcome_message=message)

    async def find_page_by_url(self, url: str) -> PageRecord | None:
        rows = await asyncio.to_thread(self._query, "SELECT * FROM pages WHERE url=?", (url,))
        return PageRecord(**rows[0]) if rows else None

    async def list_pages(self) -> list[PageRecord]:
        rows = await asyncio.to_thread(self._query, "SELECT * FROM pages ORDER BY updated_at DESC")
        return [PageRecord(**r) for r in rows]

    # --- runs -----------------------------------------------------------------
    def _insert_run_sync(self, run: RunRecord) -> None:
        data = run.model_dump()
        for key in _JSON_FIELDS & data.keys():
            data[key] = json.dumps(data[key])
        data["tokens_estimated"] = int(run.tokens_estimated)
        cols = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        with closing(self._connect()) as conn, conn:
            # Column names come from the RunRecord model, never from user input.
            sql = f"INSERT INTO runs ({cols}) VALUES ({marks})"  # noqa: S608
            conn.execute(sql, tuple(data.values()))

    async def insert_run(self, run: RunRecord) -> None:
        await asyncio.to_thread(self._insert_run_sync, run)

    async def get_run(self, run_id: str) -> RunRecord | None:
        rows = await asyncio.to_thread(self._query, "SELECT * FROM runs WHERE run_id=?", (run_id,))
        return RunRecord(**rows[0]) if rows else None

    async def list_runs(self, *, page_id: str | None = None, limit: int = 50) -> list[RunRecord]:
        params: tuple[Any, ...]
        if page_id:
            sql, params = (
                "SELECT * FROM runs WHERE page_id=? ORDER BY created_at DESC LIMIT ?",
                (page_id, limit),
            )
        else:
            sql, params = "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        rows = await asyncio.to_thread(self._query, sql, params)
        return [RunRecord(**r) for r in rows]

    async def session_history(
        self, session_id: str, page_id: str, turns: int
    ) -> list[tuple[str, str]]:
        """Last `turns` answered exchanges, oldest first, for conversational memory."""
        rows = await asyncio.to_thread(
            self._query,
            """SELECT user_message, ai_response FROM runs
               WHERE session_id=? AND page_id=? AND route='answered'
               ORDER BY created_at DESC LIMIT ?""",
            (session_id, page_id, turns),
        )
        return [(r["user_message"], r["ai_response"]) for r in reversed(rows)]
