"""HTTP routes: pages (ingest/snapshot), chat (SSE), runs (audit), meta."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from journey_agent.api.schemas import AppConfig, ChatRequest, Health, IngestRequest, NodeInfo
from journey_agent.company import CompanyIdentificationError
from journey_agent.container import Container
from journey_agent.graph import prompts
from journey_agent.graph.state import AgentState
from journey_agent.guardrails.policy import guardrail_policy
from journey_agent.ingestion.fetcher import FetchError, browser_available
from journey_agent.ingestion.har import HarError
from journey_agent.ingestion.service import NeedsUploadError
from journey_agent.ingestion.url_safety import UnsafeURLError, normalize_url
from journey_agent.persistence.repository import PageRecord, RunRecord

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

NODES = [
    NodeInfo(
        id="pre_guardrail",
        label="Pre-guardrail",
        description="Deterministic checks (length, injection, abuse, PII, small talk), then an LLM input judge.",
    ),
    NodeInfo(
        id="retrieve",
        label="Retrieval",
        description="FAISS search over page content and response payloads + relevance gate.",
    ),
    NodeInfo(
        id="prompt_builder",
        label="Prompt builder",
        description="System prompt + retrieved context + payload inputs + history, "
        "within a token budget.",
    ),
    NodeInfo(
        id="generate",
        label="LLM",
        description="Local open-source model via an OpenAI-compatible server.",
    ),
    NodeInfo(
        id="post_guardrail",
        label="Post-guardrail",
        description="Deterministic checks (leaks, PII, links, numbers, grounding), then an LLM output judge.",
    ),
    NodeInfo(
        id="write",
        label="Write",
        description="Persists prompts, messages, tokens, guardrail verdicts and latencies.",
    ),
]


# The snapshot is untrusted third-party HTML: no scripts, ever.
def snapshot_csp(frame_ancestors: list[str]) -> str:
    return (
        "default-src * data: blob: 'unsafe-inline'; script-src 'none'; object-src 'none'; "
        f"frame-ancestors 'self' {' '.join(frame_ancestors)}; form-action 'none'"
    )


def container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


Dep = Annotated[Container, Depends(container)]
# Page ids become file paths; a strict pattern rules out traversal.
PageId = Annotated[str, Path(pattern=r"^[0-9a-f]{16}$")]


def _needs_upload(detail: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"code": "needs_upload", "detail": detail})


def _llm_unavailable(exc: CompanyIdentificationError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"code": "llm_unavailable", "detail": str(exc)})


# --- meta -----------------------------------------------------------------------
@router.get("/health", response_model=Health)
async def health(c: Dep) -> Health:
    ok = await c.llm.health()
    return Health(
        status="ok" if ok else "degraded",
        llm_reachable=ok,
        model=c.llm.model,
        embedder=c.embedder.name,
    )


@router.get("/config", response_model=AppConfig)
async def config(c: Dep) -> AppConfig:
    return AppConfig(
        nodes=NODES,
        default_persona=prompts.DEFAULT_PERSONA,
        grounding_rules=prompts.GROUNDING_RULES,
        allow_prompt_override=c.settings.allow_prompt_override,
        model=c.llm.model,
        embedder=c.embedder.name,
        browser_fetch=browser_available(),
        guardrails=guardrail_policy(c.settings),
    )


# --- pages ----------------------------------------------------------------------
@router.post("/pages", response_model=PageRecord)
async def ingest_url(body: IngestRequest, c: Dep) -> PageRecord | JSONResponse:
    try:
        return await c.ingestion.ingest_url(body.url)
    except UnsafeURLError as exc:
        return JSONResponse(status_code=400, content={"code": "unsafe_url", "detail": str(exc)})
    except (FetchError, NeedsUploadError) as exc:
        return _needs_upload(str(exc))
    except CompanyIdentificationError as exc:
        return _llm_unavailable(exc)


@router.post("/pages/upload", response_model=PageRecord)
async def ingest_upload(
    c: Dep,
    file: Annotated[UploadFile, File(description=".html/.htm page export or .har archive")],
    url: Annotated[str | None, Form()] = None,
) -> PageRecord | JSONResponse:
    data = await file.read(c.settings.max_page_bytes * 4 + 1)  # HARs are bigger than pages
    if len(data) > c.settings.max_page_bytes * 4:
        return JSONResponse(
            status_code=413, content={"code": "bad_request", "detail": "File too large."}
        )
    try:
        return await c.ingestion.ingest_upload(file.filename or "page.html", data, url)
    except (HarError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"code": "bad_request", "detail": str(exc)})
    except NeedsUploadError as exc:
        return _needs_upload(str(exc))
    except CompanyIdentificationError as exc:
        return _llm_unavailable(exc)


@router.get("/pages", response_model=list[PageRecord])
async def list_pages(c: Dep) -> list[PageRecord]:
    return await c.repo.list_pages()


@router.get("/pages/lookup", response_model=PageRecord)
async def lookup_page(url: str, c: Dep) -> PageRecord:
    page = await c.repo.find_page_by_url(normalize_url(url))
    if page is None:
        raise HTTPException(404, "Page has not been ingested.")
    return page


@router.get("/pages/{page_id}", response_model=PageRecord)
async def get_page(page_id: PageId, c: Dep) -> PageRecord:
    page = await c.repo.get_page(page_id)
    if page is None:
        raise HTTPException(404, "Unknown page.")
    return page


@router.get("/pages/{page_id}/snapshot", response_class=HTMLResponse)
async def page_snapshot(page_id: PageId, c: Dep) -> HTMLResponse:
    path = c.ingestion.snapshot_path(page_id)
    if not path.is_file():
        raise HTTPException(404, "No snapshot for this page.")
    return HTMLResponse(
        path.read_text("utf-8"),
        headers={
            "Content-Security-Policy": snapshot_csp(c.settings.frontend_origins),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/pages/{page_id}/theme")
async def page_theme(page_id: PageId, c: Dep) -> dict[str, Any]:
    """The page's brand palette (primary first), so the assistant can wear the page's colours."""
    page = await c.repo.get_page(page_id)
    if page is None:
        raise HTTPException(404, "Unknown page.")
    palette = await c.ingestion.theme_palette(page)
    return {"page_id": page_id, "theme_color": palette[0] if palette else None, "palette": palette}


@router.get("/pages/{page_id}/welcome", response_model=None)
async def page_welcome(page_id: PageId, c: Dep) -> dict[str, str] | JSONResponse:
    """The assistant's opening message and the company it names (LLM-written, cached)."""
    page = await c.repo.get_page(page_id)
    if page is None:
        raise HTTPException(404, "Unknown page.")
    try:
        return (await c.welcome.welcome(page)).model_dump()
    except CompanyIdentificationError as exc:
        return _llm_unavailable(exc)


@router.get("/pages/{page_id}/chunks")
async def page_chunks(page_id: PageId, c: Dep) -> list[dict[str, Any]]:
    index = await c.registry.get(page_id)
    if index is None:
        raise HTTPException(404, "Unknown page.")
    return [ch.model_dump() for ch in index.chunks]


# --- chat -----------------------------------------------------------------------
@router.post("/chat", response_model=None)
async def chat(body: ChatRequest, c: Dep) -> EventSourceResponse | JSONResponse:
    page = await c.repo.get_page(body.page_id)
    if page is None:
        raise HTTPException(404, "Unknown page — ingest it first.")
    try:
        page = await c.ingestion.ensure_company(page)  # pages ingested before it was required
    except CompanyIdentificationError as exc:
        return _llm_unavailable(exc)

    run_id = uuid.uuid4().hex
    state: AgentState = {
        "run_id": run_id,
        "session_id": body.session_id,
        "page_id": page.page_id,
        "page_url": page.url,
        "page_title": page.title,
        "company_name": page.company_name or page.title,  # ensure_company has set it
        "page_description": page.description,
        "greeting": page.welcome_message or "",
        "topics": page.topics,
        "user_message": body.message,
        "persona_override": body.persona_override,
        "started_at": time.perf_counter(),
        "route": "continue",
        "trace": [],
    }

    async def events() -> AsyncIterator[dict[str, str]]:
        yield {"event": "start", "data": json.dumps({"run_id": run_id})}
        final: dict[str, Any] = {}
        try:
            async for update in c.graph.astream(state, stream_mode="updates"):
                for node, delta in update.items():
                    for entry in delta.get("trace", []):
                        yield {"event": "node", "data": json.dumps(entry)}
                    if node == "write":
                        final = delta
        except Exception:
            log.exception("event=graph_failed run_id=%s", run_id)
            yield {"event": "error", "data": json.dumps({"detail": prompts.ERROR_REPLY})}
            return
        yield {
            "event": "answer",
            "data": json.dumps(
                {
                    "run_id": run_id,
                    "route": final.get("route"),
                    "answer": final.get("final_answer", prompts.ERROR_REPLY),
                }
            ),
        }

    return EventSourceResponse(events())


# --- runs (write-node audit trail) ------------------------------------------------
@router.get("/runs", response_model=list[RunRecord])
async def list_runs(c: Dep, page_id: str | None = None, limit: int = 50) -> list[RunRecord]:
    return await c.repo.list_runs(page_id=page_id, limit=min(limit, 500))


@router.get("/runs/{run_id}", response_model=RunRecord)
async def get_run(run_id: str, c: Dep) -> RunRecord:
    run = await c.repo.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Unknown run.")
    return run
