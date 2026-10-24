"""FastAPI application factory and server entry point (API only; the UI lives in frontend/)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from journey_agent import __version__
from journey_agent.api.routes import router
from journey_agent.config import Settings, get_settings
from journey_agent.container import Container, build_container
from journey_agent.log import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or await build_container(settings)
        yield

    app = FastAPI(title="Nubazo Journey Agent", version=__version__, lifespan=lifespan)
    # The console and widget iframe are served from the frontend origin(s), not this one.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.frontend_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def console() -> RedirectResponse:
        # The console is served by frontend/; keep old bookmarks of the API root working.
        return RedirectResponse(settings.frontend_origins[0])

    return app


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "journey_agent.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
