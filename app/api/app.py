"""FastAPI application factory (HLD 6.6 / 12).

Wires the runtime on startup: the event bus fans out to the state projector
(updates the read model), the persistence projector (writes time-series to the
DB off the hot path), and — when configured — the Digital Twin push publisher.
SSE subscribers attach per-request. Cameras/zones/lines are managed via routers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import all_routers
from app.config.settings import get_settings
from app.db.session import init_db
from app.events.event_bus import get_event_bus
from app.services.state_store import get_state_store
from app.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()

    # The same wiring used by the worker entry points, so the bus consumers
    # (read-model projector, persistence projector) never drift.
    from app.services.runtime_wiring import RuntimeWiring

    # Schema first: the wiring starts the history writers, which begin polling the DB
    # immediately. Starting them against a database whose tables do not exist yet is a
    # guaranteed first-tick failure (it self-heals on the next tick, but it logs a stack
    # trace on every cold start, which trains people to ignore stack traces).
    # Convenience for local/dev (SQLite). Production schema is managed by Alembic.
    if settings.database.url.startswith("sqlite"):
        init_db()

    wiring = RuntimeWiring(get_event_bus(), get_state_store())
    wiring.start()
    app.state.runtime_wiring = wiring

    try:
        yield
    finally:
        await wiring.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Camera-Based Analytics Service",
        version="1.0.0",
        description="Backend analytics for the Digital Twin (occupancy, entry/exit).",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in all_routers:
        app.include_router(router, prefix=settings.api.prefix)

    @app.get("/health", tags=["ops"], include_in_schema=False)
    def liveness() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
