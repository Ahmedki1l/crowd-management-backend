"""API router aggregation.

Collects every domain router exposed under :mod:`app.api.routers` into a single
ordered ``all_routers`` list so the application factory can mount them in one
pass. The order mirrors the HLD section sequence: camera registry and geometry
first, then configuration, then the metric/state/history read paths, and finally
the streaming, snapshot, and operational endpoints.

Each imported module owns its own ``APIRouter`` (prefix, tags, and auth
dependency); this package only sequences them and adds nothing of its own.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routers.cameras import router as cameras_router
from app.api.routers.config import router as config_router
from app.api.routers.engine import router as engine_router
from app.api.routers.entry_exit import router as entry_exit_router
from app.api.routers.health import router as health_router
from app.api.routers.history import router as history_router
from app.api.routers.lines import router as lines_router
from app.api.routers.occupancy import router as occupancy_router
from app.api.routers.state import router as state_router
from app.api.routers.stream import router as stream_router
from app.api.routers.tools import router as tools_router
from app.api.routers.zones import router as zones_router

all_routers: list[APIRouter] = [
    cameras_router,
    zones_router,
    lines_router,
    config_router,
    occupancy_router,
    entry_exit_router,
    state_router,
    history_router,
    stream_router,
    tools_router,
    engine_router,
    health_router,
]

__all__ = ["all_routers"]
