"""Engine-control endpoints: switch what the engine runs at runtime (ops).

Buttons for the single-node ``--api`` deployment to change the engine's scope
without restarting the process:

* ``POST /engine/all``          — run every enabled (allowlisted) camera
* ``POST /engine/entry-exit``   — run only the ``entry_exit`` cameras
* ``POST /engine/occupancy``    — run only the ``occupancy`` cameras
* ``POST /engine/reset``        — reload config (.env + YAML) and restart on all
* ``POST /engine/stop``         — stop every pipeline (idle)
* ``GET  /engine/status``       — current scope + pipeline count

All routes require auth. A switch rebuilds the perception backends, so it takes a
few seconds; requests run in FastAPI's thread pool (plain ``def``) so this does
not block the event loop.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.deps import AuthDep
from app.engine.manager import get_engine_manager

router = APIRouter(prefix="/engine", tags=["engine"])


@router.get("/status", dependencies=[AuthDep])
def engine_status() -> dict:
    """Report the current engine scope, whether it's running, and pipeline count."""
    return get_engine_manager().status()


@router.post("/all", dependencies=[AuthDep])
def run_all() -> dict:
    """Run every enabled (allowlisted) camera — occupancy and entry/exit."""
    return get_engine_manager().switch("all")


@router.post("/entry-exit", dependencies=[AuthDep])
def run_entry_exit() -> dict:
    """Run only the cameras with the ``entry_exit`` role."""
    return get_engine_manager().switch("entry_exit")


@router.post("/occupancy", dependencies=[AuthDep])
def run_occupancy() -> dict:
    """Run only the cameras with the ``occupancy`` role."""
    return get_engine_manager().switch("occupancy")


@router.post("/reset", dependencies=[AuthDep])
def reset_engine() -> dict:
    """Reload config (.env + YAML) and restart the engine on every camera."""
    return get_engine_manager().reset()


@router.post("/stop", dependencies=[AuthDep])
def stop_engine() -> dict:
    """Stop every pipeline and leave the engine idle."""
    return get_engine_manager().stop()
