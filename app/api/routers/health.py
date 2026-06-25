"""Operational health, readiness, and metrics router (HLD 8.4 / 15).

Three operational endpoints:

* ``GET /ready`` — readiness probe: ``200`` only when the database is reachable;
  reports the best-effort ``models_loaded`` flag exposed by the state store.
* ``GET /metrics`` — Prometheus exposition built from the live
  :class:`~app.services.state_store.StateStore` (per-camera gauges plus totals).
* ``GET /cameras/{camera_id}/health`` — per-camera health from the state store.

Liveness (``GET /health``) is defined in :mod:`app.api.app`; it is intentionally
not redefined here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import db_session, state_store
from app.api.schemas.ops import CameraHealthOut, ReadyOut
from app.services.state_store import StateStore
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["ops"])


def _db_reachable(session: Session) -> bool:
    """Return whether the database answers a trivial ``SELECT 1``.

    A failure is logged and reported as not-ready rather than propagated, so the
    probe can return a structured ``503`` instead of a ``500``.

    Args:
        session: The database session to probe.

    Returns:
        ``True`` if the query succeeded, ``False`` otherwise.
    """
    try:
        session.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("readiness DB check failed")
        return False


def _models_loaded(store: StateStore) -> bool:
    """Report the state store's ``models_loaded`` flag when it exposes one.

    The in-memory store does not currently track inference-model readiness, so
    this defaults to ``True`` (the API is not the component that loads models)
    while honouring a future ``models_loaded`` attribute if one is added.

    Args:
        store: The live state store.

    Returns:
        The store's ``models_loaded`` flag, or ``True`` when unavailable.
    """
    flag = getattr(store, "models_loaded", True)
    return bool(flag)


@router.get("/ready", response_model=ReadyOut)
def readiness(
    response: Response,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> ReadyOut:
    """Report readiness: ``200`` only when the database is reachable.

    Args:
        response: The response whose status code is set to ``503`` when not ready.
        session: The injected database session.
        store: The injected live state store.

    Returns:
        A :class:`ReadyOut` describing DB reachability and model-load state. The
        HTTP status is ``503`` when the database is unreachable.
    """
    db_ok = _db_reachable(session)
    models_loaded = _models_loaded(store)
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyOut(
        status="ready" if db_ok else "not_ready",
        db_ok=db_ok,
        models_loaded=models_loaded,
    )


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(store: StateStore = Depends(state_store)) -> Response:
    """Expose per-camera and total runtime metrics in Prometheus text format.

    Builds a request-scoped registry from the live state store so the exposition
    reflects the current read model without mutating any global collector.

    Args:
        store: The injected live state store.

    Returns:
        A Prometheus exposition response (``text/plain``).
    """
    registry = CollectorRegistry()
    fps_gauge = Gauge(
        "camera_fps", "Processing FPS per camera.", ["camera_id"], registry=registry
    )
    frame_age_gauge = Gauge(
        "camera_last_frame_age_seconds",
        "Seconds since the last processed frame per camera.",
        ["camera_id"],
        registry=registry,
    )
    queue_depth_gauge = Gauge(
        "camera_queue_depth", "Frame queue depth per camera.", ["camera_id"], registry=registry
    )
    healthy_gauge = Gauge(
        "camera_healthy", "1 if the camera is healthy, else 0.", ["camera_id"], registry=registry
    )
    cameras_total_gauge = Gauge(
        "cameras_total", "Number of cameras reporting health.", registry=registry
    )
    cameras_healthy_gauge = Gauge(
        "cameras_healthy", "Number of healthy cameras.", registry=registry
    )
    total_occupancy_gauge = Gauge(
        "total_occupancy", "Total people across all occupancy zones.", registry=registry
    )

    health_states = store.all_camera_health()
    for health in health_states:
        label = str(health.camera_id)
        fps_gauge.labels(camera_id=label).set(health.fps)
        frame_age_gauge.labels(camera_id=label).set(health.last_frame_age_s)
        queue_depth_gauge.labels(camera_id=label).set(health.queue_depth)
        healthy_gauge.labels(camera_id=label).set(1 if health.healthy else 0)

    cameras_total_gauge.set(len(health_states))
    cameras_healthy_gauge.set(sum(1 for h in health_states if h.healthy))
    total_occupancy_gauge.set(store.total_occupancy())

    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@router.get("/cameras/{camera_id}/health", response_model=CameraHealthOut)
def camera_health(
    camera_id: int,
    store: StateStore = Depends(state_store),
) -> CameraHealthOut:
    """Return the latest health snapshot for one camera.

    Args:
        camera_id: Identifier of the camera to inspect.
        store: The injected live state store.

    Returns:
        The camera's latest :class:`CameraHealthOut`.

    Raises:
        HTTPException: ``404`` if the camera has not reported any health yet.
    """
    health = store.get_camera_health(camera_id)
    if health is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="camera health unknown"
        )
    return CameraHealthOut(
        camera_id=health.camera_id,
        fps=health.fps,
        last_frame_age_s=health.last_frame_age_s,
        queue_depth=health.queue_depth,
        healthy=health.healthy,
        ts=health.ts,
    )
