"""Live occupancy metric router (HLD 8.2).

Exposes the current per-zone people count read from the in-memory state store,
optionally filtered by zone or by the area its camera belongs to. All read logic
lives in :class:`~app.services.occupancy_service.OccupancyService`; this router
only binds request parameters to that service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session, state_store
from app.api.schemas.metrics import FloorOccupancyOut, OccupancyOut, SpaceOccupancyOut
from app.api.validation_capture import capture_validation_images
from app.services.occupancy_service import OccupancyService
from app.services.state_store import StateStore

router = APIRouter(tags=["metrics"], dependencies=[AuthDep])


@router.get("/occupancy", response_model=list[OccupancyOut])
def get_occupancy(
    zone_id: int | None = None,
    area_id: str | None = None,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> list[OccupancyOut]:
    """Return current occupancy, optionally filtered by ``zone_id``/``area_id``.

    Args:
        zone_id: When given, restrict the result to this single zone.
        area_id: When given, restrict to zones whose camera belongs to the area.
        session: Injected DB session (used to resolve area membership).
        store: Injected live-state cache holding current counts.

    Returns:
        One :class:`OccupancyOut` per matching zone with a live count.
    """
    service = OccupancyService(session, store)
    return service.current(zone_id=zone_id, area_id=area_id)


@router.get("/occupancy/spaces", response_model=list[SpaceOccupancyOut])
def get_occupancy_by_space(
    dt_space_id: str | None = None,
    area_id: str | None = None,
    refresh_images: bool = False,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> list[SpaceOccupancyOut]:
    """Return current occupancy summed per logical space (``dt_space_id``).

    A logical space covered by several cameras is stored as one zone row per
    camera sharing a ``dt_space_id``; this sums them into a single count — the
    per-zone number for a multi-camera area.

    Args:
        dt_space_id: When given, return only this logical space (e.g.
            ``b1-waiting-area``). This is the value the results are grouped by.
        area_id: When given, restrict to zones whose camera's ``area`` matches —
            a coarser, camera-level filter (not the space id).
        refresh_images: Dev aid — when true, also refresh the annotated
            validation snapshots (fetch + detect on every camera, wiping old)
            before returning. Heavy; leave false for normal reads.
        session: Injected DB session (used to resolve area membership).
        store: Injected live-state cache holding current counts.

    Returns:
        One :class:`SpaceOccupancyOut` per ``dt_space_id`` with a live count.
    """
    if refresh_images:
        capture_validation_images(session)
    service = OccupancyService(session, store)
    return service.by_space(area_id=area_id, dt_space_id=dt_space_id)


@router.get("/occupancy/floors", response_model=list[FloorOccupancyOut])
def get_occupancy_by_floor(
    floor: str | None = None,
    refresh_images: bool = False,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> list[FloorOccupancyOut]:
    """Return current occupancy per physical floor, with a per-space breakdown.

    A floor groups the logical spaces that group the zones (resolved via each
    zone's camera ``floor``). Each floor's ``count`` is the sum of every zone on
    that floor; ``spaces`` breaks it down per ``dt_space_id``. Filter to one
    floor with ``?floor=B1``.

    Args:
        floor: When given, return only this floor (e.g. ``B1``).
        refresh_images: Dev aid — when true, also refresh the annotated
            validation snapshots (fetch + detect on every camera, wiping old)
            before returning. Heavy; leave false for normal reads.
        session: Injected DB session (resolves zone -> camera -> floor).
        store: Injected live-state cache holding current counts.

    Returns:
        One :class:`FloorOccupancyOut` per floor with a live count.
    """
    if refresh_images:
        capture_validation_images(session)
    service = OccupancyService(session, store)
    return service.by_floor(floor=floor)
