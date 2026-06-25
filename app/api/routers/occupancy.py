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
from app.api.schemas.metrics import OccupancyOut
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
