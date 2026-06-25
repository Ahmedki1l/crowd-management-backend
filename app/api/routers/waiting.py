"""Live waiting/dwell metric router (HLD 8.2).

Exposes the current per-zone waiting state (current waits and rolling average
dwell) read from the in-memory state store, optionally filtered by zone. All read
logic lives in :class:`~app.services.waiting_service.WaitingService`; this router
only binds request parameters to that service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session, state_store
from app.api.schemas.metrics import WaitingOut
from app.services.state_store import StateStore
from app.services.waiting_service import WaitingService

router = APIRouter(tags=["metrics"], dependencies=[AuthDep])


@router.get("/waiting", response_model=list[WaitingOut])
def get_waiting(
    zone_id: int | None = None,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> list[WaitingOut]:
    """Return current waiting metrics, optionally for a single zone.

    Args:
        zone_id: When given, restrict the result to this zone; otherwise return
            every zone with a live waiting state.
        session: Injected DB session (held for parity with history queries).
        store: Injected live-state cache holding current waiting state.

    Returns:
        One :class:`WaitingOut` per matching zone.
    """
    service = WaitingService(session, store)
    return service.current(zone_id=zone_id)
