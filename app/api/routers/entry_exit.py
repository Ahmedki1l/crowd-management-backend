"""Live entry/exit metric router (HLD 8.2).

Exposes the current per-area IN/OUT/net counts (with per-line breakdowns) read
from the in-memory state store, optionally filtered by area. All read logic lives
in :class:`~app.services.entryexit_service.EntryExitService`; this router only
binds request parameters to that service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session, state_store
from app.api.schemas.metrics import EntryExitOut
from app.services.entryexit_service import EntryExitService
from app.services.state_store import StateStore

router = APIRouter(tags=["metrics"], dependencies=[AuthDep])


@router.get("/entry-exit", response_model=list[EntryExitOut])
def get_entry_exit(
    area_id: str | None = None,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> list[EntryExitOut]:
    """Return current entry/exit counts, optionally for a single area.

    Args:
        area_id: When given, restrict the result to this area; otherwise return
            every area with a live count state.
        session: Injected DB session (held for parity with history queries).
        store: Injected live-state cache holding current crossing counts.

    Returns:
        One :class:`EntryExitOut` per matching area, including its per-line
        breakdown.
    """
    service = EntryExitService(session, store)
    return service.current(area_id=area_id)
