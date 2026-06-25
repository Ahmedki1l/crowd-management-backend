"""Consolidated state and dashboard-stats router (HLD 8.2).

``GET /state`` returns the full current world (occupancy, entry/exit, waiting,
active alerts) a Digital Twin client loads before following the live event stream.
``GET /stats`` returns the operator dashboard summary tile. Assembly logic lives
in :class:`~app.services.state_service.StateService` and
:class:`~app.services.stats_service.StatsService`; this router only binds the
injected session/store to them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session, state_store
from app.api.schemas.metrics import StateOut, StatsOut
from app.services.state_service import StateService
from app.services.state_store import StateStore
from app.services.stats_service import StatsService

router = APIRouter(tags=["state"], dependencies=[AuthDep])


@router.get("/state", response_model=StateOut)
def get_state(
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> StateOut:
    """Return the consolidated current-state snapshot for the DT initial load.

    Args:
        session: Injected DB session shared by the delegate metric services.
        store: Injected live-state cache feeding the metric views.

    Returns:
        A :class:`StateOut` with every live metric plus active alerts.
    """
    service = StateService(session, store)
    return service.snapshot()


@router.get("/stats", response_model=StatsOut)
def get_stats(
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> StatsOut:
    """Return the operator dashboard summary tile.

    Args:
        session: Injected DB session used for the static totals.
        store: Injected live-state cache feeding the live counts.

    Returns:
        A :class:`StatsOut` combining DB totals with live health/occupancy/alerts.
    """
    service = StatsService(session, store)
    return service.build()
