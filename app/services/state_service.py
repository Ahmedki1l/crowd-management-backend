"""Consolidated state snapshot service backing GET /state (HLD 8.2).

When a Digital Twin / dashboard client first connects it needs the whole current
world in one shot before it starts following the live event stream. This service
assembles that snapshot by delegating to the per-metric query services, so the
state->DTO mapping lives in exactly one place per metric.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.schemas.metrics import StateOut
from app.services.entryexit_service import EntryExitService
from app.services.occupancy_service import OccupancyService
from app.services.state_store import StateStore
from app.utils.clock import system_clock


class StateService:
    """Assembles the consolidated current-state snapshot for the DT initial load."""

    def __init__(self, session: Session, store: StateStore) -> None:
        """Bind the service to a DB session and the live state store.

        Args:
            session: An open SQLAlchemy session shared by the delegate services.
            store: The current-state cache feeding the live metric views.
        """
        self._occupancy = OccupancyService(session, store)
        self._entry_exit = EntryExitService(session, store)

    def snapshot(self) -> StateOut:
        """Return the full current state: occupancy and entry/exit.

        Returns:
            A :class:`StateOut` carrying every live metric, stamped with the wall clock.
        """
        return StateOut(
            occupancy=self._occupancy.current(),
            entry_exit=self._entry_exit.current(),
            ts=system_clock().now(),
        )
