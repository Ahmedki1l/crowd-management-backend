"""Waiting/dwell read service backing the waiting metric router (HLD 8.2, 8.3).

Reads live per-zone waiting state (current waits and rolling average dwell) from
the :class:`~app.services.state_store.StateStore` and bucketed average-dwell
history from the :class:`~app.db.repositories.dwell_repo.DwellRepository`. It maps
the internal state value objects onto the pure response DTOs in
:mod:`app.api.schemas.metrics`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.api.schemas.metrics import (
    HistorySeriesOut,
    TimeBucket,
    WaitingOut,
)
from app.db.repositories.dwell_repo import DwellRepository
from app.services.state_store import StateStore, WaitingState


class WaitingService:
    """Query/read service for live and historical zone waiting metrics."""

    def __init__(self, session: Session, store: StateStore) -> None:
        """Bind the service to a DB session and the live state store.

        Args:
            session: An open SQLAlchemy session used for history queries.
            store: The current-state cache feeding the live ``current`` view.
        """
        self._session = session
        self._store = store
        self._dwell_repo = DwellRepository(session)

    def current(self, zone_id: int | None = None) -> list[WaitingOut]:
        """Return current waiting metrics, optionally for a single zone.

        Args:
            zone_id: When given, restrict the result to this zone; otherwise
                return every zone with a live waiting state.

        Returns:
            One :class:`WaitingOut` per matching zone.
        """
        if zone_id is not None:
            state = self._store.get_waiting(zone_id)
            states = [state] if state is not None else []
        else:
            states = self._store.all_waiting()
        return [self._to_out(state) for state in states]

    def history(
        self,
        zone_id: int,
        frm: datetime,
        to: datetime,
        bucket_seconds: int,
    ) -> HistorySeriesOut:
        """Return bucketed average-dwell history for one zone over ``[frm, to)``.

        Args:
            zone_id: The zone to query.
            frm: Inclusive start of the window (timezone-aware UTC).
            to: Exclusive end of the window (timezone-aware UTC).
            bucket_seconds: Width of each aggregation bucket in seconds.

        Returns:
            A :class:`HistorySeriesOut` whose points are
            ``(bucket_start, avg_dwell_s)``.
        """
        series = self._dwell_repo.query_series(zone_id, frm, to, bucket_seconds)
        points = [TimeBucket(ts=ts, value=value) for ts, value in series]
        return HistorySeriesOut(
            key=f"zone:{zone_id}",
            bucket=f"{bucket_seconds}s",
            points=points,
        )

    @staticmethod
    def _to_out(state: WaitingState) -> WaitingOut:
        """Map a live :class:`WaitingState` onto its response DTO."""
        return WaitingOut(
            zone_id=state.zone_id,
            dt_space_id=state.dt_space_id,
            current_waits=state.current_waits,
            avg_dwell_s=state.avg_dwell_s,
            ts=state.ts,
        )
