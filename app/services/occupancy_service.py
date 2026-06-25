"""Occupancy read service backing the occupancy metric router (HLD 8.2, 8.3).

Reads live per-zone people counts from the in-memory
:class:`~app.services.state_store.StateStore` and bucketed history from the
:class:`~app.db.repositories.occupancy_repo.OccupancyRepository`. It maps the
internal state/value objects onto the pure response DTOs in
:mod:`app.api.schemas.metrics` — it never re-models those concepts.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.api.schemas.metrics import (
    HistorySeriesOut,
    OccupancyOut,
    TimeBucket,
)
from app.db.repositories.occupancy_repo import OccupancyRepository
from app.db.repositories.zone_repo import ZoneRepository
from app.services.state_store import OccupancyState, StateStore


class OccupancyService:
    """Query/read service for live and historical zone occupancy."""

    def __init__(self, session: Session, store: StateStore) -> None:
        """Bind the service to a DB session and the live state store.

        Args:
            session: An open SQLAlchemy session used for history queries.
            store: The current-state cache feeding the live ``current`` view.
        """
        self._session = session
        self._store = store
        self._occupancy_repo = OccupancyRepository(session)
        self._zone_repo = ZoneRepository(session)

    def current(
        self,
        zone_id: int | None = None,
        area_id: str | None = None,
    ) -> list[OccupancyOut]:
        """Return current occupancy, optionally filtered by zone and/or area.

        Args:
            zone_id: When given, restrict the result to this single zone.
            area_id: When given, restrict to zones whose camera's ``area``
                matches (resolved by joining each zone to its camera).

        Returns:
            One :class:`OccupancyOut` per matching zone with a live state entry.
        """
        if zone_id is not None:
            state = self._store.get_occupancy(zone_id)
            states = [state] if state is not None else []
        else:
            states = self._store.all_occupancy()

        if area_id is not None:
            allowed = self._zone_ids_in_area(area_id)
            states = [s for s in states if s.zone_id in allowed]

        return [self._to_out(state) for state in states]

    def history(
        self,
        zone_id: int,
        frm: datetime,
        to: datetime,
        bucket_seconds: int,
    ) -> HistorySeriesOut:
        """Return bucketed mean-occupancy history for one zone over ``[frm, to)``.

        Args:
            zone_id: The zone to query.
            frm: Inclusive start of the window (timezone-aware UTC).
            to: Exclusive end of the window (timezone-aware UTC).
            bucket_seconds: Width of each aggregation bucket in seconds.

        Returns:
            A :class:`HistorySeriesOut` whose points are ``(bucket_start, avg)``.
        """
        series = self._occupancy_repo.query_series(zone_id, frm, to, bucket_seconds)
        points = [TimeBucket(ts=ts, value=value) for ts, value in series]
        return HistorySeriesOut(
            key=f"zone:{zone_id}",
            bucket=f"{bucket_seconds}s",
            points=points,
        )

    def _zone_ids_in_area(self, area_id: str) -> set[int]:
        """Return the ids of zones whose camera belongs to ``area_id``."""
        return {
            zone.id
            for zone in self._zone_repo.list()
            if zone.camera is not None and zone.camera.area == area_id
        }

    @staticmethod
    def _to_out(state: OccupancyState) -> OccupancyOut:
        """Map a live :class:`OccupancyState` onto its response DTO."""
        return OccupancyOut(
            zone_id=state.zone_id,
            dt_space_id=state.dt_space_id,
            count=state.count,
            ts=state.ts,
        )
