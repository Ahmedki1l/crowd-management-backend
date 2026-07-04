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
    FloorOccupancyOut,
    FloorSpaceOccupancy,
    HistorySeriesOut,
    OccupancyOut,
    SpaceOccupancyOut,
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

    def by_space(
        self, area_id: str | None = None, dt_space_id: str | None = None
    ) -> list[SpaceOccupancyOut]:
        """Return current occupancy summed per logical space (``dt_space_id``).

        Groups the live per-zone counts by ``dt_space_id`` and sums each group, so
        a logical space covered by several cameras — one zone row per camera, all
        sharing one ``dt_space_id`` — reports a single count. Zones with no
        ``dt_space_id`` are excluded (they belong to no logical space).

        Args:
            area_id: When given, restrict to zones whose camera's ``area`` matches
                (resolved by joining each zone to its camera), before grouping.
            dt_space_id: When given, return only this logical space (filter by the
                space id itself — the value the results are grouped by).

        Returns:
            One :class:`SpaceOccupancyOut` per ``dt_space_id`` with a live count,
            ordered by ``dt_space_id``.
        """
        states = self._store.all_occupancy()
        if area_id is not None:
            allowed = self._zone_ids_in_area(area_id)
            states = [s for s in states if s.zone_id in allowed]
        if dt_space_id is not None:
            states = [s for s in states if s.dt_space_id == dt_space_id]

        groups: dict[str, list[OccupancyState]] = {}
        for state in states:
            if state.dt_space_id is None:
                continue
            groups.setdefault(state.dt_space_id, []).append(state)

        result = [
            SpaceOccupancyOut(
                dt_space_id=space_id,
                count=sum(s.count for s in members),
                zone_ids=sorted(s.zone_id for s in members),
                ts=max(s.ts for s in members),
            )
            for space_id, members in groups.items()
        ]
        result.sort(key=lambda out: out.dt_space_id)
        return result

    def by_floor(self, floor: str | None = None) -> list[FloorOccupancyOut]:
        """Return current occupancy per physical floor, with a per-space breakdown.

        Resolves each live zone count to its camera's ``floor`` (zone -> camera),
        sums all of a floor's zones for the floor total, and nests the per-space
        (``dt_space_id``) sums within. Zones whose camera has no ``floor`` are
        excluded (they belong to no floor).

        Args:
            floor: When given, return only this floor (e.g. ``"B1"``).

        Returns:
            One :class:`FloorOccupancyOut` per floor with a live count, ordered by
            floor name.
        """
        states = self._store.all_occupancy()
        zones_by_id = {zone.id: zone for zone in self._zone_repo.list()}

        floor_states: dict[str, list[OccupancyState]] = {}
        for state in states:
            zone = zones_by_id.get(state.zone_id)
            camera = zone.camera if zone is not None else None
            if camera is None or camera.floor is None:
                continue
            floor_states.setdefault(camera.floor, []).append(state)

        if floor is not None:
            floor_states = {f: s for f, s in floor_states.items() if f == floor}

        result = [
            self._to_floor_out(name, members)
            for name, members in floor_states.items()
        ]
        result.sort(key=lambda out: out.floor)
        return result

    @staticmethod
    def _to_floor_out(
        floor: str, members: list[OccupancyState]
    ) -> FloorOccupancyOut:
        """Build a floor rollup: total across all zones + per-space breakdown."""
        spaces_map: dict[str, list[OccupancyState]] = {}
        for state in members:
            if state.dt_space_id is None:
                continue
            spaces_map.setdefault(state.dt_space_id, []).append(state)
        spaces = [
            FloorSpaceOccupancy(
                dt_space_id=space_id,
                count=sum(s.count for s in space_members),
                zone_ids=sorted(s.zone_id for s in space_members),
            )
            for space_id, space_members in sorted(spaces_map.items())
        ]
        return FloorOccupancyOut(
            floor=floor,
            count=sum(s.count for s in members),
            spaces=spaces,
            ts=max(s.ts for s in members),
        )

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
