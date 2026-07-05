"""Entry/exit read service backing the entry-exit metric router (HLD 8.2, 8.3).

Reads live per-area IN/OUT/net counts (with per-line breakdowns) from the
:class:`~app.services.state_store.StateStore` and bucketed crossing history from
the :class:`~app.db.repositories.crossing_repo.CrossingRepository`. It maps the
internal state value objects onto the pure response DTOs in
:mod:`app.api.schemas.metrics`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.api.schemas.metrics import (
    DailyEntryExitOut,
    EntryExitOut,
    HistorySeriesOut,
    LineCount,
    TimeBucket,
)
from app.db.repositories.crossing_repo import CrossingRepository
from app.services.state_store import AreaCountState, StateStore


class EntryExitService:
    """Query/read service for live and historical per-area entry/exit counts."""

    def __init__(self, session: Session, store: StateStore) -> None:
        """Bind the service to a DB session and the live state store.

        Args:
            session: An open SQLAlchemy session used for history queries.
            store: The current-state cache feeding the live ``current`` view.
        """
        self._session = session
        self._store = store
        self._crossing_repo = CrossingRepository(session)

    def current(self, area_id: str | None = None) -> list[EntryExitOut]:
        """Return current entry/exit counts, optionally for a single area.

        Args:
            area_id: When given, restrict the result to this area; otherwise
                return every area with a live count state.

        Returns:
            One :class:`EntryExitOut` per matching area, including its per-line
            breakdown.
        """
        if area_id is not None:
            state = self._store.get_counts(area_id)
            states = [state] if state is not None else []
        else:
            states = self._store.all_counts()
        return [self._to_out(state) for state in states]

    def history(
        self,
        area_id: str,
        frm: datetime,
        to: datetime,
        bucket_seconds: int,
    ) -> HistorySeriesOut:
        """Return bucketed net-crossing history for one area over ``[frm, to)``.

        The value of each bucket is the net flow (``in - out``) within it, which
        is the single scalar the generic history series carries.

        Args:
            area_id: The area to query.
            frm: Inclusive start of the window (timezone-aware UTC).
            to: Exclusive end of the window (timezone-aware UTC).
            bucket_seconds: Width of each aggregation bucket in seconds.

        Returns:
            A :class:`HistorySeriesOut` whose points are ``(bucket_start, net)``.
        """
        series = self._crossing_repo.query_series(area_id, frm, to, bucket_seconds)
        points = [
            TimeBucket(ts=ts, value=float(counts["net"])) for ts, counts in series
        ]
        return HistorySeriesOut(
            key=f"area:{area_id}",
            bucket=f"{bucket_seconds}s",
            points=points,
        )

    def daily(
        self,
        area_id: str,
        day_start: datetime,
        day_end: datetime,
        date_label: str,
    ) -> DailyEntryExitOut:
        """Return durable IN/OUT/net totals for ``area_id`` over one local day.

        Reads the persisted crossing events in ``[day_start, day_end)`` (UTC), so
        the totals survive process restarts — unlike the live in-memory counter.

        Args:
            area_id: The area to total (e.g. ``"main-entrance"``).
            day_start: Inclusive UTC start of the local day.
            day_end: Exclusive UTC end of the local day.
            date_label: The local calendar day (``YYYY-MM-DD``) being reported.

        Returns:
            A :class:`DailyEntryExitOut` with the day's IN/OUT/net totals; zeros
            when the area had no crossings that day.
        """
        counts = self._crossing_repo.totals(area_id, day_start, day_end)
        return DailyEntryExitOut(
            area_id=area_id,
            date=date_label,
            in_count=counts["in"],
            out_count=counts["out"],
            net=counts["net"],
        )

    @staticmethod
    def _to_out(state: AreaCountState) -> EntryExitOut:
        """Map a live :class:`AreaCountState` onto its response DTO."""
        lines = [
            LineCount(
                line_id=line.line_id,
                in_count=line.in_count,
                out_count=line.out_count,
            )
            for line in state.lines.values()
        ]
        return EntryExitOut(
            area_id=state.area_id,
            in_count=state.in_count,
            out_count=state.out_count,
            net=state.net,
            lines=lines,
            ts=state.ts,
        )
