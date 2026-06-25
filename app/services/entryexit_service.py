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
