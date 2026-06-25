"""Historical time-series router (HLD 8.3).

Serves bucketed history for the live metrics — occupancy and waiting keyed by
``zone_id``, entry/exit keyed by ``area_id`` — plus per-zone alert history. The
window bounds (``from``/``to``) accept epoch seconds or ISO-8601 and default to
the last 24 hours; ``bucket`` accepts a duration (default ``1h``). All query and
aggregation logic lives in the per-metric services; this router parses the window
and bucket and binds them to those services.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session, state_store
from app.api.routers._timeparse import parse_bucket_seconds, parse_instant
from app.api.schemas.metrics import AlertOut, HistorySeriesOut
from app.services.alert_service import AlertService
from app.services.entryexit_service import EntryExitService
from app.services.occupancy_service import OccupancyService
from app.services.state_store import StateStore
from app.services.waiting_service import WaitingService

router = APIRouter(tags=["history"], dependencies=[AuthDep])

_DEFAULT_WINDOW = timedelta(hours=24)
_DEFAULT_BUCKET = "1h"


def _resolve_window(frm: str | None, to: str | None) -> tuple[datetime, datetime]:
    """Resolve the ``[from, to)`` window, defaulting to the last 24 hours.

    Args:
        frm: Raw ``from`` value (epoch seconds or ISO-8601), or ``None``.
        to: Raw ``to`` value (epoch seconds or ISO-8601), or ``None``.

    Returns:
        The ``(start, end)`` pair as timezone-aware UTC datetimes.

    Raises:
        HTTPException: ``422`` if the resolved window is empty (``from >= to``).
    """
    window_end = parse_instant(to) if to is not None else datetime.now(tz=UTC)
    window_start = parse_instant(frm) if frm is not None else window_end - _DEFAULT_WINDOW
    if window_start >= window_end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="'from' must be earlier than 'to'",
        )
    return window_start, window_end


@router.get("/history/occupancy", response_model=HistorySeriesOut)
def occupancy_history(
    zone_id: int,
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    bucket: str = _DEFAULT_BUCKET,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> HistorySeriesOut:
    """Return bucketed mean-occupancy history for one zone.

    Args:
        zone_id: The zone to query.
        frm: Inclusive window start (epoch seconds or ISO-8601); aliased ``from``.
        to: Exclusive window end (epoch seconds or ISO-8601).
        bucket: Aggregation bucket width (e.g. ``"15m"``, ``"1h"``).
        session: Injected DB session owning the history query.
        store: Injected live-state cache (required by the service constructor).

    Returns:
        A :class:`HistorySeriesOut` of ``(bucket_start, avg_count)`` points.
    """
    window_start, window_end = _resolve_window(frm, to)
    bucket_seconds = parse_bucket_seconds(bucket)
    service = OccupancyService(session, store)
    return service.history(zone_id, window_start, window_end, bucket_seconds)


@router.get("/history/entry-exit", response_model=HistorySeriesOut)
def entry_exit_history(
    area_id: str,
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    bucket: str = _DEFAULT_BUCKET,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> HistorySeriesOut:
    """Return bucketed net-crossing history for one area.

    Args:
        area_id: The area to query.
        frm: Inclusive window start (epoch seconds or ISO-8601); aliased ``from``.
        to: Exclusive window end (epoch seconds or ISO-8601).
        bucket: Aggregation bucket width (e.g. ``"15m"``, ``"1h"``).
        session: Injected DB session owning the history query.
        store: Injected live-state cache (required by the service constructor).

    Returns:
        A :class:`HistorySeriesOut` of ``(bucket_start, net)`` points.
    """
    window_start, window_end = _resolve_window(frm, to)
    bucket_seconds = parse_bucket_seconds(bucket)
    service = EntryExitService(session, store)
    return service.history(area_id, window_start, window_end, bucket_seconds)


@router.get("/history/waiting", response_model=HistorySeriesOut)
def waiting_history(
    zone_id: int,
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    bucket: str = _DEFAULT_BUCKET,
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> HistorySeriesOut:
    """Return bucketed average-dwell history for one zone.

    Args:
        zone_id: The zone to query.
        frm: Inclusive window start (epoch seconds or ISO-8601); aliased ``from``.
        to: Exclusive window end (epoch seconds or ISO-8601).
        bucket: Aggregation bucket width (e.g. ``"15m"``, ``"1h"``).
        session: Injected DB session owning the history query.
        store: Injected live-state cache (required by the service constructor).

    Returns:
        A :class:`HistorySeriesOut` of ``(bucket_start, avg_dwell_s)`` points.
    """
    window_start, window_end = _resolve_window(frm, to)
    bucket_seconds = parse_bucket_seconds(bucket)
    service = WaitingService(session, store)
    return service.history(zone_id, window_start, window_end, bucket_seconds)


@router.get("/history/alerts", response_model=list[AlertOut])
def alert_history(
    zone_id: int,
    session: Session = Depends(db_session),
) -> list[AlertOut]:
    """Return every alert for ``zone_id``, newest first.

    Args:
        zone_id: The zone whose alert history to return.
        session: Injected DB session owning the read transaction.

    Returns:
        The zone's alerts as response DTOs.
    """
    service = AlertService(session)
    return service.history(zone_id)
