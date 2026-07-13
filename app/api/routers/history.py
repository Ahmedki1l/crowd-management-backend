"""Historical time-series router (HLD 8.3).

Serves history for occupancy (keyed by logical space, ``dt_space_id``) and entry/exit
(keyed by ``area_id``). Window bounds (``from``/``to``) accept epoch seconds or ISO-8601,
are interpreted and returned in **UTC**, and default to the last 24 hours.

Occupancy history is **pre-aggregated** into minute and hour rollups by
:mod:`app.services.occupancy_sampler` and :mod:`app.services.history_worker`, so ``bucket``
selects a stored grain rather than an arbitrary width: a query reads the grain it asked
for and never scans or re-buckets raw rows. All aggregation logic lives in the per-metric
services; this router parses the window and binds it to them.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session, state_store
from app.api.routers._timeparse import parse_bucket_seconds, parse_instant
from app.api.schemas.metrics import (
    DailyEntryExitOut,
    FloorHistoryOut,
    HistorySeriesOut,
    OccupancyHistoryOut,
)
from app.db.repositories.occupancy_repo import Grain
from app.services.entryexit_service import EntryExitService
from app.services.occupancy_service import OccupancyService
from app.services.state_store import StateStore

router = APIRouter(tags=["history"], dependencies=[AuthDep])

_DEFAULT_WINDOW = timedelta(hours=24)
_DEFAULT_BUCKET = "1h"

# The only occupancy buckets that exist, because they are the only ones stored.
_GRAINS = {"1h": Grain.HOUR, "1m": Grain.MINUTE}

# Cap on rollup rows a single request may read. At the hour grain that is ~5 years for
# one space; at the minute grain ~35 days. Bounded so an absurd window degrades into a
# truncated answer rather than an out-of-memory API process.
_MAX_ROWS = 50_000


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


def _resolve_day(date_str: str | None) -> tuple[datetime, datetime, str]:
    """Resolve one local calendar day to its ``[start, end)`` UTC bounds.

    The day is interpreted in the server's local timezone — correct for an
    on-site deployment where "today" means the building's day, while crossing
    timestamps are stored in UTC.

    Args:
        date_str: A ``YYYY-MM-DD`` calendar date, or ``None`` for today.

    Returns:
        ``(day_start_utc, day_end_utc, "YYYY-MM-DD")``.

    Raises:
        HTTPException: ``422`` if ``date_str`` is not a valid ``YYYY-MM-DD`` date.
    """
    local_tz = datetime.now().astimezone().tzinfo
    if date_str is not None:
        try:
            day = date.fromisoformat(date_str)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="'date' must be a YYYY-MM-DD calendar date",
            ) from exc
    else:
        day = datetime.now(local_tz).date()
    start_local = datetime(day.year, day.month, day.day, tzinfo=local_tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC), day.isoformat()


@router.get("/history/entry-exit/daily", response_model=DailyEntryExitOut)
def entry_exit_daily(
    area_id: str,
    day: str | None = Query(
        default=None,
        alias="date",
        description="Local calendar day YYYY-MM-DD; defaults to today",
    ),
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> DailyEntryExitOut:
    """Return durable IN/OUT/net totals for one area on one local day.

    Unlike the live ``/entry-exit`` counter (in-memory, resets on restart), this
    reads the persisted crossing events, so today's totals survive restarts.

    Args:
        area_id: The area to total (e.g. ``"main-entrance"``).
        day: Local calendar day ``YYYY-MM-DD`` (query param ``date``); today if omitted.
        session: Injected DB session owning the read.
        store: Injected live-state cache (required by the service constructor).

    Returns:
        A :class:`DailyEntryExitOut` with the day's totals and the date covered.
    """
    day_start, day_end, date_label = _resolve_day(day)
    return EntryExitService(session, store).daily(area_id, day_start, day_end, date_label)


def _resolve_grain(bucket: str) -> Grain:
    """Map the requested bucket onto a stored grain.

    Only the two grains that are actually stored are offered. Accepting an arbitrary
    width would mean re-bucketing at read time, which is exactly the raw-row scan these
    rollups exist to avoid.

    Raises:
        HTTPException: ``422`` for any bucket other than ``1h`` or ``1m``.
    """
    grain = _GRAINS.get(bucket)
    if grain is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"bucket must be one of {sorted(_GRAINS)}; occupancy history is "
            "pre-aggregated at those grains",
        )
    return grain


@router.get("/history/occupancy", response_model=OccupancyHistoryOut)
def occupancy_history(
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    bucket: str = _DEFAULT_BUCKET,
    space_id: list[str] | None = Query(default=None),
    floor: list[str] | None = Query(default=None),
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> OccupancyHistoryOut:
    """Return pre-aggregated occupancy history, one series per logical space.

    This is the Digital Twin's history feed. Each bucket carries the mean, the **peak**,
    and how many samples backed it — chart the average, but alarm on the peak, since an
    hourly mean of 12 can contain a 90-person spike.

    Args:
        frm: Inclusive window start (epoch seconds or ISO-8601, UTC); aliased ``from``.
            Defaults to 24 hours before ``to``.
        to: Exclusive window end (epoch seconds or ISO-8601, UTC). Defaults to now.
        bucket: ``1h`` (default) or ``1m`` — the two stored grains.
        space_id: Repeatable. Restrict to these spaces; omit for every space.
        floor: Repeatable. Restrict to spaces on these floors; omit for every floor.
        session: Injected DB session owning the history query.
        store: Injected live-state cache (required by the service constructor).

    Raises:
        HTTPException: ``422`` if the window is empty or ``bucket`` is not a stored grain.
    """
    window_start, window_end = _resolve_window(frm, to)
    grain = _resolve_grain(bucket)
    service = OccupancyService(session, store)
    return service.history(
        grain, window_start, window_end, space_id, floor, limit=_MAX_ROWS
    )


@router.get("/history/occupancy/floors", response_model=FloorHistoryOut)
def occupancy_history_by_floor(
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    bucket: str = _DEFAULT_BUCKET,
    floor: list[str] | None = Query(default=None),
    session: Session = Depends(db_session),
    store: StateStore = Depends(state_store),
) -> FloorHistoryOut:
    """Return occupancy history aggregated to floors — a floor's spaces summed per bucket.

    Args:
        frm: Inclusive window start (epoch seconds or ISO-8601, UTC); aliased ``from``.
        to: Exclusive window end (epoch seconds or ISO-8601, UTC).
        bucket: ``1h`` (default) or ``1m``.
        floor: Repeatable. Restrict to these floors; omit for every floor.
        session: Injected DB session owning the history query.
        store: Injected live-state cache (required by the service constructor).

    Raises:
        HTTPException: ``422`` if the window is empty or ``bucket`` is not a stored grain.
    """
    window_start, window_end = _resolve_window(frm, to)
    grain = _resolve_grain(bucket)
    service = OccupancyService(session, store)
    return service.history_by_floor(
        grain, window_start, window_end, floor, limit=_MAX_ROWS
    )


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
