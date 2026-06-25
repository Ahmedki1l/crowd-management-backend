"""Camera heat-map router (HLD 7, 8.3).

Returns the summed density grid for one camera over a time window. The window
bounds (``from``/``to``) accept epoch seconds or ISO-8601 and default to the last
24 hours. All aggregation logic lives in
:class:`~app.services.heatmap_service.HeatmapService`; this router only parses the
window and binds parameters to that service.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, config, db_session
from app.api.routers._timeparse import parse_instant
from app.api.schemas.metrics import HeatmapOut
from app.config.schema import AppConfig
from app.services.heatmap_service import HeatmapService

router = APIRouter(tags=["metrics"], dependencies=[AuthDep])

_DEFAULT_WINDOW = timedelta(hours=24)


@router.get("/heatmap", response_model=HeatmapOut)
def get_heatmap(
    camera_id: int,
    frm: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None, alias="to"),
    resolution: int | None = None,
    include_overlay: bool = Query(default=False),
    session: Session = Depends(db_session),
    settings: AppConfig = Depends(config),
) -> HeatmapOut:
    """Return the density heat-map for ``camera_id`` over ``[from, to)``.

    When ``to`` is omitted it defaults to now; when ``from`` is omitted it
    defaults to 24 hours before ``to``.

    Args:
        camera_id: The camera whose density grids to aggregate.
        frm: Inclusive window start (epoch seconds or ISO-8601); aliased ``from``.
        to: Exclusive window end (epoch seconds or ISO-8601).
        resolution: Optional caller-requested down-sampling hint (forwarded).
        include_overlay: When ``True``, also render a heat overlay PNG and return
            its relative path in ``overlay_url`` (FR-HM-03). Defaults to
            ``False``, leaving ``overlay_url`` ``None`` as before. Rendering is
            best-effort and never fails the read.
        session: Injected DB session owning the range query.
        settings: Injected application configuration (for ``snapshots.dir``).

    Returns:
        A :class:`HeatmapOut` with one cell per non-empty grid position.

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

    service = HeatmapService(session)
    return service.range(
        camera_id,
        window_start,
        window_end,
        resolution=resolution,
        include_overlay=include_overlay,
        base_dir=settings.snapshots.dir,
    )
