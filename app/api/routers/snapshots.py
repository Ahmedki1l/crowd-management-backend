"""Evidence-snapshot serving router (HLD 8.4).

Two read paths over the stored JPEG snapshots that live under
``snapshots.dir`` (see :mod:`app.utils.snapshot`):

* ``GET /snapshots/{filepath}`` — serve a specific stored snapshot by its
  relative path (the value persisted in ``Snapshot.path``), with strict
  path-traversal protection.
* ``GET /zones/{zone_id}/snapshot/live`` — a best-effort "live" view: the most
  recent stored snapshot for the zone's camera. Live frame capture is not served
  from the API process, so this honestly returns ``503`` when no snapshot is
  available rather than fabricating an image.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, config, db_session
from app.config.schema import AppConfig
from app.db.models.alerts import Snapshot
from app.db.models.geometry import Zone
from app.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["snapshots"], dependencies=[AuthDep])

_JPEG_MEDIA_TYPE = "image/jpeg"


@router.get("/snapshots/{filepath:path}")
def get_snapshot(
    filepath: str,
    settings: AppConfig = Depends(config),
) -> FileResponse:
    """Serve a stored snapshot file by its relative path.

    Args:
        filepath: Snapshot path relative to ``snapshots.dir`` (the value stored
            in ``Snapshot.path``).
        settings: The injected application configuration.

    Returns:
        The JPEG snapshot as a file response.

    Raises:
        HTTPException: ``400`` if the path escapes the snapshot directory,
            ``404`` if no file exists at the resolved location.
    """
    from app.utils.snapshot import resolve_snapshot

    try:
        abs_path = resolve_snapshot(settings.snapshots.dir, filepath)
    except ValueError as exc:
        logger.warning(
            "rejected snapshot path",
            extra={"event": "snapshot_path_rejected"},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid snapshot path"
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="snapshot not found"
        ) from exc
    return FileResponse(abs_path, media_type=_JPEG_MEDIA_TYPE)


@router.get("/zones/{zone_id}/snapshot/live")
def get_zone_live_snapshot(
    zone_id: int,
    session: Session = Depends(db_session),
    settings: AppConfig = Depends(config),
) -> FileResponse:
    """Return the most recent stored snapshot for the zone's camera.

    Live frame capture runs in the worker processes, not the API, so this
    endpoint serves the freshest snapshot already persisted for the zone's
    camera as a best-effort "live" view.

    Args:
        zone_id: Identifier of the zone whose camera's snapshot is requested.
        session: The injected database session.
        settings: The injected application configuration.

    Returns:
        The most recent JPEG snapshot for the zone's camera.

    Raises:
        HTTPException: ``404`` if the zone is unknown; ``503`` if no snapshot is
            available (none stored, or the file is missing on disk).
    """
    from app.utils.snapshot import resolve_snapshot

    zone = session.get(Zone, zone_id)
    if zone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="zone not found"
        )

    snapshot = session.scalars(
        select(Snapshot)
        .where(Snapshot.camera_id == zone.camera_id)
        .order_by(Snapshot.ts.desc())
        .limit(1)
    ).first()
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no live snapshot available for this zone",
        )

    try:
        abs_path = resolve_snapshot(settings.snapshots.dir, snapshot.path)
    except (ValueError, FileNotFoundError) as exc:
        # The DB references a snapshot whose file is missing or unsafe; this is a
        # real availability gap, not a client error — report it honestly.
        logger.warning(
            "stored snapshot file unavailable",
            extra={"camera_id": zone.camera_id, "event": "snapshot_file_unavailable"},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="live snapshot unavailable",
        ) from exc
    return FileResponse(abs_path, media_type=_JPEG_MEDIA_TYPE)
