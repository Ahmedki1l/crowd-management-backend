"""Zone geometry router (HLD 8.1).

CRUD over the ``zones`` table via :class:`~app.db.repositories.zone_repo.ZoneRepository`.
The ORM row maps straight onto :class:`~app.api.schemas.geometry.ZoneOut`
(``type`` is stored as its string value, ``polygon`` as ``[[x, y], ...]``), so
``ZoneOut.model_validate`` is the single mapping point — no parallel DTO.

Creates are rejected with 404 if their ``camera_id`` does not exist, so a zone
can never be orphaned from its camera (HLD 9).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session
from app.api.schemas.geometry import ZoneCreate, ZoneOut, ZoneUpdate
from app.db.models.geometry import Zone
from app.db.repositories.camera_repo import CameraRepository
from app.db.repositories.zone_repo import ZoneRepository

router = APIRouter(prefix="/zones", tags=["zones"], dependencies=[AuthDep])


def _to_columns(payload: ZoneCreate | ZoneUpdate) -> dict[str, Any]:
    """Build a column->value map from a set zone payload, encoding the enum type.

    Only fields the caller actually provided are included so updates stay
    partial; the ``type`` enum is stored as its string value to match the ORM
    column.
    """
    data = payload.model_dump(exclude_unset=True)
    if data.get("type") is not None:
        data["type"] = payload.type.value
    return data


def _to_out(zone: Zone) -> ZoneOut:
    """Map a :class:`Zone` ORM row to its response model."""
    return ZoneOut.model_validate(zone)


@router.get("", response_model=list[ZoneOut])
def list_zones(session: Session = Depends(db_session)) -> list[ZoneOut]:
    """Return every zone ordered by id."""
    return [_to_out(zone) for zone in ZoneRepository(session).list()]


@router.post("", response_model=ZoneOut, status_code=status.HTTP_201_CREATED)
def create_zone(payload: ZoneCreate, session: Session = Depends(db_session)) -> ZoneOut:
    """Create a zone after verifying its parent camera exists."""
    if CameraRepository(session).get(payload.camera_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"camera {payload.camera_id} not found",
        )
    zone = ZoneRepository(session).create(_to_columns(payload))
    return _to_out(zone)


@router.get("/{zone_id}", response_model=ZoneOut)
def get_zone(zone_id: int, session: Session = Depends(db_session)) -> ZoneOut:
    """Return one zone by id, or 404 if it does not exist."""
    zone = ZoneRepository(session).get(zone_id)
    if zone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"zone {zone_id} not found"
        )
    return _to_out(zone)


@router.patch("/{zone_id}", response_model=ZoneOut)
def update_zone(
    zone_id: int, payload: ZoneUpdate, session: Session = Depends(db_session)
) -> ZoneOut:
    """Apply a partial update to a zone, or 404 if it does not exist."""
    zone = ZoneRepository(session).update(zone_id, _to_columns(payload))
    if zone is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"zone {zone_id} not found"
        )
    return _to_out(zone)


@router.delete("/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_zone(zone_id: int, session: Session = Depends(db_session)) -> None:
    """Delete a zone, or 404 if it does not exist."""
    if not ZoneRepository(session).delete(zone_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"zone {zone_id} not found"
        )
