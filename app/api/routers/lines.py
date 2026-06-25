"""Counting-line geometry router (HLD 8.1).

CRUD over the ``lines`` table via :class:`~app.db.repositories.line_repo.LineRepository`.
The ORM row maps straight onto :class:`~app.api.schemas.geometry.LineOut`
(``points`` and ``in_direction`` are stored as JSON), so ``LineOut.model_validate``
is the single mapping point — no parallel DTO.

As with zones, a create whose ``camera_id`` is unknown is rejected with 404 so a
line is never orphaned from its camera (HLD 9).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session
from app.api.schemas.geometry import LineCreate, LineOut, LineUpdate
from app.db.models.geometry import Line
from app.db.repositories.camera_repo import CameraRepository
from app.db.repositories.line_repo import LineRepository

router = APIRouter(prefix="/lines", tags=["lines"], dependencies=[AuthDep])


def _to_columns(payload: LineCreate | LineUpdate) -> dict[str, Any]:
    """Build a column->value map from the set fields of a line payload.

    Only fields the caller actually provided are included so updates stay
    partial.
    """
    return payload.model_dump(exclude_unset=True)


def _to_out(line: Line) -> LineOut:
    """Map a :class:`Line` ORM row to its response model."""
    return LineOut.model_validate(line)


@router.get("", response_model=list[LineOut])
def list_lines(session: Session = Depends(db_session)) -> list[LineOut]:
    """Return every counting line ordered by id."""
    return [_to_out(line) for line in LineRepository(session).list()]


@router.post("", response_model=LineOut, status_code=status.HTTP_201_CREATED)
def create_line(payload: LineCreate, session: Session = Depends(db_session)) -> LineOut:
    """Create a counting line after verifying its parent camera exists."""
    if CameraRepository(session).get(payload.camera_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"camera {payload.camera_id} not found",
        )
    line = LineRepository(session).create(_to_columns(payload))
    return _to_out(line)


@router.get("/{line_id}", response_model=LineOut)
def get_line(line_id: int, session: Session = Depends(db_session)) -> LineOut:
    """Return one counting line by id, or 404 if it does not exist."""
    line = LineRepository(session).get(line_id)
    if line is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"line {line_id} not found"
        )
    return _to_out(line)


@router.patch("/{line_id}", response_model=LineOut)
def update_line(
    line_id: int, payload: LineUpdate, session: Session = Depends(db_session)
) -> LineOut:
    """Apply a partial update to a counting line, or 404 if it does not exist."""
    line = LineRepository(session).update(line_id, _to_columns(payload))
    if line is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"line {line_id} not found"
        )
    return _to_out(line)


@router.delete("/{line_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_line(line_id: int, session: Session = Depends(db_session)) -> None:
    """Delete a counting line, or 404 if it does not exist."""
    if not LineRepository(session).delete(line_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"line {line_id} not found"
        )
