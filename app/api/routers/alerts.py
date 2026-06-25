"""Alert read/transition router (HLD 8.2).

Lists safety alerts filtered by status and/or type, and transitions a single
alert to ``acknowledged`` or ``resolved``. All lifecycle and persistence logic
lives in :class:`~app.services.alert_service.AlertService`; this router only
binds request parameters to that service and maps a missing alert to ``404``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session
from app.api.schemas.metrics import AlertOut
from app.services.alert_service import AlertService
from app.utils.clock import system_clock
from app.utils.timeutil import to_datetime

router = APIRouter(tags=["alerts"], dependencies=[AuthDep])


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(
    status: str | None = None,
    type: str | None = None,
    session: Session = Depends(db_session),
) -> list[AlertOut]:
    """List alerts (newest first) filtered by status and/or type.

    Args:
        status: Restrict to this lifecycle status when given.
        type: Restrict to this alert type when given.
        session: Injected DB session owning the read transaction.

    Returns:
        The matching alerts as response DTOs.
    """
    service = AlertService(session)
    return service.list_alerts(status=status, type=type)


@router.post("/alerts/{alert_id}/ack", response_model=AlertOut)
def ack_alert(
    alert_id: int,
    resolve: bool = False,
    session: Session = Depends(db_session),
) -> AlertOut:
    """Acknowledge or resolve a single alert.

    Args:
        alert_id: The alert id to transition.
        resolve: When ``True`` move straight to ``resolved``; otherwise mark it
            ``acknowledged``.
        session: Injected DB session owning the write transaction.

    Returns:
        The updated alert as a response DTO.

    Raises:
        HTTPException: ``404`` if no alert with ``alert_id`` exists.
    """
    service = AlertService(session)
    now = to_datetime(system_clock().now())
    updated = service.ack_alert(id=alert_id, resolve=resolve, now=now)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"alert {alert_id} not found",
        )
    return updated
