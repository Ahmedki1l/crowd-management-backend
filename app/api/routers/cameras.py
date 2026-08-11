"""Camera registry router (HLD 8.1).

CRUD plus a connectivity probe over the camera registry. All persistence and
credential handling is delegated to :class:`~app.services.camera_service.CameraService`;
this router only translates HTTP <-> service calls and maps rows to ``CameraOut``.

The RTSP password is write-only in registry responses: it is accepted on
create/update but never returned by a GET — ``CameraOut`` carries only
``has_password`` (HLD 8.1 / 14). A dedicated authenticated POST resolves one
camera's credentials by IP for the camera server.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session
from app.api.schemas.camera import (
    CameraCreate,
    CameraCredentialsByIp,
    CameraCredentialsResolveOut,
    CameraOut,
    CameraTestResult,
    CameraUpdate,
)
from app.services.camera_service import CameraService

router = APIRouter(prefix="/cameras", tags=["cameras"], dependencies=[AuthDep])

_NO_STORE_HEADERS = {"Cache-Control": "no-store", "Pragma": "no-cache"}


def _get_service(session: Session = Depends(db_session)) -> CameraService:
    """Provide a :class:`CameraService` bound to the request-scoped session."""
    return CameraService(session)


@router.get("", response_model=list[CameraOut])
def list_cameras(service: CameraService = Depends(_get_service)) -> list[CameraOut]:
    """Return every registered camera (passwords withheld)."""
    return [service.to_out(camera) for camera in service.list()]


@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(
    payload: CameraCreate, service: CameraService = Depends(_get_service)
) -> CameraOut:
    """Create a camera, encrypting its password before it is persisted."""
    camera = service.create_camera(payload)
    return service.to_out(camera)


@router.post("/credentials/resolve", response_model=CameraCredentialsResolveOut)
def resolve_camera_credentials(
    payload: CameraCredentialsByIp,
    response: Response,
    service: CameraService = Depends(_get_service),
) -> CameraCredentialsResolveOut:
    """Return one camera's credentials by IP for an authenticated camera server.

    The response contains plaintext credentials, so intermediaries must not
    cache it. Callers must also use TLS; bearer authentication is enforced by
    the router-level dependency.
    """
    response.headers.update(_NO_STORE_HEADERS)
    try:
        return service.resolve_credentials_by_ip(payload.ip)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"camera with IP {payload.ip} not found",
            headers=_NO_STORE_HEADERS,
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="camera credentials are unavailable",
            headers=_NO_STORE_HEADERS,
        ) from exc


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(
    camera_id: int, service: CameraService = Depends(_get_service)
) -> CameraOut:
    """Return one camera by id, or 404 if it does not exist."""
    camera = service.get(camera_id)
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"camera {camera_id} not found"
        )
    return service.to_out(camera)


@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(
    camera_id: int,
    payload: CameraUpdate,
    service: CameraService = Depends(_get_service),
) -> CameraOut:
    """Apply a partial update; a non-null password rotates the stored credential."""
    camera = service.update_camera(camera_id, payload)
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"camera {camera_id} not found"
        )
    return service.to_out(camera)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(
    camera_id: int, service: CameraService = Depends(_get_service)
) -> None:
    """Delete a camera (cascading to its zones/lines), or 404 if it is missing."""
    if not service.delete_camera(camera_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"camera {camera_id} not found"
        )


@router.post("/{camera_id}/test", response_model=CameraTestResult)
def test_camera(
    camera_id: int, service: CameraService = Depends(_get_service)
) -> CameraTestResult:
    """Probe a camera's sub stream and report reachability and stream metadata.

    A missing camera yields 404; a reachable-but-failing probe is reported in the
    body with ``reachable=False`` rather than as an HTTP error (the service never
    raises for connectivity faults).
    """
    if service.get(camera_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"camera {camera_id} not found"
        )
    return service.test_camera(camera_id)
