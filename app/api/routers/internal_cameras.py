"""Internal camera credential contract consumed by the HLS camera relay."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import CrowdCameraInternalAuthDep, db_session
from app.api.schemas.camera import CameraCredentialsOut
from app.services.camera_service import CameraService

router = APIRouter(
    prefix="/internal/cameras",
    tags=["internal-cameras"],
    dependencies=[CrowdCameraInternalAuthDep],
)


def _get_service(session: Session = Depends(db_session)) -> CameraService:
    return CameraService(session)


@router.get("/by-ip/{camera_ip}/credentials", response_model=CameraCredentialsOut)
def get_camera_credentials_by_ip(
    camera_ip: str,
    service: CameraService = Depends(_get_service),
) -> CameraCredentialsOut:
    """Return one enabled camera's credentials for an exact IP match."""
    matches = service.list_enabled_by_ip(camera_ip)
    if not matches:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="camera not found",
        )
    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="multiple enabled cameras share this IP",
        )
    try:
        return service.to_credentials_out(matches[0])
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="camera credential decryption is not configured",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="camera credential is unavailable",
        ) from exc
