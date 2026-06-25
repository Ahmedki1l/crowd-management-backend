"""Integration tests for CameraService credential handling (HLD 6.5, 5.8, 14).

These exercise the real ``CredentialCipher`` (keyed from the
``CAMERA_CREDENTIALS_KEY`` env set by ``_isolated_environment``) against the
per-test in-memory DB. The contract under test: the plaintext password is
encrypted before it reaches the row, never surfaces in the response model, and
round-trips back only through ``resolve_password``.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.schemas.camera import CameraCreate, CameraUpdate
from app.domain.models import CameraRole
from app.services.camera_service import CameraService

_PLAINTEXT = "s3cr3t-rtsp-pw"


def _create_payload(password: str = _PLAINTEXT, **over) -> CameraCreate:
    data: dict = {
        "name": "front-door",
        "area": "lobby",
        "ip": "10.0.0.5",
        "port": 554,
        "username": "admin",
        "password": password,
        "roles": [CameraRole.OCCUPANCY],
    }
    data.update(over)
    return CameraCreate(**data)


def test_create_camera_stores_encrypted_password_not_plaintext(session: Session) -> None:
    camera = CameraService(session).create_camera(_create_payload())

    assert camera.password_encrypted is not None
    assert camera.password_encrypted != _PLAINTEXT.encode("utf-8")
    assert _PLAINTEXT.encode("utf-8") not in camera.password_encrypted


def test_to_out_never_exposes_password_and_flags_has_password(session: Session) -> None:
    service = CameraService(session)
    camera = service.create_camera(_create_payload())

    out = service.to_out(camera)

    assert out.has_password is True
    # The response schema has no password field at all; the credential cannot leak.
    assert "password" not in out.model_dump()
    assert "password_encrypted" not in out.model_dump()


def test_resolve_password_decrypts_back_to_original_plaintext(session: Session) -> None:
    service = CameraService(session)
    camera = service.create_camera(_create_payload())

    assert service.resolve_password(camera.id) == _PLAINTEXT


def test_update_camera_with_new_password_rotates_the_stored_credential(
    session: Session,
) -> None:
    service = CameraService(session)
    camera = service.create_camera(_create_payload())
    original_cipher = camera.password_encrypted

    service.update_camera(camera.id, CameraUpdate(password="rotated-pw"))

    assert camera.password_encrypted != original_cipher
    assert service.resolve_password(camera.id) == "rotated-pw"


def test_update_camera_without_password_keeps_existing_credential(session: Session) -> None:
    service = CameraService(session)
    camera = service.create_camera(_create_payload())
    original_cipher = camera.password_encrypted

    service.update_camera(camera.id, CameraUpdate(name="renamed-cam"))

    assert camera.name == "renamed-cam"
    assert camera.password_encrypted == original_cipher
    assert service.resolve_password(camera.id) == _PLAINTEXT
