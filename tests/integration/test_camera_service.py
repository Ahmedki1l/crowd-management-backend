"""Integration tests for CameraService credential handling (HLD 6.5, 5.8, 14).

These exercise the real ``CredentialCipher`` (keyed from the
``CAMERA_CREDENTIALS_KEY`` env set by ``_isolated_environment``) against the
per-test in-memory DB. The contract under test: the plaintext password is
encrypted before it reaches the row, never surfaces in registry response models,
and round-trips only through the engine or authenticated camera-server resolver.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.schemas.camera import (
    CameraCreate,
    CameraCredentialsOut,
    CameraCredentialsResolveOut,
    CameraUpdate,
)
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


def test_resolve_credentials_by_ip_returns_username_and_password(session: Session) -> None:
    service = CameraService(session)
    service.create_camera(_create_payload())

    credentials = service.resolve_credentials_by_ip("10.0.0.5")

    assert credentials.model_dump() == {
        "ip": "10.0.0.5",
        "username": "admin",
        "password": _PLAINTEXT,
    }


def test_credentials_resolve_response_hides_plaintext_in_validation_errors() -> None:
    plaintext = "must-not-appear-in-validation-error"

    with pytest.raises(ValidationError) as exc_info:
        CameraCredentialsResolveOut(
            ip=None,  # type: ignore[arg-type]
            username="admin",
            password=plaintext,
        )

    assert plaintext not in str(exc_info.value)


def test_internal_credentials_response_hides_plaintext_in_validation_errors() -> None:
    plaintext = "must-not-appear-in-validation-error"

    with pytest.raises(ValidationError) as exc_info:
        CameraCredentialsOut(password=plaintext)  # type: ignore[call-arg]

    assert plaintext not in str(exc_info.value)


def test_resolve_credentials_by_ip_rejects_duplicate_ip(session: Session) -> None:
    service = CameraService(session)
    service.create_camera(_create_payload(name="front-door"))
    service.create_camera(_create_payload(name="rear-door"))

    with pytest.raises(ValueError, match="multiple cameras"):
        service.resolve_credentials_by_ip("10.0.0.5")


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
