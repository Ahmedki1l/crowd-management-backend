"""Per-camera detector imgsz override flows from the DB into the CameraSpec."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.services.camera_service import CameraService


def test_camera_spec_carries_imgsz(session: Session, make_camera) -> None:
    cam = make_camera(session, name="hi-res", imgsz=1280)

    spec = CameraService(session).build_camera_spec(cam.id)

    assert spec is not None
    assert spec.imgsz == 1280


def test_camera_spec_imgsz_defaults_to_none(session: Session, make_camera) -> None:
    cam = make_camera(session, name="default-res")  # no imgsz -> use global

    spec = CameraService(session).build_camera_spec(cam.id)

    assert spec is not None
    assert spec.imgsz is None


def test_camera_api_roundtrips_imgsz(client, auth_headers) -> None:
    payload = {
        "name": "gate", "area": "Entrance", "ip": "10.0.0.7", "username": "u",
        "password": "p", "roles": ["entry_exit"], "imgsz": 1280,
    }
    created = client.post("/api/v1/cameras", json=payload, headers=auth_headers)
    assert created.status_code == 201
    assert created.json()["imgsz"] == 1280

    fetched = client.get(f"/api/v1/cameras/{created.json()['id']}", headers=auth_headers)
    assert fetched.json()["imgsz"] == 1280
