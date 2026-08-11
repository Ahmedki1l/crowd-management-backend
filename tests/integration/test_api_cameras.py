"""Integration tests for the camera registry router (HLD 8.1).

Exercises the full HTTP surface through the ``TestClient`` + ``auth_headers``
fixtures: auth enforcement, the write-only-password contract on create, the
read paths, partial update, and the delete -> 404 lifecycle. The real
``CameraService`` and DB are used end to end — no service mocks.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models.camera import Camera

from app.config.settings import reset_settings_cache


def _create_payload(**over) -> dict:
    payload: dict = {
        "name": "lobby-cam",
        "area": "lobby",
        "ip": "10.0.0.7",
        "port": 554,
        "username": "admin",
        "password": "s3cret",
        "roles": ["occupancy"],
    }
    payload.update(over)
    return payload


def test_list_cameras_without_auth_returns_401(client: TestClient) -> None:
    response = client.get("/api/v1/cameras")

    assert response.status_code == 401


def test_resolve_credentials_without_auth_returns_401(client: TestClient) -> None:
    response = client.post(
        "/api/v1/cameras/credentials/resolve", json={"ip": "10.0.0.7"}
    )

    assert response.status_code == 401


def test_create_camera_returns_201(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/cameras", json=_create_payload(), headers=auth_headers
    )

    assert response.status_code == 201


def test_create_camera_response_omits_password(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/cameras", json=_create_payload(), headers=auth_headers
    )

    assert "password" not in response.json()


def test_create_camera_with_password_sets_has_password_true(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/cameras", json=_create_payload(), headers=auth_headers
    )

    assert response.json()["has_password"] is True


def test_create_camera_echoes_submitted_fields(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/cameras",
        json=_create_payload(name="dock-cam", area="dock"),
        headers=auth_headers,
    )

    body = response.json()
    assert (body["name"], body["area"], body["username"]) == (
        "dock-cam",
        "dock",
        "admin",
    )


def test_list_cameras_returns_created_camera(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    client.post("/api/v1/cameras", json=_create_payload(), headers=auth_headers)

    response = client.get("/api/v1/cameras", headers=auth_headers)

    assert response.status_code == 200
    assert [c["name"] for c in response.json()] == ["lobby-cam"]
    assert "password" not in response.json()[0]


def test_resolve_credentials_by_ip_returns_secret_without_caching(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    client.post("/api/v1/cameras", json=_create_payload(), headers=auth_headers)

    response = client.post(
        "/api/v1/cameras/credentials/resolve",
        json={"ip": "10.0.0.7"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "ip": "10.0.0.7",
        "username": "admin",
        "password": "s3cret",
    }
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"


def test_resolve_credentials_for_unknown_ip_returns_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/cameras/credentials/resolve",
        json={"ip": "10.0.0.99"},
        headers=auth_headers,
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "camera with IP 10.0.0.99 not found"}
    assert response.headers["cache-control"] == "no-store"


def test_resolve_credentials_rejects_blank_ip(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/cameras/credentials/resolve",
        json={"ip": "   "},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_resolve_credentials_rejects_duplicate_ip(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    client.post(
        "/api/v1/cameras",
        json=_create_payload(name="lobby-cam-a"),
        headers=auth_headers,
    )
    client.post(
        "/api/v1/cameras",
        json=_create_payload(name="lobby-cam-b"),
        headers=auth_headers,
    )

    response = client.post(
        "/api/v1/cameras/credentials/resolve",
        json={"ip": "10.0.0.7"},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "camera credentials are unavailable"}


def test_resolve_credentials_without_stored_password_returns_409(
    client: TestClient,
    auth_headers: dict[str, str],
    session: Session,
) -> None:
    session.add(
        Camera(
            name="passwordless-camera",
            area="lobby",
            ip="10.0.0.8",
            port=554,
            username="admin",
            roles=["occupancy"],
            enabled=True,
        )
    )
    session.commit()

    response = client.post(
        "/api/v1/cameras/credentials/resolve",
        json={"ip": "10.0.0.8"},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "camera credentials are unavailable"}
    assert response.headers["cache-control"] == "no-store"


def test_get_camera_by_id_returns_camera(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = client.post(
        "/api/v1/cameras", json=_create_payload(), headers=auth_headers
    ).json()

    response = client.get(f"/api/v1/cameras/{created['id']}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_unknown_camera_returns_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get("/api/v1/cameras/9999", headers=auth_headers)

    assert response.status_code == 404


def test_patch_camera_updates_field(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = client.post(
        "/api/v1/cameras", json=_create_payload(), headers=auth_headers
    ).json()

    response = client.patch(
        f"/api/v1/cameras/{created['id']}",
        json={"area": "warehouse"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["area"] == "warehouse"


def test_delete_camera_returns_204(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = client.post(
        "/api/v1/cameras", json=_create_payload(), headers=auth_headers
    ).json()

    response = client.delete(
        f"/api/v1/cameras/{created['id']}", headers=auth_headers
    )

    assert response.status_code == 204


def test_get_deleted_camera_returns_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    created = client.post(
        "/api/v1/cameras", json=_create_payload(), headers=auth_headers
    ).json()
    client.delete(f"/api/v1/cameras/{created['id']}", headers=auth_headers)

    response = client.get(f"/api/v1/cameras/{created['id']}", headers=auth_headers)

    assert response.status_code == 404


def test_internal_credentials_require_dedicated_token(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get(
        "/api/v1/internal/cameras/by-ip/10.0.0.7/credentials",
        headers=auth_headers,
    )

    assert response.status_code == 401


def test_internal_credentials_return_decrypted_camera_config(
    client: TestClient,
    auth_headers: dict[str, str],
    crowd_camera_internal_headers: dict[str, str],
) -> None:
    client.post("/api/v1/cameras", json=_create_payload(), headers=auth_headers)

    response = client.get(
        "/api/v1/internal/cameras/by-ip/10.0.0.7/credentials",
        headers=crowd_camera_internal_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "lobby-cam"
    assert body["username"] == "admin"
    assert body["password"] == "s3cret"
    assert body["stream_channel_main"] == 101
    assert body["stream_channel_sub"] == 102


def test_internal_credentials_return_404_for_unknown_ip(
    client: TestClient,
    crowd_camera_internal_headers: dict[str, str],
) -> None:
    response = client.get(
        "/api/v1/internal/cameras/by-ip/10.0.0.99/credentials",
        headers=crowd_camera_internal_headers,
    )

    assert response.status_code == 404


def test_internal_credentials_ignore_disabled_camera(
    client: TestClient,
    auth_headers: dict[str, str],
    crowd_camera_internal_headers: dict[str, str],
) -> None:
    client.post(
        "/api/v1/cameras",
        json=_create_payload(enabled=False),
        headers=auth_headers,
    )

    response = client.get(
        "/api/v1/internal/cameras/by-ip/10.0.0.7/credentials",
        headers=crowd_camera_internal_headers,
    )

    assert response.status_code == 404


def test_internal_credentials_return_503_without_decryption_key(
    client: TestClient,
    auth_headers: dict[str, str],
    crowd_camera_internal_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client.post("/api/v1/cameras", json=_create_payload(), headers=auth_headers)
    monkeypatch.delenv("CAMERA_CREDENTIALS_KEY")
    reset_settings_cache()

    response = client.get(
        "/api/v1/internal/cameras/by-ip/10.0.0.7/credentials",
        headers=crowd_camera_internal_headers,
    )

    assert response.status_code == 503


def test_internal_credentials_reject_ambiguous_ip(
    client: TestClient,
    auth_headers: dict[str, str],
    crowd_camera_internal_headers: dict[str, str],
) -> None:
    client.post("/api/v1/cameras", json=_create_payload(), headers=auth_headers)
    client.post(
        "/api/v1/cameras",
        json=_create_payload(name="second-lobby-cam"),
        headers=auth_headers,
    )

    response = client.get(
        "/api/v1/internal/cameras/by-ip/10.0.0.7/credentials",
        headers=crowd_camera_internal_headers,
    )

    assert response.status_code == 409
