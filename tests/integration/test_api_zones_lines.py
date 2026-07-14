"""Integration tests for the zone and counting-line geometry routers (HLD 8.1).

A camera is created over HTTP, then the zone and line CRUD surfaces are exercised
end to end against the real repositories. Also covers the polygon-shape guard
(``< 3 points`` -> 422) enforced by the ``ZoneCreate`` schema validator.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _make_camera(client: TestClient, auth_headers: dict[str, str]) -> int:
    response = client.post(
        "/api/v1/cameras",
        json={
            "name": "geom-cam",
            "area": "lobby",
            "ip": "10.0.0.9",
            "username": "admin",
            "password": "pw",
            "roles": ["occupancy"],
        },
        headers=auth_headers,
    )
    assert response.status_code == 201
    return response.json()["id"]


def _zone_payload(camera_id: int, **over) -> dict:
    payload: dict = {
        "camera_id": camera_id,
        "name": "queue-zone",
        "type": "occupancy",
        "polygon": [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]],
        # Required: occupancy history is keyed by space.
        "dt_space_id": "b1-waiting-area",
    }
    payload.update(over)
    return payload


def _line_payload(camera_id: int, **over) -> dict:
    payload: dict = {
        "camera_id": camera_id,
        "name": "door-line",
        "points": [[10.0, 0.0], [10.0, 200.0]],
        "in_direction": [1.0, 0.0],
        "area_id": "lobby",
    }
    payload.update(over)
    return payload


# --- Zones -----------------------------------------------------------------
def test_create_zone_returns_201_with_polygon(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)

    response = client.post(
        "/api/v1/zones", json=_zone_payload(camera_id), headers=auth_headers
    )

    assert response.status_code == 201
    assert response.json()["polygon"] == [
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0],
    ]


def test_create_zone_with_two_point_polygon_returns_422(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)

    response = client.post(
        "/api/v1/zones",
        json=_zone_payload(camera_id, polygon=[[0.0, 0.0], [100.0, 0.0]]),
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_create_zone_without_a_space_returns_422(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """Occupancy history is keyed by ``dt_space_id``.

    A zone without one is counted live and then forgotten — it appears in no history at
    all. 31 of this deployment's 44 zones were created that way, and not one has a single
    row of history. The drawing tool warned about it in JavaScript, which is a dialog to
    click through, not a constraint; the API must be the constraint.
    """
    camera_id = _make_camera(client, auth_headers)
    payload = _zone_payload(camera_id)
    del payload["dt_space_id"]

    response = client.post("/api/v1/zones", json=payload, headers=auth_headers)

    assert response.status_code == 422


def test_create_zone_with_an_empty_space_returns_422(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """An empty string is not a space — it would orphan the zone just as effectively."""
    camera_id = _make_camera(client, auth_headers)

    response = client.post(
        "/api/v1/zones",
        json=_zone_payload(camera_id, dt_space_id=""),
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_patch_cannot_strip_a_zones_space(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """There is deliberately no way to orphan an existing zone from its history."""
    camera_id = _make_camera(client, auth_headers)
    zone_id = client.post(
        "/api/v1/zones", json=_zone_payload(camera_id), headers=auth_headers
    ).json()["id"]

    response = client.patch(
        f"/api/v1/zones/{zone_id}", json={"dt_space_id": ""}, headers=auth_headers
    )

    assert response.status_code == 422
    still_there = client.get(f"/api/v1/zones/{zone_id}", headers=auth_headers).json()
    assert still_there["dt_space_id"] == "b1-waiting-area"


def test_list_zones_returns_created_zone(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)
    client.post("/api/v1/zones", json=_zone_payload(camera_id), headers=auth_headers)

    response = client.get("/api/v1/zones", headers=auth_headers)

    assert [z["name"] for z in response.json()] == ["queue-zone"]


def test_get_zone_by_id_returns_zone(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)
    created = client.post(
        "/api/v1/zones", json=_zone_payload(camera_id), headers=auth_headers
    ).json()

    response = client.get(f"/api/v1/zones/{created['id']}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_patch_zone_updates_name(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)
    created = client.post(
        "/api/v1/zones", json=_zone_payload(camera_id), headers=auth_headers
    ).json()

    response = client.patch(
        f"/api/v1/zones/{created['id']}",
        json={"name": "renamed-zone"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["name"] == "renamed-zone"


def test_delete_zone_then_get_returns_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)
    created = client.post(
        "/api/v1/zones", json=_zone_payload(camera_id), headers=auth_headers
    ).json()

    deleted = client.delete(
        f"/api/v1/zones/{created['id']}", headers=auth_headers
    )
    follow_up = client.get(f"/api/v1/zones/{created['id']}", headers=auth_headers)

    assert deleted.status_code == 204
    assert follow_up.status_code == 404


def test_create_zone_for_unknown_camera_returns_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/api/v1/zones", json=_zone_payload(9999), headers=auth_headers
    )

    assert response.status_code == 404


# --- Lines -----------------------------------------------------------------
def test_create_line_returns_201_with_points(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)

    response = client.post(
        "/api/v1/lines", json=_line_payload(camera_id), headers=auth_headers
    )

    assert response.status_code == 201
    assert response.json()["points"] == [[10.0, 0.0], [10.0, 200.0]]


def test_list_lines_returns_created_line(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)
    client.post("/api/v1/lines", json=_line_payload(camera_id), headers=auth_headers)

    response = client.get("/api/v1/lines", headers=auth_headers)

    assert [line["name"] for line in response.json()] == ["door-line"]


def test_patch_line_updates_area_id(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)
    created = client.post(
        "/api/v1/lines", json=_line_payload(camera_id), headers=auth_headers
    ).json()

    response = client.patch(
        f"/api/v1/lines/{created['id']}",
        json={"area_id": "warehouse"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["area_id"] == "warehouse"


def test_delete_line_then_get_returns_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    camera_id = _make_camera(client, auth_headers)
    created = client.post(
        "/api/v1/lines", json=_line_payload(camera_id), headers=auth_headers
    ).json()

    deleted = client.delete(
        f"/api/v1/lines/{created['id']}", headers=auth_headers
    )
    follow_up = client.get(f"/api/v1/lines/{created['id']}", headers=auth_headers)

    assert deleted.status_code == 204
    assert follow_up.status_code == 404
