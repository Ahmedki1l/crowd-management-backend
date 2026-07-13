"""Integration tests for the live-metric read routers (HLD 8.2).

The in-memory ``StateStore`` singleton is seeded directly (it is the same
instance the routes read through ``get_state_store``), then the occupancy,
entry-exit, consolidated state, and stats endpoints are queried over
HTTP and asserted to reflect the seeded values.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.state_store import get_state_store


def test_occupancy_endpoint_reflects_seeded_count(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    get_state_store().set_occupancy(zone_id=7, count=5, dt_space_id="space-7", ts=1000.0)

    response = client.get("/api/v1/occupancy", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == [
        {"zone_id": 7, "dt_space_id": "space-7", "count": 5, "ts": 1000.0}
    ]


def test_entry_exit_endpoint_reflects_seeded_counts(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    get_state_store().set_counts(
        area_id="lobby",
        in_count=12,
        out_count=4,
        net=8,
        ts=1000.0,
        lines={3: (12, 4)},
    )

    response = client.get("/api/v1/entry-exit", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert (body[0]["area_id"], body[0]["in_count"], body[0]["out_count"], body[0]["net"]) == (
        "lobby",
        12,
        4,
        8,
    )


def test_entry_exit_endpoint_includes_per_line_breakdown(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    get_state_store().set_counts(
        area_id="lobby",
        in_count=12,
        out_count=4,
        net=8,
        ts=1000.0,
        lines={3: (12, 4)},
    )

    response = client.get("/api/v1/entry-exit", headers=auth_headers)

    assert response.json()[0]["lines"] == [
        {"line_id": 3, "in_count": 12, "out_count": 4}
    ]




def test_state_endpoint_aggregates_occupancy_and_entry_exit(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    store = get_state_store()
    store.set_occupancy(zone_id=7, count=5, dt_space_id=None, ts=1000.0)
    store.set_counts(area_id="lobby", in_count=12, out_count=4, net=8, ts=1000.0)
    response = client.get("/api/v1/state", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert (
        body["occupancy"][0]["count"],
        body["entry_exit"][0]["net"],
    ) == (5, 8)


def test_stats_endpoint_reflects_seeded_total_occupancy(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    store = get_state_store()
    store.set_occupancy(zone_id=1, count=5, dt_space_id=None, ts=1000.0)
    store.set_occupancy(zone_id=2, count=6, dt_space_id=None, ts=1000.0)

    response = client.get("/api/v1/stats", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["total_occupancy"] == 11
