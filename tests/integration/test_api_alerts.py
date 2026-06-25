"""Integration tests for the alert read/transition router (HLD 8.2).

Alerts are inserted directly through ``AlertRepository`` (committing, since
repositories only flush) so the route's own request-scoped session sees them.
Covers status/type filtering on the list endpoint and the acknowledge
transition, asserting the persisted status changes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.repositories.alert_repo import AlertRepository


def _insert_alert(
    session: Session,
    *,
    type: str = "intrusion",
    zone_id: int | None = 1,
    detail: str = "person in restricted zone",
    status: str = "active",
) -> int:
    repo = AlertRepository(session)
    alert = repo.create(
        type=type,
        zone_id=zone_id,
        camera_id=None,
        ts=datetime(2024, 6, 12, 10, 0, tzinfo=UTC),
        detail=detail,
        snapshot_url=None,
        status=status,
    )
    session.commit()
    return alert.id


def test_list_alerts_returns_inserted_alert(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    alert_id = _insert_alert(session)

    response = client.get("/api/v1/alerts", headers=auth_headers)

    assert response.status_code == 200
    assert [a["id"] for a in response.json()] == [alert_id]


def test_list_alerts_filters_by_status(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    active_id = _insert_alert(session, status="active")
    _insert_alert(session, status="resolved")

    response = client.get(
        "/api/v1/alerts", params={"status": "active"}, headers=auth_headers
    )

    assert [a["id"] for a in response.json()] == [active_id]


def test_list_alerts_filters_by_type(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    intrusion_id = _insert_alert(session, type="intrusion")
    _insert_alert(session, type="overcrowding")

    response = client.get(
        "/api/v1/alerts", params={"type": "intrusion"}, headers=auth_headers
    )

    assert [a["id"] for a in response.json()] == [intrusion_id]


def test_ack_alert_sets_status_acknowledged(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    alert_id = _insert_alert(session, status="active")

    response = client.post(
        f"/api/v1/alerts/{alert_id}/ack", headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["status"] == "acknowledged"


def test_ack_alert_with_resolve_sets_status_resolved(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    alert_id = _insert_alert(session, status="active")

    response = client.post(
        f"/api/v1/alerts/{alert_id}/ack",
        params={"resolve": True},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"


def test_ack_unknown_alert_returns_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post("/api/v1/alerts/9999/ack", headers=auth_headers)

    assert response.status_code == 404
