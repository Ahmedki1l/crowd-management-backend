"""Safety-critical end-to-end journey through the alert API (e2e-guard).

This is the one critical-path journey an operator depends on: a restricted-zone
intrusion is detected by the real :class:`~app.analytics.safety.SafetyCalculator`,
persisted through the real :class:`~app.services.alert_service.AlertService`, then
observed and transitioned across the public ``/api/v1`` surface. The flow is
driven end to end (calculator -> service -> HTTP), asserting both the HTTP status
and the response body at every hop:

1. the safety calculator raises an intrusion ``AlertRaised`` and the service
   persists it (the alert's true origin, not a hand-built row);
2. ``GET /alerts?status=active`` surfaces it with its type, zone, and timestamp;
3. ``GET /alerts?type=intrusion`` filters it in while an unrelated overcrowding
   alert is filtered out;
4. ``POST /alerts/{id}/ack`` moves it ``active -> acknowledged`` (resolve=false)
   then ``acknowledged -> resolved`` (resolve=true);
5. ``GET /history/alerts?zone_id=`` returns it in the zone's history.

A :class:`~app.utils.clock.FakeClock` is the only time source, the in-memory DB
and ``client``/``auth_headers`` fixtures come from ``conftest``, and the test
owns all of its data — no real cameras, models, or wall-clock sleeps.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.analytics.safety import SafetyCalculator
from app.domain.models import AlertType, ZoneType
from app.events.events import AlertRaised
from app.services.alert_service import AlertService
from app.utils.clock import FakeClock
from tests.fixtures.specs import make_zone_spec

_RESTRICTED_ZONE_ID = 1
_OCCUPANCY_ZONE_ID = 2
_CAMERA_ID = 7
_INTRUSION_TS = 1000.0


def _detect_intrusion(zone_id: int, camera_id: int, ts: float) -> AlertRaised:
    """Run the real safety calculator until a restricted zone raises intrusion.

    Drives a single confirmed member through the debounce window so the emitted
    event is the calculator's genuine output, not a hand-built one.
    """
    zone = make_zone_spec(
        id=zone_id, camera_id=camera_id, name="vault", type=ZoneType.RESTRICTED
    )
    calc = SafetyCalculator(
        camera_id=camera_id,
        zones=[zone],
        clock=FakeClock(start=ts),
        debounce_frames=2,
        cooldown_seconds=60.0,
    )
    calc.process({zone_id: {99}}, {zone_id: 1}, ts=ts)
    fired = calc.process({zone_id: {99}}, {zone_id: 1}, ts=ts)
    (event,) = fired
    return event


def _detect_overcrowding(zone_id: int, camera_id: int, ts: float) -> AlertRaised:
    """Run the real safety calculator until a zone raises overcrowding."""
    zone = make_zone_spec(
        id=zone_id,
        camera_id=camera_id,
        name="lobby",
        type=ZoneType.OCCUPANCY,
        safe_limit=2,
    )
    calc = SafetyCalculator(
        camera_id=camera_id,
        zones=[zone],
        clock=FakeClock(start=ts),
        debounce_frames=1,
        cooldown_seconds=60.0,
    )
    fired = calc.process({zone_id: {1, 2, 3}}, {zone_id: 3}, ts=ts)
    (event,) = fired
    return event


def _persist(session: Session, event: AlertRaised) -> int:
    """Persist a raised alert via the real service and return its id."""
    service = AlertService(session)
    alert = service.raise_alert(event, snapshot_url=None)
    session.commit()
    return alert.id


def test_intrusion_journey_from_detection_to_history(
    client: TestClient, auth_headers: dict[str, str], session: Session
) -> None:
    # 1. The calculator detects the intrusion and the service persists it.
    event = _detect_intrusion(_RESTRICTED_ZONE_ID, _CAMERA_ID, _INTRUSION_TS)
    assert event.alert_type is AlertType.INTRUSION
    alert_id = _persist(session, event)

    # An unrelated overcrowding alert in another zone exists alongside it, so the
    # type filter has something to exclude.
    other = _detect_overcrowding(_OCCUPANCY_ZONE_ID, _CAMERA_ID, _INTRUSION_TS)
    _persist(session, other)

    # 2. It appears in the active list with type, zone, and timestamp.
    active = client.get(
        "/api/v1/alerts", params={"status": "active"}, headers=auth_headers
    )
    assert active.status_code == 200
    bodies = {a["id"]: a for a in active.json()}
    assert alert_id in bodies
    surfaced = bodies[alert_id]
    assert surfaced["type"] == AlertType.INTRUSION.value
    assert surfaced["zone_id"] == _RESTRICTED_ZONE_ID
    assert surfaced["ts"].startswith("1970-01-01T00:16:40")  # ts=1000.0 epoch

    # 3. The type filter keeps the intrusion and drops the overcrowding alert.
    filtered = client.get(
        "/api/v1/alerts", params={"type": "intrusion"}, headers=auth_headers
    )
    assert filtered.status_code == 200
    assert [a["id"] for a in filtered.json()] == [alert_id]

    # 4a. Acknowledge (resolve=false) moves active -> acknowledged.
    acked = client.post(f"/api/v1/alerts/{alert_id}/ack", headers=auth_headers)
    assert acked.status_code == 200
    assert acked.json()["status"] == "acknowledged"

    # 4b. Acknowledge again with resolve=true moves it -> resolved.
    resolved = client.post(
        f"/api/v1/alerts/{alert_id}/ack",
        params={"resolve": True},
        headers=auth_headers,
    )
    assert resolved.status_code == 200
    assert resolved.json()["status"] == "resolved"

    # It is no longer in the active list once resolved.
    still_active = client.get(
        "/api/v1/alerts", params={"status": "active"}, headers=auth_headers
    )
    assert alert_id not in [a["id"] for a in still_active.json()]

    # 5. The zone's alert history still returns it (history ignores status).
    history = client.get(
        "/api/v1/history/alerts",
        params={"zone_id": _RESTRICTED_ZONE_ID},
        headers=auth_headers,
    )
    assert history.status_code == 200
    history_bodies = {a["id"]: a for a in history.json()}
    assert alert_id in history_bodies
    assert history_bodies[alert_id]["status"] == "resolved"
