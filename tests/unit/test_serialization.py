"""Unit tests for the cross-process event wire form (app.events.serialization).

These cover the JSON round-trip the Redis bus relies on: every event subclass
must serialize and reconstruct to an equal instance, enum fields must survive the
trip, and an unknown type must fail loudly rather than silently produce a wrong
object.
"""

from __future__ import annotations

import pytest

from app.domain.models import AlertType, CrossingDirection
from app.events.events import (
    AlertRaised,
    CameraHealth,
    CountUpdate,
    CrossingEvent,
    DwellClosed,
    Event,
    HeatmapFlushed,
    OccupancyUpdate,
    WaitingUpdate,
)
from app.events.serialization import deserialize, event_from_dict, serialize

# One instance of every Event subclass (HLD 6.5). The bus reconstructs the exact
# subclass on the far side, so each must round-trip to an *equal* object — this
# also exercises the two enum-valued fields (direction, alert_type).
_EVENTS: list[Event] = [
    OccupancyUpdate(ts=1000.0, zone_id=5, camera_id=2, count=3, dt_space_id="space-a"),
    CrossingEvent(
        ts=1000.0, line_id=7, area_id="area-1", direction=CrossingDirection.OUT, track_ref=3
    ),
    CountUpdate(ts=1000.0, area_id="area-1", in_count=10, out_count=4, net=6, line_id=7),
    WaitingUpdate(ts=1000.0, zone_id=9, current_waits=2, avg_dwell_s=12.5),
    DwellClosed(ts=1000.0, zone_id=9, track_ref=3, enter_ts=900.0, leave_ts=950.0, dwell_s=50.0),
    AlertRaised(
        ts=1000.0, alert_type=AlertType.OVERCROWDING, zone_id=5, camera_id=2, detail="full"
    ),
    HeatmapFlushed(ts=1000.0, camera_id=4, ts_bucket=960.0),
    CameraHealth(ts=1000.0, camera_id=4, fps=15.0, queue_depth=1, healthy=False),
]


@pytest.mark.parametrize("event", _EVENTS, ids=lambda e: e.type.value)
def test_event_survives_json_roundtrip(event: Event) -> None:
    assert deserialize(serialize(event)) == event


def test_crossing_event_enum_direction_roundtrips() -> None:
    event = CrossingEvent(
        ts=1000.0, line_id=7, area_id="area-1", direction=CrossingDirection.OUT, track_ref=3
    )

    restored = deserialize(serialize(event))

    assert isinstance(restored, CrossingEvent)
    assert restored.direction is CrossingDirection.OUT


def test_alert_event_enum_alert_type_roundtrips() -> None:
    event = AlertRaised(
        ts=1000.0, alert_type=AlertType.OVERCROWDING, zone_id=5, camera_id=2, detail="full"
    )

    restored = deserialize(serialize(event))

    assert isinstance(restored, AlertRaised)
    assert restored.alert_type is AlertType.OVERCROWDING


def test_unknown_event_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown event type"):
        event_from_dict({"__type__": "not_a_real_event", "ts": 1000.0})
