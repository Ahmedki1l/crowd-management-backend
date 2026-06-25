"""Event (de)serialization for the cross-process bus (HLD 6.5 / 13.1).

The in-process bus passes :class:`~app.events.events.Event` objects by reference;
the Redis-backed bus must serialize them to JSON and reconstruct the exact
subclass on the other side. This module is the single source of truth for that
wire form, kept generic over the event dataclasses so a new event type only needs
a registry entry.
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any

from app.domain.models import AlertType, CrossingDirection
from app.events.events import (
    AlertRaised,
    CameraHealth,
    CountUpdate,
    CrossingEvent,
    DwellClosed,
    Event,
    EventType,
    HeatmapFlushed,
    OccupancyUpdate,
    WaitingUpdate,
)

# Maps an EventType value to its concrete Event subclass for reconstruction.
_REGISTRY: dict[str, type[Event]] = {
    EventType.OCCUPANCY_UPDATE.value: OccupancyUpdate,
    EventType.COUNT_UPDATE.value: CountUpdate,
    EventType.WAITING_UPDATE.value: WaitingUpdate,
    EventType.ALERT.value: AlertRaised,
    EventType.CROSSING.value: CrossingEvent,
    EventType.DWELL_CLOSED.value: DwellClosed,
    EventType.HEATMAP_FLUSH.value: HeatmapFlushed,
    EventType.CAMERA_HEALTH.value: CameraHealth,
}

# Event fields whose Python type is an Enum, with the enum to rebuild them from.
_ENUM_FIELDS: dict[str, type[Enum]] = {
    "direction": CrossingDirection,
    "alert_type": AlertType,
}

_TYPE_KEY = "__type__"


def event_to_dict(event: Event) -> dict[str, Any]:
    """Convert an event to a JSON-safe dict (enums flattened to their values)."""
    data: dict[str, Any] = {}
    for field in dataclasses.fields(event):
        # ``type`` is init=False on every subclass — it is re-derived on rebuild.
        if field.name == "type":
            continue
        value = getattr(event, field.name)
        data[field.name] = value.value if isinstance(value, Enum) else value
    data[_TYPE_KEY] = event.type.value
    return data


def event_from_dict(data: dict[str, Any]) -> Event:
    """Reconstruct the concrete event subclass from :func:`event_to_dict` output."""
    type_value = data.get(_TYPE_KEY)
    event_cls = _REGISTRY.get(type_value)
    if event_cls is None:
        raise ValueError(f"unknown event type {type_value!r}")
    kwargs = {key: value for key, value in data.items() if key != _TYPE_KEY}
    for name, enum_cls in _ENUM_FIELDS.items():
        if kwargs.get(name) is not None:
            kwargs[name] = enum_cls(kwargs[name])
    return event_cls(**kwargs)


def serialize(event: Event) -> str:
    """Serialize an event to a JSON string for transport."""
    return json.dumps(event_to_dict(event))


def deserialize(raw: str | bytes) -> Event:
    """Reconstruct an event from a JSON string/bytes produced by :func:`serialize`."""
    return event_from_dict(json.loads(raw))
