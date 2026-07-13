"""Event/message schema and bus contracts.

A single internal event bus drives both delivery mechanisms to the Digital Twin
(HLD 10): SSE subscription and outbound push. Every analytics result is emitted
as one of these immutable events; ``payload()`` produces the JSON object sent on
the wire, and ``sse_event`` is the SSE event name (HLD 8.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from app.domain.models import CrossingDirection


class EventType(str, Enum):
    OCCUPANCY_UPDATE = "occupancy_update"
    COUNT_UPDATE = "count_update"
    CROSSING = "crossing"
    CAMERA_HEALTH = "camera_health"


# NOTE: deliberately NOT slots=True. The subclasses call zero-argument
# ``super().payload()``; with ``slots=True`` the dataclass machinery recreates
# the class and breaks the ``__class__`` cell that bare ``super()`` relies on,
# raising TypeError on Python 3.11 (fixed in CPython 3.12). The service targets
# 3.11+, so slots stays off here.
@dataclass(frozen=True)
class Event:
    """Base event. Subclasses set ``type`` and implement ``payload()``."""

    type: EventType
    ts: float

    @property
    def sse_event(self) -> str:
        return self.type.value

    @property
    def topic(self) -> str:
        """Coarse routing key (``"area:<id>"`` / ``"zone:<id>"``) for subscriptions."""
        return "all"

    def payload(self) -> dict[str, Any]:
        return {"type": self.type.value, "ts": self.ts}


@dataclass(frozen=True)
class OccupancyUpdate(Event):
    zone_id: int = 0
    camera_id: int = 0
    count: int = 0
    dt_space_id: str | None = None
    type: EventType = field(default=EventType.OCCUPANCY_UPDATE, init=False)

    @property
    def topic(self) -> str:
        return f"zone:{self.zone_id}"

    def payload(self) -> dict[str, Any]:
        return {
            **super().payload(),
            "zone_id": self.zone_id,
            "camera_id": self.camera_id,
            "count": self.count,
            "dt_space_id": self.dt_space_id,
        }


@dataclass(frozen=True)
class CrossingEvent(Event):
    line_id: int = 0
    area_id: str = ""
    direction: CrossingDirection = CrossingDirection.IN
    track_ref: int = 0
    dt_space_id: str | None = None
    type: EventType = field(default=EventType.CROSSING, init=False)

    @property
    def topic(self) -> str:
        return f"area:{self.area_id}"

    def payload(self) -> dict[str, Any]:
        return {
            **super().payload(),
            "line_id": self.line_id,
            "area_id": self.area_id,
            "direction": self.direction.value,
            "track_ref": self.track_ref,
            "dt_space_id": self.dt_space_id,
        }


@dataclass(frozen=True)
class CountUpdate(Event):
    """Per-area IN/OUT/net (entry-exit). Emitted after each crossing."""

    area_id: str = ""
    in_count: int = 0
    out_count: int = 0
    net: int = 0
    line_id: int | None = None
    dt_space_id: str | None = None
    type: EventType = field(default=EventType.COUNT_UPDATE, init=False)

    @property
    def topic(self) -> str:
        return f"area:{self.area_id}"

    def payload(self) -> dict[str, Any]:
        return {
            **super().payload(),
            "area_id": self.area_id,
            "in": self.in_count,
            "out": self.out_count,
            "net": self.net,
            "line_id": self.line_id,
            "dt_space_id": self.dt_space_id,
        }


@dataclass(frozen=True)
class CameraHealth(Event):
    camera_id: int = 0
    fps: float = 0.0
    last_frame_age_s: float = 0.0
    queue_depth: int = 0
    healthy: bool = True
    type: EventType = field(default=EventType.CAMERA_HEALTH, init=False)

    def payload(self) -> dict[str, Any]:
        return {
            **super().payload(),
            "camera_id": self.camera_id,
            "fps": round(self.fps, 2),
            "last_frame_age_s": round(self.last_frame_age_s, 2),
            "queue_depth": self.queue_depth,
            "healthy": self.healthy,
        }


# --------------------------------------------------------------------------- #
# Bus contracts (implemented in app.events.event_bus)
# --------------------------------------------------------------------------- #
@runtime_checkable
class Subscription(Protocol):
    """An async stream of events for one SSE client."""

    def __aiter__(self) -> Subscription: ...
    async def __anext__(self) -> Event: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class EventBus(Protocol):
    """Thread-safe fan-out: workers ``publish`` (sync); SSE consumers ``subscribe``."""

    def publish(self, event: Event) -> None: ...
    def subscribe(self, topics: set[str] | None = None) -> Subscription: ...
