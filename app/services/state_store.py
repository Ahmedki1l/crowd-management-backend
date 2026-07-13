"""Current-state cache — the read model for fast API queries (HLD 6.5).

Thread-safe. Workers/projectors write; API routes read. This is the query side;
the event bus is the streaming side. Both are fed from the same analytics events.
In scaled-out deployments a Redis-backed implementation replaces the in-memory
maps behind the same accessor (``get_state_store``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only; lazy-imported at runtime
    from app.services.redis_state_store import RedisStateStore


@dataclass(slots=True)
class OccupancyState:
    zone_id: int
    count: int
    dt_space_id: str | None
    ts: float


@dataclass(slots=True)
class LineCountState:
    line_id: int
    in_count: int = 0
    out_count: int = 0


@dataclass(slots=True)
class AreaCountState:
    area_id: str
    in_count: int = 0
    out_count: int = 0
    net: int = 0
    ts: float = 0.0
    lines: dict[int, LineCountState] = field(default_factory=dict)


@dataclass(slots=True)
class CameraHealthState:
    camera_id: int
    fps: float = 0.0
    last_frame_age_s: float = 0.0
    queue_depth: int = 0
    healthy: bool = True
    ts: float = 0.0


class StateStore:
    """In-memory current-state cache. All public methods are thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._occupancy: dict[int, OccupancyState] = {}
        self._counts: dict[str, AreaCountState] = {}
        self._health: dict[int, CameraHealthState] = {}

    # --- Occupancy ---------------------------------------------------------
    def set_occupancy(self, zone_id: int, count: int, dt_space_id: str | None, ts: float) -> None:
        with self._lock:
            self._occupancy[zone_id] = OccupancyState(zone_id, count, dt_space_id, ts)

    def get_occupancy(self, zone_id: int) -> OccupancyState | None:
        with self._lock:
            return self._occupancy.get(zone_id)

    def all_occupancy(self) -> list[OccupancyState]:
        with self._lock:
            return list(self._occupancy.values())

    # --- Entry/Exit counts -------------------------------------------------
    def set_counts(
        self,
        area_id: str,
        in_count: int,
        out_count: int,
        net: int,
        ts: float,
        lines: dict[int, tuple[int, int]] | None = None,
    ) -> None:
        with self._lock:
            state = AreaCountState(area_id, in_count, out_count, net, ts)
            if lines:
                state.lines = {
                    lid: LineCountState(lid, i, o) for lid, (i, o) in lines.items()
                }
            self._counts[area_id] = state

    def get_counts(self, area_id: str) -> AreaCountState | None:
        with self._lock:
            return self._counts.get(area_id)

    def all_counts(self) -> list[AreaCountState]:
        with self._lock:
            return list(self._counts.values())


    # --- Camera health -----------------------------------------------------
    def set_camera_health(self, health: CameraHealthState) -> None:
        with self._lock:
            self._health[health.camera_id] = health

    def get_camera_health(self, camera_id: int) -> CameraHealthState | None:
        with self._lock:
            return self._health.get(camera_id)

    def all_camera_health(self) -> list[CameraHealthState]:
        with self._lock:
            return list(self._health.values())

    def total_occupancy(self) -> int:
        with self._lock:
            return sum(o.count for o in self._occupancy.values())

    def clear(self) -> None:
        with self._lock:
            self._occupancy.clear()
            self._counts.clear()
            self._health.clear()


_store: StateStore | RedisStateStore | None = None


def get_state_store() -> StateStore | RedisStateStore:
    """Return the process-wide state-store singleton.

    Uses the Redis-backed store when ``cache.url`` is configured (scaled-out
    topology, HLD 6.5), else the in-memory store. The Redis connection is
    deferred, so selecting it does not require a live server at import time.
    """
    global _store
    if _store is None:
        from app.config.settings import get_settings

        cache_url = get_settings().cache.url
        if cache_url:
            from app.services.redis_state_store import RedisStateStore

            _store = RedisStateStore(cache_url)
        else:
            _store = StateStore()
    return _store


def reset_state_store() -> None:
    global _store
    _store = None
