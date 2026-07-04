"""Shared domain value objects — the single vocabulary of the pipeline.

Every layer (ingestion -> inference -> localisation -> analytics -> publish)
exchanges these types. Keeping one definition per concept here is deliberate:
it prevents the same concept (a detection, a zone, a camera) from being
re-modelled differently in each package.

ORM rows (``app.db.models``) and API payloads (``app.api.schemas``) are mapped
*to and from* these domain objects by repositories/services — they are not used
directly inside the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import numpy as np


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class CameraRole(str, Enum):
    """Drives the per-camera frame-rate tier (HLD 5.1)."""

    ENTRY_EXIT = "entry_exit"
    WAITING = "waiting"
    OCCUPANCY = "occupancy"
    HEATMAP = "heatmap"


class ZoneType(str, Enum):
    OCCUPANCY = "occupancy"
    WAITING = "waiting"
    RESTRICTED = "restricted"


class CrossingDirection(str, Enum):
    IN = "in"
    OUT = "out"


class ZoneState(str, Enum):
    """Debounced presence state machine (HLD 5.7)."""

    VACANT = "vacant"
    ENTERING = "entering"
    OCCUPIED = "occupied"
    LEAVING = "leaving"


class AlertType(str, Enum):
    INTRUSION = "intrusion"
    OVERCROWDING = "overcrowding"


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #
class Point(NamedTuple):
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned box in image pixel coordinates (x1,y1 top-left)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centroid(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Ground-contact point used for zone/line tests (HLD 5.4)."""
        return Point((self.x1 + self.x2) / 2.0, self.y2)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


# --------------------------------------------------------------------------- #
# Perception outputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Detection:
    """One person located in a single frame (Layer 2 output, pre-tracking)."""

    bbox: BBox
    confidence: float
    class_id: int = 0


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection with a stable track identity (Layer 2 output, post-tracking)."""

    track_id: int
    bbox: BBox
    confidence: float
    class_id: int = 0
    embedding: np.ndarray | None = None
    # Cross-camera identity assigned by the Re-ID manager (None when Re-ID off).
    global_id: int | None = None

    @property
    def bottom_center(self) -> Point:
        return self.bbox.bottom_center


@dataclass(slots=True)
class FramePacket:
    """A decoded frame in flight from capture to inference."""

    camera_id: int
    frame_idx: int
    ts: float  # epoch seconds (monotonic-source clock); see app.utils.clock
    image: np.ndarray
    role: CameraRole = CameraRole.OCCUPANCY


# --------------------------------------------------------------------------- #
# Pipeline configuration value objects (built from DB rows by repositories).
# Decoupled from the ORM so the engine never imports the persistence layer.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ZoneSpec:
    id: int
    camera_id: int
    name: str
    type: ZoneType
    polygon: tuple[Point, ...]
    dt_space_id: str | None = None
    safe_limit: int | None = None


@dataclass(frozen=True, slots=True)
class LineSpec:
    id: int
    camera_id: int
    name: str
    points: tuple[Point, Point]
    # Reference direction that counts as IN, expressed as a unit-ish vector.
    in_direction: Point
    area_id: str
    dt_space_id: str | None = None


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """Everything a worker needs to run one camera (credentials resolved separately)."""

    id: int
    name: str
    area: str
    ip: str
    port: int
    username: str
    roles: tuple[CameraRole, ...]
    stream_channel_sub: int
    stream_channel_main: int
    enabled: bool = True
    # Per-camera detector input size; None => use the global detector.imgsz.
    imgsz: int | None = None
    zones: tuple[ZoneSpec, ...] = field(default_factory=tuple)
    lines: tuple[LineSpec, ...] = field(default_factory=tuple)

    @property
    def fps_role(self) -> CameraRole:
        """The most motion-sensitive role decides the camera's fps tier."""
        priority = [
            CameraRole.ENTRY_EXIT,
            CameraRole.WAITING,
            CameraRole.OCCUPANCY,
            CameraRole.HEATMAP,
        ]
        for role in priority:
            if role in self.roles:
                return role
        return CameraRole.OCCUPANCY
