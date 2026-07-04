"""ORM-row -> domain value-object mappers (HLD 6.5).

Repositories return ORM rows; the engine consumes the immutable value objects in
:mod:`app.domain.models`. These functions are the single conversion point so the
JSON-column encodings (polygon as ``[[x, y], ...]``, roles as ``list[str]``,
line ``in_direction`` as ``[dx, dy]``) are decoded in exactly one place and the
pipeline never imports the ORM.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.db.models.camera import Camera
from app.db.models.geometry import Line, Zone
from app.domain.models import (
    CameraRole,
    CameraSpec,
    LineSpec,
    Point,
    ZoneSpec,
    ZoneType,
)


def _to_polygon(raw: Sequence[Sequence[float]]) -> tuple[Point, ...]:
    """Decode a JSON ``[[x, y], ...]`` polygon into a tuple of :class:`Point`."""
    return tuple(Point(float(pt[0]), float(pt[1])) for pt in raw)


def to_zone_spec(zone: Zone) -> ZoneSpec:
    """Map a :class:`Zone` ORM row to an immutable :class:`ZoneSpec`."""
    return ZoneSpec(
        id=zone.id,
        camera_id=zone.camera_id,
        name=zone.name,
        type=ZoneType(zone.type),
        polygon=_to_polygon(zone.polygon),
        dt_space_id=zone.dt_space_id,
        safe_limit=zone.safe_limit,
    )


def to_line_spec(line: Line) -> LineSpec:
    """Map a :class:`Line` ORM row to an immutable :class:`LineSpec`.

    :raises ValueError: If ``line.points`` does not hold exactly two endpoints.
    """
    points = line.points
    if len(points) != 2:
        raise ValueError(
            f"line {line.id!r} must have exactly two endpoints, got {len(points)}"
        )
    start, end = points
    dx, dy = line.in_direction
    return LineSpec(
        id=line.id,
        camera_id=line.camera_id,
        name=line.name,
        points=(Point(float(start[0]), float(start[1])), Point(float(end[0]), float(end[1]))),
        in_direction=Point(float(dx), float(dy)),
        area_id=line.area_id,
        dt_space_id=line.dt_space_id,
    )


def to_camera_spec(
    camera: Camera,
    zones: Iterable[Zone],
    lines: Iterable[Line],
) -> CameraSpec:
    """Assemble a :class:`CameraSpec` from a camera row and its zones/lines.

    ``zones`` and ``lines`` are passed explicitly rather than read off the
    relationships so callers can control whether the collections are already
    loaded (avoiding lazy-load surprises outside a session).
    """
    return CameraSpec(
        id=camera.id,
        name=camera.name,
        area=camera.area,
        ip=camera.ip,
        port=camera.port,
        username=camera.username,
        roles=tuple(CameraRole(role) for role in camera.roles),
        stream_channel_sub=camera.stream_channel_sub,
        stream_channel_main=camera.stream_channel_main,
        enabled=camera.enabled,
        imgsz=camera.imgsz,
        zones=tuple(to_zone_spec(zone) for zone in zones),
        lines=tuple(to_line_spec(line) for line in lines),
    )
