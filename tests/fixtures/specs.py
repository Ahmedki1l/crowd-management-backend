"""Domain spec builders and scripted detection sequences for the fakes.

These helpers construct the domain value objects (:class:`ZoneSpec`,
:class:`LineSpec`, :class:`CameraSpec`) the analytics pipeline consumes, plus
ready-made detection scripts for :class:`~app.inference.fakes.FakeDetector`:

* :func:`walk_across_line` — one person whose ground point crosses a line.
* :func:`dwell_inside_polygon` — one person standing still inside a polygon.

Coordinates are in image pixels and match the ``bottom_center`` convention the
geometry layer uses for zone/line membership (HLD 5.4).
"""

from __future__ import annotations

from app.domain.models import (
    BBox,
    CameraRole,
    CameraSpec,
    Detection,
    LineSpec,
    Point,
    ZoneSpec,
    ZoneType,
)

# A square polygon in the middle of a 1280x720 frame, ground-plane oriented.
DEFAULT_POLYGON: tuple[Point, ...] = (
    Point(400.0, 300.0),
    Point(800.0, 300.0),
    Point(800.0, 600.0),
    Point(400.0, 600.0),
)

# A vertical counting line at x=640; IN direction points to +x (left -> right).
DEFAULT_LINE_POINTS: tuple[Point, Point] = (Point(640.0, 200.0), Point(640.0, 600.0))
DEFAULT_IN_DIRECTION: Point = Point(1.0, 0.0)


def make_zone_spec(
    *,
    id: int = 1,
    camera_id: int = 1,
    name: str = "zone-1",
    type: ZoneType = ZoneType.OCCUPANCY,
    polygon: tuple[Point, ...] = DEFAULT_POLYGON,
    dt_space_id: str | None = None,
    safe_limit: int | None = None,
) -> ZoneSpec:
    """Build a :class:`ZoneSpec` with sensible defaults (square polygon)."""
    return ZoneSpec(
        id=id,
        camera_id=camera_id,
        name=name,
        type=type,
        polygon=polygon,
        dt_space_id=dt_space_id,
        safe_limit=safe_limit,
    )


def make_line_spec(
    *,
    id: int = 1,
    camera_id: int = 1,
    name: str = "line-1",
    points: tuple[Point, Point] = DEFAULT_LINE_POINTS,
    in_direction: Point = DEFAULT_IN_DIRECTION,
    area_id: str = "area-1",
    dt_space_id: str | None = None,
) -> LineSpec:
    """Build a :class:`LineSpec` with sensible defaults (vertical line at x=640)."""
    return LineSpec(
        id=id,
        camera_id=camera_id,
        name=name,
        points=points,
        in_direction=in_direction,
        area_id=area_id,
        dt_space_id=dt_space_id,
    )


def make_camera_spec(
    *,
    id: int = 1,
    name: str = "cam-1",
    area: str = "lobby",
    ip: str = "10.0.0.1",
    port: int = 554,
    username: str = "admin",
    roles: tuple[CameraRole, ...] = (CameraRole.OCCUPANCY,),
    stream_channel_sub: int = 102,
    stream_channel_main: int = 101,
    enabled: bool = True,
    zones: tuple[ZoneSpec, ...] = (),
    lines: tuple[LineSpec, ...] = (),
) -> CameraSpec:
    """Build a :class:`CameraSpec`, optionally carrying zones/lines."""
    return CameraSpec(
        id=id,
        name=name,
        area=area,
        ip=ip,
        port=port,
        username=username,
        roles=roles,
        stream_channel_sub=stream_channel_sub,
        stream_channel_main=stream_channel_main,
        enabled=enabled,
        zones=zones,
        lines=lines,
    )


def _person_bbox_at(cx: float, cy: float, w: float = 60.0, h: float = 160.0) -> BBox:
    """A person box whose ``bottom_center`` ground point is ``(cx, cy)``."""
    return BBox(x1=cx - w / 2.0, y1=cy - h, x2=cx + w / 2.0, y2=cy)


def walk_across_line(
    *,
    n: int = 6,
    line_x: float = 640.0,
    y: float = 400.0,
    start_x: float = 500.0,
    end_x: float = 780.0,
    confidence: float = 0.9,
) -> list[list[Detection]]:
    """Script one person walking left-to-right across the vertical line at ``line_x``.

    The ground point sweeps linearly from ``start_x`` (left of the line) to
    ``end_x`` (right of the line) over ``n`` frames, so the same track crosses
    once in the IN direction. Returns one single-detection list per frame.
    """
    if n < 2:
        raise ValueError("walk_across_line needs at least 2 frames")
    if not (start_x < line_x < end_x):
        raise ValueError("start_x must be left and end_x right of line_x")
    step = (end_x - start_x) / (n - 1)
    return [
        [Detection(bbox=_person_bbox_at(start_x + step * i, y), confidence=confidence)]
        for i in range(n)
    ]


def dwell_inside_polygon(
    *,
    n: int = 6,
    cx: float = 600.0,
    cy: float = 450.0,
    confidence: float = 0.9,
) -> list[list[Detection]]:
    """Script one person standing still inside the default polygon for ``n`` frames.

    The ground point ``(cx, cy)`` stays fixed (default is inside
    :data:`DEFAULT_POLYGON`), giving the same stable track every frame so dwell
    accumulates. Returns one single-detection list per frame.
    """
    if n < 1:
        raise ValueError("dwell_inside_polygon needs at least 1 frame")
    box = _person_bbox_at(cx, cy)
    return [[Detection(bbox=box, confidence=confidence)] for _ in range(n)]
