"""Pure image-space geometry helpers (HLD 6.3).

All functions here are deterministic and side-effect free: they take domain
value objects (:class:`~app.domain.models.Point`, :class:`~app.domain.models.BBox`)
plus ``shapely`` polygons and return plain scalars/booleans. They form the
mathematical core shared by the zone, line, and state-machine localisers so the
same point-in-polygon / crossing logic is never re-implemented per package.

Coordinates are image pixels (x right, y down). ``shapely`` is a core dependency
and may be imported at module level.
"""

from __future__ import annotations

from shapely.geometry import LineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry.polygon import Polygon

from app.domain.models import BBox, CrossingDirection, Point


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Return whether ``point`` lies inside ``polygon``, boundary included.

    Uses :meth:`shapely.Polygon.covers`, which (unlike ``contains``) treats
    points lying exactly on an edge or vertex as inside — the desired behaviour
    for zone membership where a person standing on a boundary should count.

    Args:
        point: Image-space point to test.
        polygon: Pre-built shapely polygon for the zone.

    Returns:
        ``True`` if the point is on or inside the polygon, else ``False``.
    """
    return bool(polygon.covers(ShapelyPoint(point.x, point.y)))


def polygon_overlap_ratio(bbox: BBox, polygon: Polygon) -> float:
    """Fraction of ``bbox`` area covered by ``polygon``.

    Computed as ``intersection_area / bbox.area``. Returns ``0.0`` for an empty
    (zero-area) bbox to avoid division by zero.

    Args:
        bbox: Axis-aligned detection box.
        polygon: Pre-built shapely polygon for the zone.

    Returns:
        Overlap ratio in ``[0.0, 1.0]``.
    """
    box_area = bbox.area
    if box_area <= 0.0:
        return 0.0
    box_polygon = Polygon(
        [
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y1),
            (bbox.x2, bbox.y2),
            (bbox.x1, bbox.y2),
        ]
    )
    intersection_area = box_polygon.intersection(polygon).area
    return intersection_area / box_area


def point_side(a: Point, b: Point, p: Point) -> float:
    """Signed side of point ``p`` relative to the directed line ``a -> b``.

    Returns the z-component of the 2-D cross product
    ``(b - a) x (p - a)``. Using image coordinates (y pointing down):

    * ``> 0`` — ``p`` is on the left of ``a -> b``,
    * ``< 0`` — ``p`` is on the right,
    * ``== 0`` — ``p`` is collinear with the line.

    Args:
        a: Line start point.
        b: Line end point.
        p: Point to classify.

    Returns:
        The signed cross-product magnitude.
    """
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)


def segment_crosses_line(
    prev: Point, curr: Point, line_a: Point, line_b: Point
) -> bool:
    """Return whether segment ``prev -> curr`` intersects segment ``line_a -> line_b``.

    Uses shapely segment intersection, which is robust for collinear-overlap and
    touching-endpoint cases.

    Args:
        prev: Previous track position.
        curr: Current track position.
        line_a: First endpoint of the counting line.
        line_b: Second endpoint of the counting line.

    Returns:
        ``True`` if the two segments share at least one point.
    """
    movement = LineString([(prev.x, prev.y), (curr.x, curr.y)])
    line = LineString([(line_a.x, line_a.y), (line_b.x, line_b.y)])
    return bool(movement.intersects(line))


def crossing_direction(
    prev: Point,
    curr: Point,
    line_a: Point,
    line_b: Point,
    in_normal: Point,
) -> CrossingDirection | None:
    """Classify the direction in which a track crossed a counting line.

    Returns ``None`` when the movement segment does not cross the line. When it
    does, the direction is decided by *which side of the line the track ends on*
    relative to ``in_normal``: a track that moves onto the IN side counts as
    :class:`CrossingDirection.IN`, otherwise OUT. This side-based test is correct
    for any crossing angle, unlike a movement-vector dot product which
    misclassifies shallow/diagonal crossings.

    Args:
        prev: Previous track position.
        curr: Current track position.
        line_a: First endpoint of the counting line.
        line_b: Second endpoint of the counting line.
        in_normal: Reference vector pointing in the direction that counts as IN.

    Returns:
        :class:`CrossingDirection.IN`, :class:`CrossingDirection.OUT`, or
        ``None`` if there was no crossing.
    """
    if not segment_crosses_line(prev, curr, line_a, line_b):
        return None
    # Decide IN/OUT from which side of the line the track moved across — not from
    # ``movement · in_normal``. The dot product misclassifies shallow/diagonal
    # crossings: when ``in_normal`` is not exactly perpendicular to the line, the
    # (large) component of motion parallel to the line leaks into the dot and can
    # flip the sign. ``point_side`` only depends on which side of the line each
    # endpoint lies on, so it is correct for any crossing angle.
    side_delta = point_side(line_a, line_b, curr) - point_side(line_a, line_b, prev)
    # Sign of the side that ``in_normal`` points toward, relative to a -> b.
    in_side = (line_b.x - line_a.x) * in_normal.y - (line_b.y - line_a.y) * in_normal.x
    return CrossingDirection.IN if side_delta * in_side > 0 else CrossingDirection.OUT
