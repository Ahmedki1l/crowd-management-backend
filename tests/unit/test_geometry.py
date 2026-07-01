"""Unit tests for the pure image-space geometry helpers (HLD 6.3).

These cover the mathematical core shared by the zone, line, and state-machine
localisers: point-in-polygon membership, bbox/zone overlap ratio, signed point
side, segment-vs-line intersection, and crossing-direction classification.

The real shapely-backed implementations and real domain value objects are
exercised directly — there is nothing to fake here, so no fakes/clock are used.
"""

from __future__ import annotations

from shapely.geometry.polygon import Polygon

from app.domain.models import BBox, CrossingDirection, Point
from app.localisation.geometry import (
    crossing_direction,
    point_in_polygon,
    point_side,
    polygon_overlap_ratio,
    segment_crosses_line,
)

# A square zone spanning (100,100)-(300,300) in image pixels.
_SQUARE = Polygon([(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)])


# --------------------------------------------------------------------------- #
# point_in_polygon
# --------------------------------------------------------------------------- #
def test_point_strictly_inside_polygon_is_member() -> None:
    assert point_in_polygon(Point(200.0, 200.0), _SQUARE) is True


def test_point_outside_polygon_is_not_member() -> None:
    assert point_in_polygon(Point(50.0, 50.0), _SQUARE) is False


def test_point_on_polygon_boundary_counts_as_member() -> None:
    # A point exactly on the left edge must be inside (covers, not contains).
    assert point_in_polygon(Point(100.0, 200.0), _SQUARE) is True


# --------------------------------------------------------------------------- #
# polygon_overlap_ratio
# --------------------------------------------------------------------------- #
def test_overlap_ratio_is_one_when_bbox_fully_inside_polygon() -> None:
    bbox = BBox(x1=150.0, y1=150.0, x2=250.0, y2=250.0)
    assert polygon_overlap_ratio(bbox, _SQUARE) == 1.0


def test_overlap_ratio_is_partial_when_bbox_half_inside_polygon() -> None:
    # Bbox (200,200)-(400,300): exactly the left half (x in [200,300]) overlaps.
    bbox = BBox(x1=200.0, y1=200.0, x2=400.0, y2=300.0)
    assert polygon_overlap_ratio(bbox, _SQUARE) == 0.5


def test_overlap_ratio_is_zero_when_bbox_disjoint_from_polygon() -> None:
    bbox = BBox(x1=500.0, y1=500.0, x2=600.0, y2=600.0)
    assert polygon_overlap_ratio(bbox, _SQUARE) == 0.0


# --------------------------------------------------------------------------- #
# point_side
# --------------------------------------------------------------------------- #
def test_point_side_is_positive_for_point_on_the_left() -> None:
    # Line a->b points down (+y); a point at smaller x is on the left (>0).
    a, b = Point(640.0, 200.0), Point(640.0, 600.0)
    assert point_side(a, b, Point(500.0, 400.0)) > 0.0


def test_point_side_is_negative_for_point_on_the_right() -> None:
    a, b = Point(640.0, 200.0), Point(640.0, 600.0)
    assert point_side(a, b, Point(780.0, 400.0)) < 0.0


def test_point_side_is_zero_for_collinear_point() -> None:
    a, b = Point(640.0, 200.0), Point(640.0, 600.0)
    assert point_side(a, b, Point(640.0, 400.0)) == 0.0


# --------------------------------------------------------------------------- #
# segment_crosses_line
# --------------------------------------------------------------------------- #
def test_segment_crossing_the_line_is_detected() -> None:
    # Horizontal movement sweeping through the vertical line at x=640.
    assert (
        segment_crosses_line(
            Point(500.0, 400.0),
            Point(780.0, 400.0),
            Point(640.0, 200.0),
            Point(640.0, 600.0),
        )
        is True
    )


def test_parallel_segment_does_not_cross_the_line() -> None:
    # A vertical movement parallel to (and offset from) the vertical line.
    assert (
        segment_crosses_line(
            Point(500.0, 200.0),
            Point(500.0, 600.0),
            Point(640.0, 200.0),
            Point(640.0, 600.0),
        )
        is False
    )


def test_non_crossing_segment_on_one_side_is_not_detected() -> None:
    # Horizontal movement entirely to the left of the line never reaches it.
    assert (
        segment_crosses_line(
            Point(400.0, 400.0),
            Point(600.0, 400.0),
            Point(640.0, 200.0),
            Point(640.0, 600.0),
        )
        is False
    )


# --------------------------------------------------------------------------- #
# crossing_direction
# --------------------------------------------------------------------------- #
def test_crossing_along_in_normal_is_classified_in() -> None:
    # Moving left->right (+x) along an in_normal of (1,0) counts as IN.
    direction = crossing_direction(
        Point(500.0, 400.0),
        Point(780.0, 400.0),
        Point(640.0, 200.0),
        Point(640.0, 600.0),
        in_normal=Point(1.0, 0.0),
    )
    assert direction is CrossingDirection.IN


def test_crossing_against_in_normal_is_classified_out() -> None:
    # Moving right->left (-x) against an in_normal of (1,0) counts as OUT.
    direction = crossing_direction(
        Point(780.0, 400.0),
        Point(500.0, 400.0),
        Point(640.0, 200.0),
        Point(640.0, 600.0),
        in_normal=Point(1.0, 0.0),
    )
    assert direction is CrossingDirection.OUT


def test_no_crossing_returns_none() -> None:
    # Movement stays left of the line, so there is no crossing to classify.
    direction = crossing_direction(
        Point(400.0, 400.0),
        Point(600.0, 400.0),
        Point(640.0, 200.0),
        Point(640.0, 600.0),
        in_normal=Point(1.0, 0.0),
    )
    assert direction is None
