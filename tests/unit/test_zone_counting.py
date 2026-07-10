"""Occupancy counting anchors on the ground point, not on bbox overlap.

Regression: a person standing *outside* an ROI whose tall bbox clipped the
polygon edge by >= MIN_OVERLAP_RATIO was counted as occupying the zone, so a
waiting-area count read one higher than the people actually in it. Tracked
presence (:meth:`ZoneEvaluator.membership`) deliberately keeps the overlap
fallback; raw occupancy counting must not.
"""

from __future__ import annotations

from app.domain.models import BBox, Detection, Point, TrackedDetection
from app.localisation.zones import MIN_OVERLAP_RATIO, ZoneEvaluator
from tests.fixtures.specs import make_zone_spec

# Unit square 0..100 on both axes.
SQUARE = (Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100))

# Ground point (20, 150) is below the polygon, but the box overlaps it:
# box is 40 x 100 = 4000 px^2; intersection is 40 x 50 = 2000 px^2 -> ratio 0.5.
OUTSIDE_BUT_OVERLAPPING = BBox(0, 50, 40, 150)

# Ground point (50, 90) sits inside the polygon.
INSIDE = BBox(30, 10, 70, 90)


def _evaluator() -> ZoneEvaluator:
    return ZoneEvaluator([make_zone_spec(id=7, polygon=SQUARE)])


def test_overlapping_bbox_with_ground_point_outside_is_not_counted() -> None:
    """The exact CAM-26 case: feet outside the ROI, bbox clipping its corner."""
    evaluator = _evaluator()
    detection = Detection(bbox=OUTSIDE_BUT_OVERLAPPING, confidence=0.79)

    # Precondition: this box really does trip the overlap rule.
    from shapely.geometry.polygon import Polygon

    from app.localisation.geometry import polygon_overlap_ratio

    ratio = polygon_overlap_ratio(OUTSIDE_BUT_OVERLAPPING, Polygon([(p.x, p.y) for p in SQUARE]))
    assert ratio >= MIN_OVERLAP_RATIO

    assert evaluator.count_in_zones([detection]) == {7: 0}


def test_ground_point_inside_is_counted() -> None:
    evaluator = _evaluator()
    detection = Detection(bbox=INSIDE, confidence=0.83)
    assert evaluator.count_in_zones([detection]) == {7: 1}


def test_count_is_one_per_person_not_one_per_overlap() -> None:
    """Three people inside, one loitering outside with an overlapping bbox -> 3."""
    evaluator = _evaluator()
    detections = [
        Detection(bbox=BBox(10, 10, 30, 40), confidence=0.9),
        Detection(bbox=BBox(40, 20, 60, 50), confidence=0.9),
        Detection(bbox=BBox(70, 30, 90, 60), confidence=0.9),
        Detection(bbox=OUTSIDE_BUT_OVERLAPPING, confidence=0.79),
    ]
    assert evaluator.count_in_zones(detections) == {7: 3}


def test_membership_keeps_the_partial_entry_overlap_rule() -> None:
    """Tracked presence still tolerates a partially-entered track (HLD 6.3)."""
    evaluator = _evaluator()
    tracked = TrackedDetection(track_id=1, bbox=OUTSIDE_BUT_OVERLAPPING, confidence=0.79)
    assert evaluator.membership([tracked]) == {7: {1}}
