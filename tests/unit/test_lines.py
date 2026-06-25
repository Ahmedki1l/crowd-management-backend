"""Unit tests for line-crossing detection (HLD 6.3).

:class:`LineCrossingDetector` keeps the last ground point per track and emits a
:class:`Crossing` when a track's movement segment intersects a configured line.
These tests drive it with real :class:`TrackedDetection` objects whose
``bottom_center`` ground points sweep across the default vertical line, and
assert the observable crossings (count, direction, area_id, track_id) — the real
geometry is exercised, nothing is mocked.

Timestamps come from the deterministic :class:`FakeClock` fixture.
"""

from __future__ import annotations

from app.domain.models import BBox, CrossingDirection, TrackedDetection
from app.localisation.lines import LineCrossingDetector
from app.utils.clock import FakeClock
from tests.fixtures.specs import make_line_spec

# The default line is vertical at x=640 with IN direction +x (left -> right).
_LEFT_GROUND_X = 500.0
_RIGHT_GROUND_X = 780.0
_GROUND_Y = 400.0


def _person_at(track_id: int, ground_x: float) -> TrackedDetection:
    """Build a tracked detection whose bbox bottom-centre is (ground_x, _GROUND_Y)."""
    half_w = 30.0
    height = 160.0
    bbox = BBox(
        x1=ground_x - half_w,
        y1=_GROUND_Y - height,
        x2=ground_x + half_w,
        y2=_GROUND_Y,
    )
    return TrackedDetection(track_id=track_id, bbox=bbox, confidence=0.9)


def test_first_sighting_of_a_track_produces_no_crossing(
    fake_clock: FakeClock,
) -> None:
    detector = LineCrossingDetector([make_line_spec()])
    # A track positioned to the right of the line, but with no previous point.
    crossings = detector.update([_person_at(1, _RIGHT_GROUND_X)], ts=fake_clock.now())
    assert crossings == []


def test_track_crossing_line_left_to_right_produces_one_in_crossing(
    fake_clock: FakeClock,
) -> None:
    detector = LineCrossingDetector([make_line_spec(id=5, area_id="lobby-door")])

    # Frame 0: establish the previous point to the left of the line.
    detector.update([_person_at(1, _LEFT_GROUND_X)], ts=fake_clock.now())
    fake_clock.advance(1.0)
    # Frame 1: move across to the right of the line.
    crossings = detector.update([_person_at(1, _RIGHT_GROUND_X)], ts=fake_clock.now())

    assert len(crossings) == 1
    crossing = crossings[0]
    assert crossing.direction is CrossingDirection.IN
    assert crossing.area_id == "lobby-door"
    assert crossing.track_id == 1
    assert crossing.line_id == 5


def test_track_crossing_line_right_to_left_produces_one_out_crossing(
    fake_clock: FakeClock,
) -> None:
    detector = LineCrossingDetector([make_line_spec()])

    detector.update([_person_at(1, _RIGHT_GROUND_X)], ts=fake_clock.now())
    fake_clock.advance(1.0)
    crossings = detector.update([_person_at(1, _LEFT_GROUND_X)], ts=fake_clock.now())

    assert len(crossings) == 1
    assert crossings[0].direction is CrossingDirection.OUT


def test_two_tracks_crossing_produce_two_crossings(fake_clock: FakeClock) -> None:
    detector = LineCrossingDetector([make_line_spec()])

    # Both tracks start left of the line.
    detector.update(
        [_person_at(1, _LEFT_GROUND_X), _person_at(2, _LEFT_GROUND_X)],
        ts=fake_clock.now(),
    )
    fake_clock.advance(1.0)
    # Both move across to the right.
    crossings = detector.update(
        [_person_at(1, _RIGHT_GROUND_X), _person_at(2, _RIGHT_GROUND_X)],
        ts=fake_clock.now(),
    )

    assert len(crossings) == 2
    assert {c.track_id for c in crossings} == {1, 2}
    assert all(c.direction is CrossingDirection.IN for c in crossings)


def test_track_not_reaching_line_produces_no_crossing(
    fake_clock: FakeClock,
) -> None:
    detector = LineCrossingDetector([make_line_spec()])

    # Move within the left side only (500 -> 600), never reaching x=640.
    detector.update([_person_at(1, _LEFT_GROUND_X)], ts=fake_clock.now())
    fake_clock.advance(1.0)
    crossings = detector.update([_person_at(1, 600.0)], ts=fake_clock.now())

    assert crossings == []
