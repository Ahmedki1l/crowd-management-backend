"""Unit tests for :class:`app.analytics.heatmap.HeatmapAccumulator`.

The accumulator maps each tracked detection's ground point (bbox bottom-centre)
to a grid cell and increments that cell's weight; ``maybe_flush`` returns a
``GridSnapshot`` of accumulated cells only once the flush interval has elapsed,
then resets the accumulator. The deterministic ``FakeClock`` is the time source;
the per-call ``ts`` drives the flush clock.
"""

from __future__ import annotations

from app.analytics.heatmap import GridSnapshot, HeatmapAccumulator
from app.domain.models import BBox, TrackedDetection


def _tracked_at(cx, cy, *, track_id=1, w=60.0, h=160.0):
    """A tracked detection whose bottom-centre ground point is ``(cx, cy)``."""
    return TrackedDetection(
        track_id=track_id,
        bbox=BBox(x1=cx - w / 2.0, y1=cy - h, x2=cx + w / 2.0, y2=cy),
        confidence=0.9,
    )


def test_accumulate_maps_bottom_center_to_expected_cell(fake_clock):
    # 4x4 grid over 1280x720. Ground point (480, 270): col=int(480/1280*4)=1,
    # row=int(270/720*4)=1 -> cell key "1,1".
    acc = HeatmapAccumulator(
        camera_id=1, cols=4, rows=4, flush_interval_seconds=10.0, clock=fake_clock
    )

    acc.accumulate([_tracked_at(480.0, 270.0)], frame_w=1280, frame_h=720, ts=1000.0)
    snapshot = acc.maybe_flush(ts=1010.0)

    assert snapshot is not None
    assert snapshot.cells == {"1,1": 1.0}


def test_accumulate_increments_weight_for_repeated_points(fake_clock):
    acc = HeatmapAccumulator(
        camera_id=1, cols=4, rows=4, flush_interval_seconds=10.0, clock=fake_clock
    )

    acc.accumulate(
        [_tracked_at(480.0, 270.0, track_id=1)], frame_w=1280, frame_h=720, ts=1000.0
    )
    acc.accumulate(
        [_tracked_at(480.0, 270.0, track_id=2)], frame_w=1280, frame_h=720, ts=1001.0
    )
    snapshot = acc.maybe_flush(ts=1010.0)

    assert snapshot is not None
    assert snapshot.cells["1,1"] == 2.0


def test_maybe_flush_returns_none_before_interval(fake_clock):
    acc = HeatmapAccumulator(
        camera_id=1, cols=4, rows=4, flush_interval_seconds=10.0, clock=fake_clock
    )

    acc.accumulate([_tracked_at(480.0, 270.0)], frame_w=1280, frame_h=720, ts=1000.0)
    early = acc.maybe_flush(ts=1005.0)  # only 5s of a 10s interval

    assert early is None


def test_maybe_flush_returns_snapshot_after_interval(fake_clock):
    acc = HeatmapAccumulator(
        camera_id=9, cols=4, rows=4, flush_interval_seconds=10.0, clock=fake_clock
    )

    acc.accumulate([_tracked_at(480.0, 270.0)], frame_w=1280, frame_h=720, ts=1000.0)
    snapshot = acc.maybe_flush(ts=1010.0)

    assert isinstance(snapshot, GridSnapshot)
    assert snapshot.camera_id == 9


def test_flush_resets_accumulator(fake_clock):
    acc = HeatmapAccumulator(
        camera_id=1, cols=4, rows=4, flush_interval_seconds=10.0, clock=fake_clock
    )

    acc.accumulate([_tracked_at(480.0, 270.0)], frame_w=1280, frame_h=720, ts=1000.0)
    acc.maybe_flush(ts=1010.0)  # drains the accumulator
    next_snapshot = acc.maybe_flush(ts=1020.0)  # nothing added since the flush

    assert next_snapshot is not None
    assert next_snapshot.cells == {}
