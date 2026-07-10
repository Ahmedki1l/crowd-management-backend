"""A backlogged queue must not add latency on the snapshot-occupancy path.

Regression: `put` drops the oldest frame but `get` also returns the oldest, so a
depth-N queue handed the worker a frame (N-1) producer-periods old whenever the
consumer fell behind. At queue_maxsize=4 and a 3s snapshot interval that is ~9s
of constant staleness in the occupancy the API reports.
"""

from __future__ import annotations

import numpy as np

from app.domain.models import FramePacket
from app.ingestion.frame_queue import BoundedFrameQueue

_IMAGE = np.zeros((2, 2, 3), dtype=np.uint8)


def _packet(ts: float) -> FramePacket:
    return FramePacket(camera_id=1, frame_idx=int(ts), ts=ts, image=_IMAGE)


def test_get_latest_returns_newest_and_discards_backlog() -> None:
    q = BoundedFrameQueue(4)
    for ts in (1.0, 2.0, 3.0, 4.0):
        q.put(_packet(ts))

    packet = q.get_latest(timeout=0)

    assert packet is not None
    assert packet.ts == 4.0, "must hand the worker the freshest frame"
    assert q.qsize() == 0, "the stale backlog must be discarded, not left to accumulate"


def test_get_latest_counts_skipped_frames_as_drops() -> None:
    q = BoundedFrameQueue(4)
    for ts in (1.0, 2.0, 3.0):
        q.put(_packet(ts))
    assert q.dropped == 0

    q.get_latest(timeout=0)
    assert q.dropped == 2, "skipped frames stay observable"


def test_get_latest_is_monotonic_across_calls() -> None:
    """Unlike a LIFO pop, successive calls never hand back an older frame."""
    q = BoundedFrameQueue(4)
    q.put(_packet(1.0))
    q.put(_packet(2.0))
    first = q.get_latest(timeout=0)

    q.put(_packet(3.0))
    second = q.get_latest(timeout=0)

    assert first is not None and second is not None
    assert second.ts > first.ts


def test_get_latest_returns_none_on_empty_queue() -> None:
    assert BoundedFrameQueue(4).get_latest(timeout=0) is None


def test_get_latest_matches_get_when_not_backlogged() -> None:
    """With a single buffered frame the two reads are indistinguishable."""
    q = BoundedFrameQueue(4)
    q.put(_packet(7.0))
    assert q.get_latest(timeout=0).ts == 7.0

    q.put(_packet(8.0))
    assert q.get(timeout=0).ts == 8.0


def test_fifo_get_still_preserves_order_for_the_tracked_path() -> None:
    """The tracker/line-crossing path must keep every frame, in order."""
    q = BoundedFrameQueue(4)
    for ts in (1.0, 2.0, 3.0):
        q.put(_packet(ts))
    assert [q.get(timeout=0).ts for _ in range(3)] == [1.0, 2.0, 3.0]
