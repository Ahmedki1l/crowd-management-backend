"""Unit tests for the bounded drop-oldest frame queue (app.ingestion.frame_queue)."""

from __future__ import annotations

import numpy as np

from app.domain.models import FramePacket
from app.ingestion.frame_queue import BoundedFrameQueue


def _packet(frame_idx: int) -> FramePacket:
    """A FramePacket tagged by ``frame_idx`` so identity is observable."""
    return FramePacket(
        camera_id=1,
        frame_idx=frame_idx,
        ts=float(frame_idx),
        image=np.zeros((2, 2, 3), dtype=np.uint8),
    )


def test_put_drops_oldest_when_full_and_retains_newest() -> None:
    queue = BoundedFrameQueue(maxsize=2)
    queue.put(_packet(0))
    queue.put(_packet(1))
    queue.put(_packet(2))  # overflows: oldest (0) is dropped

    first = queue.get(timeout=0)
    second = queue.get(timeout=0)

    assert [first.frame_idx, second.frame_idx] == [1, 2]
    assert queue.dropped == 1


def test_get_returns_none_on_timeout_when_empty() -> None:
    queue = BoundedFrameQueue(maxsize=2)

    assert queue.get(timeout=0) is None


def test_qsize_reflects_buffered_count() -> None:
    queue = BoundedFrameQueue(maxsize=3)
    queue.put(_packet(0))
    queue.put(_packet(1))

    assert queue.qsize() == 2
