"""Unit tests for :class:`SnapshotCaptureThread`.

Drives one ``_capture_once`` iteration at a time with an injected frame provider
(no HTTP, no cv2, no real thread) to assert the enqueue path, the fps/health
bookkeeping, and the self-healing behaviour on a failed or raising fetch.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.domain.models import CameraRole
from app.ingestion.capture import SnapshotCaptureThread
from app.ingestion.frame_queue import BoundedFrameQueue
from app.utils.clock import FakeClock


def _thread(
    provider: Callable[[], np.ndarray | None],
    clock: FakeClock,
    *,
    interval_s: float = 2.0,
    watchdog: float = 15.0,
) -> SnapshotCaptureThread:
    return SnapshotCaptureThread(
        camera_id=7,
        frame_provider=provider,
        queue=BoundedFrameQueue(4),
        clock=clock,
        interval_s=interval_s,
        watchdog_stale_seconds=watchdog,
        role=CameraRole.OCCUPANCY,
    )


def test_capture_once_enqueues_stamped_frame_and_marks_healthy() -> None:
    clock = FakeClock(start=1000.0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    thread = _thread(lambda: image, clock)

    thread._capture_once()

    packet = thread._queue.get(timeout=0.1)
    assert packet is not None
    assert packet.camera_id == 7
    assert packet.role is CameraRole.OCCUPANCY
    assert packet.ts == 1000.0
    assert thread.last_frame_ts == 1000.0
    assert thread.is_healthy() is True


def test_capture_once_none_provider_skips_and_stays_unhealthy() -> None:
    clock = FakeClock(start=1000.0)
    thread = _thread(lambda: None, clock)

    thread._capture_once()

    assert thread._queue.get(timeout=0.05) is None
    assert thread.last_frame_ts is None
    assert thread.is_healthy() is False


def test_capture_once_swallows_provider_exception() -> None:
    clock = FakeClock(start=1000.0)

    def boom() -> np.ndarray:
        raise RuntimeError("fetch failed")

    thread = _thread(boom, clock)

    thread._capture_once()  # must not raise

    assert thread._queue.get(timeout=0.05) is None
    assert thread.last_frame_ts is None


def test_is_healthy_goes_stale_past_watchdog_window() -> None:
    clock = FakeClock(start=1000.0)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    thread = _thread(lambda: image, clock, watchdog=15.0)

    thread._capture_once()
    assert thread.is_healthy() is True

    clock.advance(20.0)  # beyond the watchdog window
    assert thread.is_healthy() is False


def test_fps_estimate_reflects_pull_interval() -> None:
    clock = FakeClock(start=1000.0)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    thread = _thread(lambda: image, clock)

    thread._capture_once()
    assert thread.fps_estimate == 0.0  # no interval from a single frame

    clock.advance(2.0)
    thread._capture_once()  # 2s cadence -> 0.5 fps
    assert abs(thread.fps_estimate - 0.5) < 1e-9


def test_run_invokes_on_close_when_loop_exits() -> None:
    closed: list[bool] = []
    thread = SnapshotCaptureThread(
        camera_id=1,
        frame_provider=lambda: None,
        queue=BoundedFrameQueue(2),
        clock=FakeClock(start=1000.0),
        interval_s=0.01,
        watchdog_stale_seconds=15.0,
        role=CameraRole.OCCUPANCY,
        on_close=lambda: closed.append(True),
    )

    thread.start()
    thread.stop()
    thread.join(timeout=2.0)

    assert closed == [True]  # HTTP client (or any resource) released on exit
    assert not thread.is_alive()
