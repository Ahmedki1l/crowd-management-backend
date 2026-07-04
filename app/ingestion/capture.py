"""RTSP capture thread — decoupled decode loop (HLD 6.1, 5.3).

One :class:`RtspCaptureThread` owns one camera's RTSP connection. It runs the
blocking OpenCV/FFmpeg read loop on its own thread, samples frames down to the
role's target fps, and pushes :class:`~app.domain.models.FramePacket` objects
into a :class:`~app.ingestion.frame_queue.BoundedFrameQueue`. The queue (not the
thread) is what decouples capture from inference; this thread therefore never
blocks on the consumer.

Resilience: a single read failure must not kill the thread. On error the capture
releases the handle and reconnects with exponential backoff between
``reconnect_backoff_base_s`` and ``reconnect_backoff_max_s`` (from
:class:`~app.config.schema.ProcessingConfig`). A watchdog (``is_healthy``)
reports staleness when no frame has arrived within ``watchdog_stale_seconds``.

``cv2`` is imported lazily inside :meth:`run` so this module imports cleanly with
only core dependencies installed.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from app.config.schema import ProcessingConfig
from app.domain.interfaces import Clock, FrameSource
from app.domain.models import CameraRole, FramePacket
from app.ingestion.frame_queue import BoundedFrameQueue
from app.ingestion.stream_url import redact
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    import numpy as np

logger = get_logger(__name__)

# Number of recent frame intervals used to smooth the fps estimate.
_FPS_WINDOW = 30


@runtime_checkable
class CaptureThread(Protocol):
    """Control surface a :class:`~app.engine.camera_pipeline.CameraPipeline` needs
    from its frame producer. Satisfied by :class:`RtspCaptureThread` (production)
    and :class:`FrameSourceCaptureThread` (recorded clips / tests)."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    def is_healthy(self) -> bool: ...
    @property
    def fps_estimate(self) -> float: ...
    @property
    def last_frame_ts(self) -> float | None: ...


class _CaptureHealthMixin:
    """Shared fps estimate + staleness watchdog for real capture threads.

    A host thread maintains ``_last_frame_ts`` / ``_last_emit_ts`` /
    ``_recent_intervals`` as it emits frames; this mixin turns them into the
    :class:`CaptureThread` health surface. The host must set ``_clock`` and
    ``_watchdog_stale_seconds``. Kept separate from :class:`FrameSourceCaptureThread`
    (finite test clips) which has no real-time rate to report.
    """

    _clock: Clock
    _watchdog_stale_seconds: float
    _last_frame_ts: float | None
    _last_emit_ts: float | None
    _recent_intervals: list[float]

    def is_healthy(self) -> bool:
        """Return whether a frame arrived within ``watchdog_stale_seconds``.

        Returns ``False`` until the first frame and once the source goes stale.
        """
        if self._last_frame_ts is None:
            return False
        return (self._clock.now() - self._last_frame_ts) < self._watchdog_stale_seconds

    @property
    def fps_estimate(self) -> float:
        """Smoothed measured emit rate; ``0.0`` until at least two frames emitted."""
        if not self._recent_intervals:
            return 0.0
        avg_interval = sum(self._recent_intervals) / len(self._recent_intervals)
        if avg_interval <= 0.0:
            return 0.0
        return 1.0 / avg_interval

    @property
    def last_frame_ts(self) -> float | None:
        """Epoch seconds of the most recently emitted frame, or ``None``."""
        return self._last_frame_ts

    def _record_emit_interval(self, now: float) -> None:
        """Track inter-emit intervals for the rolling fps estimate."""
        if self._last_emit_ts is not None:
            interval = now - self._last_emit_ts
            if interval > 0.0:
                self._recent_intervals.append(interval)
                if len(self._recent_intervals) > _FPS_WINDOW:
                    self._recent_intervals.pop(0)


class FrameSourceCaptureThread(threading.Thread):
    """Pump a finite/streaming :class:`FrameSource` into a queue (non-RTSP).

    Mirrors :class:`RtspCaptureThread`'s control surface so a ``CameraPipeline``
    can run over a recorded clip or synthetic frames with no camera. Reads packets
    from the source and enqueues them until the source is exhausted (``read()``
    returns ``None``) or :meth:`stop` is called.
    """

    def __init__(
        self,
        source: FrameSource,
        queue: BoundedFrameQueue,
        clock: Clock,
        stale_after_s: float = 5.0,
    ) -> None:
        super().__init__(name="frame-source-capture", daemon=True)
        self._source = source
        self._queue = queue
        self._clock = clock
        self._stale_after_s = stale_after_s
        self._stop_event = threading.Event()
        self._last_frame_ts: float | None = None
        self._emitted = 0

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                packet = self._source.read()
                if packet is None:
                    break  # source exhausted
                self._last_frame_ts = packet.ts
                self._emitted += 1
                self._queue.put(packet)
        finally:
            self._source.close()

    def stop(self) -> None:
        self._stop_event.set()

    def is_healthy(self) -> bool:
        return self._last_frame_ts is not None

    @property
    def fps_estimate(self) -> float:
        return 0.0  # recorded sources have no meaningful real-time rate

    @property
    def last_frame_ts(self) -> float | None:
        return self._last_frame_ts


class RtspCaptureThread(_CaptureHealthMixin, threading.Thread):
    """Background thread that decodes one RTSP stream into a frame queue.

    The thread is fault-tolerant: read/open failures are logged and retried with
    exponential backoff rather than propagated. Call :meth:`stop` to request a
    graceful shutdown and join the thread.
    """

    def __init__(
        self,
        camera_id: int,
        url: str,
        target_fps: float,
        processing_cfg: ProcessingConfig,
        clock: Clock,
        queue: BoundedFrameQueue,
        role: CameraRole,
    ) -> None:
        """Configure the capture thread (does not start it).

        Args:
            camera_id: Database id of the camera, stamped onto every packet.
            url: Full RTSP URL including credentials. Never logged directly;
                always redacted via :func:`~app.ingestion.stream_url.redact`.
            target_fps: Desired sampling rate. Frames arriving faster than this
                are dropped at decode time so downstream sees a steady cadence.
                Values <= 0 disable downsampling (every decoded frame is kept).
            processing_cfg: Reconnect/watchdog tuning.
            clock: Time source; ``clock.now()`` stamps each packet's ``ts``.
            queue: Drop-oldest queue the decoded frames are pushed into.
            role: Camera role tag carried on each packet.
        """
        super().__init__(name=f"rtsp-capture-{camera_id}", daemon=True)
        self._camera_id = camera_id
        self._url = url
        self._redacted_url = redact(url)
        self._target_fps = target_fps
        self._min_interval = 1.0 / target_fps if target_fps > 0 else 0.0
        self._cfg = processing_cfg
        self._watchdog_stale_seconds = processing_cfg.watchdog_stale_seconds
        self._clock = clock
        self._queue = queue
        self._role = role

        self._stop_event = threading.Event()
        self._frame_idx = 0
        self._last_frame_ts: float | None = None
        self._last_emit_ts: float | None = None
        self._recent_intervals: list[float] = []

    # ------------------------------------------------------------------ #
    # Public control / introspection
    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Signal the capture loop to exit at the next opportunity.

        Idempotent; safe to call from any thread. Does not join — call
        :meth:`~threading.Thread.join` afterwards if you need to wait.
        """
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Thread body
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Decode loop: open the stream, read frames, sample, push, reconnect.

        Runs until :meth:`stop` is called. All cv2/read errors are caught,
        logged (with the redacted URL) and recovered from via backoff; the thread
        never crashes on a stream fault.
        """
        import cv2  # lazy: heavy/optional dependency (HLD constraint)

        backoff = self._cfg.reconnect_backoff_base_s
        capture = None
        try:
            while not self._stop_event.is_set():
                if capture is None:
                    capture = self._open_capture(cv2)
                    if capture is None:
                        backoff = self._sleep_backoff(backoff)
                        continue
                    # Reset backoff after a successful (re)connect.
                    backoff = self._cfg.reconnect_backoff_base_s

                ok, frame = capture.read()
                if not ok or frame is None:
                    logger.warning(
                        "rtsp read failed; reconnecting",
                        extra={"camera_id": self._camera_id, "event": "rtsp_read_failed"},
                    )
                    capture = self._release(capture)
                    backoff = self._sleep_backoff(backoff)
                    continue

                self._handle_frame(frame)
        finally:
            self._release(capture)
            logger.info(
                "rtsp capture stopped",
                extra={"camera_id": self._camera_id, "event": "rtsp_capture_stopped"},
            )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _open_capture(self, cv2_module: object):
        """Open the RTSP stream over TCP, returning the handle or ``None``.

        Configures the FFmpeg backend to use TCP transport (more reliable than
        UDP for analytics) before opening. On failure the handle is released and
        ``None`` is returned so the caller can back off and retry.

        Args:
            cv2_module: The lazily-imported ``cv2`` module.

        Returns:
            An opened ``cv2.VideoCapture`` handle, or ``None`` if it could not be
            opened.
        """
        cv2 = cv2_module  # local alias for readability
        # Force TCP for the FFmpeg RTSP demuxer; UDP drops frames on congested
        # links and corrupts H.264 GOPs. Honour configured transport, default tcp.
        transport = self._cfg.rtsp_transport or "tcp"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        try:
            capture = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        except cv2.error as exc:  # type: ignore[attr-defined]
            logger.warning(
                "rtsp open raised: %s",
                exc,
                extra={"camera_id": self._camera_id, "event": "rtsp_open_error"},
            )
            return None

        if not capture.isOpened():
            logger.warning(
                "rtsp open failed for %s",
                self._redacted_url,
                extra={"camera_id": self._camera_id, "event": "rtsp_open_failed"},
            )
            capture.release()
            return None

        logger.info(
            "rtsp connected to %s",
            self._redacted_url,
            extra={"camera_id": self._camera_id, "event": "rtsp_connected"},
        )
        return capture

    def _handle_frame(self, frame: np.ndarray) -> None:
        """Stamp, sample and enqueue a freshly decoded frame.

        Frames arriving faster than ``target_fps`` are dropped here so the queue
        carries a steady cadence regardless of the source's native rate.

        Args:
            frame: The decoded BGR image from OpenCV.
        """
        now = self._clock.now()
        self._last_frame_ts = now

        if (
            self._min_interval > 0.0
            and self._last_emit_ts is not None
            and (now - self._last_emit_ts) < self._min_interval
        ):
            return  # downsample: too soon since last emitted frame

        self._record_emit_interval(now)
        self._last_emit_ts = now

        packet = FramePacket(
            camera_id=self._camera_id,
            frame_idx=self._frame_idx,
            ts=now,
            image=frame,
            role=self._role,
        )
        self._frame_idx += 1
        self._queue.put(packet)

    def _sleep_backoff(self, backoff: float) -> float:
        """Wait ``backoff`` seconds (interruptible) and return the next backoff.

        The wait is implemented with the stop event so :meth:`stop` aborts it
        immediately instead of sleeping out the full interval.

        Args:
            backoff: Seconds to wait before the next reconnect attempt.

        Returns:
            The next (doubled, capped) backoff value.
        """
        # Event.wait returns True if the event was set during the wait — we let
        # the outer loop observe the stop on its next iteration either way.
        self._stop_event.wait(timeout=backoff)
        return min(backoff * 2.0, self._cfg.reconnect_backoff_max_s)

    def _release(self, capture: object | None) -> None:
        """Release an OpenCV capture handle, swallowing only release errors.

        Args:
            capture: The handle to release, or ``None``.

        Returns:
            ``None`` always, so callers can write ``capture = self._release(capture)``.
        """
        if capture is None:
            return None
        try:
            capture.release()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            # Broad catch is deliberate (clean-code imperative 15 exception):
            # cv2's release() can raise backend-specific (FFmpeg) errors with no
            # stable Python type to target, and a cleanup failure must never
            # block shutdown or reconnect. We log and continue — we never return
            # a fake success — so the failure stays observable.
            logger.debug(
                "capture release failed: %s",
                exc,
                extra={"camera_id": self._camera_id, "event": "rtsp_release_error"},
            )
        return None


class SnapshotCaptureThread(_CaptureHealthMixin, threading.Thread):
    """Poll a still image every ``interval_s`` and push it as a frame (HLD 6.1).

    The CPU-cheap alternative to :class:`RtspCaptureThread` for low-rate roles
    (e.g. occupancy) on GPU-less hardware: instead of continuously decoding an
    H.264 sub-stream, it fetches one JPEG per interval via an injected
    ``frame_provider`` (an HTTP snapshot client + decoder wired in the engine),
    so decode cost scales with the pull rate, not the stream's native fps.

    Resilience mirrors the RTSP contract: a failed or empty fetch is logged and
    skipped — the thread never exits on a transient fault, so the engine
    supervisor never needs to restart it and staleness surfaces only via
    :meth:`is_healthy` / ``CameraHealth``. A ``frame_provider`` returning
    ``None`` is a skipped tick, **not** source exhaustion.
    """

    def __init__(
        self,
        *,
        camera_id: int,
        frame_provider: Callable[[], np.ndarray | None],
        queue: BoundedFrameQueue,
        clock: Clock,
        interval_s: float,
        watchdog_stale_seconds: float,
        role: CameraRole,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        """Configure the thread (does not start it).

        Args:
            camera_id: Camera id stamped onto every emitted packet.
            frame_provider: Zero-arg callable returning a freshly fetched BGR
                image, or ``None`` when this tick's fetch/decode failed.
            queue: Drop-oldest queue the frames are pushed into.
            clock: Time source; ``clock.now()`` stamps each packet.
            interval_s: Seconds between pull attempts (the effective frame rate).
            watchdog_stale_seconds: Age after which ``is_healthy`` reports stale.
            role: Camera role tag carried on each packet.
            on_close: Optional cleanup called once when the loop exits — used to
                release the frame provider's resources (e.g. the HTTP client) so a
                pipeline restart does not leak them.
        """
        super().__init__(name=f"snapshot-capture-{camera_id}", daemon=True)
        self._camera_id = camera_id
        self._frame_provider = frame_provider
        self._queue = queue
        self._clock = clock
        self._interval_s = max(interval_s, 0.0)
        self._watchdog_stale_seconds = watchdog_stale_seconds
        self._role = role
        self._on_close = on_close

        self._stop_event = threading.Event()
        self._frame_idx = 0
        self._last_frame_ts: float | None = None
        self._last_emit_ts: float | None = None
        self._recent_intervals: list[float] = []

    def stop(self) -> None:
        """Signal the poll loop to exit at the next opportunity. Idempotent."""
        self._stop_event.set()

    def run(self) -> None:
        """Poll loop: fetch a frame, enqueue it, wait ``interval_s``, repeat.

        Runs until :meth:`stop`. Every fetch fault is caught inside
        :meth:`_capture_once`, so the loop only ends on the stop signal. The
        ``on_close`` cleanup runs once on exit, however the loop ends.
        """
        try:
            while not self._stop_event.is_set():
                self._capture_once()
                # Interruptible wait: stop() aborts it immediately.
                if self._stop_event.wait(timeout=self._interval_s):
                    break
        finally:
            if self._on_close is not None:
                self._on_close()

    def _capture_once(self) -> None:
        """Fetch one still and enqueue it; a failed fetch is logged and skipped.

        On success the frame timestamp and fps/health bookkeeping are updated; on
        ``None`` or an exception nothing is emitted, so ``is_healthy`` goes stale
        via the watchdog rather than the thread dying.
        """
        try:
            image = self._frame_provider()
        except Exception:  # noqa: BLE001 - a provider fault must not kill the loop
            logger.exception(
                "snapshot fetch raised; skipping tick",
                extra={"camera_id": self._camera_id, "event": "snapshot_fetch_error"},
            )
            return

        if image is None:
            logger.warning(
                "snapshot fetch returned no image; skipping tick",
                extra={"camera_id": self._camera_id, "event": "snapshot_fetch_empty"},
            )
            return

        now = self._clock.now()
        self._record_emit_interval(now)
        self._last_frame_ts = now
        self._last_emit_ts = now
        self._queue.put(
            FramePacket(
                camera_id=self._camera_id,
                frame_idx=self._frame_idx,
                ts=now,
                image=image,
                role=self._role,
            )
        )
        self._frame_idx += 1
