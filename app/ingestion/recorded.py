"""Recorded / synthetic frame source for integration tests (HLD 12 — testing).

:class:`RecordedClipSource` implements the
:class:`~app.domain.interfaces.FrameSource` protocol so the full inference and
analytics pipeline can be driven over a deterministic clip without any live
RTSP camera. It has two construction modes:

* ``frames=[...]`` — an in-memory list of ``numpy`` arrays. Requires NO heavy
  dependency (no cv2), so unit/integration tests run on core deps alone.
* ``path="clip.mp4"`` — a video file decoded with OpenCV. ``cv2`` is imported
  lazily on first :meth:`read`, keeping this module importable without it.

Timestamps are synthetic and monotonic: frame ``i`` is stamped
``base_ts + i / fps``. This makes dwell/debounce assertions reproducible
regardless of how fast the test consumes frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.domain.models import CameraRole, FramePacket
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

logger = get_logger(__name__)


class RecordedClipSource:
    """A :class:`~app.domain.interfaces.FrameSource` backed by a clip or array list.

    Exactly one of ``path`` or ``frames`` must be supplied. Frames are emitted in
    order, each with a synthetic timestamp advancing by ``1 / fps`` from
    ``base_ts``. :meth:`read` returns ``None`` once the clip is exhausted.
    """

    def __init__(
        self,
        camera_id: int,
        fps: float,
        role: CameraRole = CameraRole.OCCUPANCY,
        *,
        path: str | None = None,
        frames: list[np.ndarray] | None = None,
        base_ts: float = 0.0,
    ) -> None:
        """Configure the source.

        Args:
            camera_id: Camera id stamped onto every emitted packet.
            fps: Frame rate used to compute synthetic timestamps. Must be > 0.
            role: Camera role tag carried on each packet.
            path: Path to a video file to decode lazily with OpenCV. Mutually
                exclusive with ``frames``.
            frames: In-memory list of BGR image arrays. Mutually exclusive with
                ``path``; uses no heavy dependency.
            base_ts: Epoch seconds assigned to frame 0; subsequent frames advance
                by ``1 / fps``.

        Raises:
            ValueError: If neither or both of ``path``/``frames`` are given, or if
                ``fps`` is not positive.
        """
        if (path is None) == (frames is None):
            raise ValueError("exactly one of 'path' or 'frames' must be provided")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")

        self._camera_id = camera_id
        self._fps = fps
        self._role = role
        self._base_ts = base_ts
        self._frame_idx = 0
        self._closed = False

        # In-memory mode: hold the frames directly, no cv2.
        self._frames: list[np.ndarray] | None = list(frames) if frames is not None else None

        # File mode: defer opening until the first read so importing/constructing
        # without cv2 installed stays possible (capture is the only cv2 user).
        self._path = path
        self._capture: object | None = None
        self._cv2_exhausted = False

    def read(self) -> FramePacket | None:
        """Return the next frame as a :class:`FramePacket`, or ``None`` at end.

        Returns:
            The next packet with a synthetic monotonic timestamp, or ``None`` when
            the clip is exhausted or the source has been closed.
        """
        if self._closed:
            return None
        if self._frames is not None:
            return self._read_from_memory()
        return self._read_from_file()

    def close(self) -> None:
        """Release any underlying resources. Idempotent.

        After ``close`` further :meth:`read` calls return ``None``.
        """
        self._closed = True
        capture = self._capture
        self._capture = None
        if capture is None:
            return
        try:
            capture.release()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            # Broad catch is deliberate (clean-code imperative 15 exception):
            # cv2 release() raises backend-specific errors with no stable Python
            # type, and a close() failure must not propagate. Logged, not hidden.
            logger.debug(
                "recorded clip release failed: %s",
                exc,
                extra={"camera_id": self._camera_id, "event": "recorded_release_error"},
            )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _read_from_memory(self) -> FramePacket | None:
        """Emit the next in-memory frame, or ``None`` when the list is exhausted."""
        assert self._frames is not None  # narrowed by caller
        if self._frame_idx >= len(self._frames):
            return None
        image = self._frames[self._frame_idx]
        return self._build_packet(image)

    def _read_from_file(self) -> FramePacket | None:
        """Decode and emit the next frame from the video file, lazily opening it."""
        import cv2  # lazy: heavy/optional dependency (HLD constraint)

        if self._cv2_exhausted:
            return None
        if self._capture is None:
            self._capture = self._open_file(cv2)
            if self._capture is None:
                self._cv2_exhausted = True
                return None

        ok, frame = self._capture.read()  # type: ignore[attr-defined]
        if not ok or frame is None:
            self._cv2_exhausted = True
            return None
        return self._build_packet(frame)

    def _open_file(self, cv2_module: object) -> object | None:
        """Open the configured video file, returning the handle or ``None``.

        Args:
            cv2_module: The lazily-imported ``cv2`` module.

        Returns:
            An opened ``cv2.VideoCapture`` handle, or ``None`` if it could not be
            opened.
        """
        cv2 = cv2_module
        try:
            capture = cv2.VideoCapture(self._path)
        except cv2.error as exc:  # type: ignore[attr-defined]
            logger.error(
                "failed to open recorded clip %s: %s",
                self._path,
                exc,
                extra={"camera_id": self._camera_id, "event": "recorded_open_error"},
            )
            return None
        if not capture.isOpened():
            logger.error(
                "recorded clip not openable: %s",
                self._path,
                extra={"camera_id": self._camera_id, "event": "recorded_open_failed"},
            )
            capture.release()
            return None
        return capture

    def _build_packet(self, image: np.ndarray) -> FramePacket:
        """Wrap ``image`` in a packet with a synthetic timestamp and advance state."""
        ts = self._base_ts + self._frame_idx / self._fps
        packet = FramePacket(
            camera_id=self._camera_id,
            frame_idx=self._frame_idx,
            ts=ts,
            image=image,
            role=self._role,
        )
        self._frame_idx += 1
        return packet
