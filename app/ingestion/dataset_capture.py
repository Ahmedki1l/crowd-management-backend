"""Persist fetched camera frames to disk for building a training dataset.

Opt-in via ``dataset_capture.enabled``. Wired into
:class:`~app.engine.camera_pipeline.CameraPipeline`, it saves each frame under
``<dir>/<camera>/<date>/`` two ways:

* **Snapshot-pull (ISAPI/occupancy) cameras** — :meth:`save` writes the *original*
  JPEG bytes, byte-for-byte what the camera sent (no decode/re-encode).
* **RTSP (e.g. the entry/exit door) cameras** — frames arrive already decoded, so
  :meth:`save_image` JPEG-encodes the frame before writing.

A failed write is logged and swallowed: collecting training data must never
disturb the live pipeline (the same "one bad frame never kills the thread"
contract as the capture layer).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only; cv2 handles the array at runtime
    import numpy as np

logger = get_logger(__name__)


class DatasetWriter:
    """Writes raw snapshot JPEG bytes to a per-camera, per-day folder tree.

    Thread-safe: one instance may be shared across camera threads, though the
    pipeline builds one per camera. ``min_interval_s`` caps the save rate per
    camera (0 disables the cap, saving every fetched frame).
    """

    def __init__(self, base_dir: str, min_interval_s: float = 0.0) -> None:
        """Configure the writer (creates no directories until the first save).

        Args:
            base_dir: Root directory the ``<camera>/<date>/`` tree is written under.
            min_interval_s: Minimum seconds between saved frames per camera; 0
                saves every fetched frame.
        """
        self._base = Path(base_dir)
        self._min_interval_s = max(min_interval_s, 0.0)
        self._last_saved: dict[int, float] = {}
        self._lock = threading.Lock()

    def save(self, camera_id: int, ip: str, jpeg: bytes, ts: float) -> None:
        """Write ``jpeg`` for ``camera_id`` unless throttled; never raises.

        Args:
            camera_id: Camera the snapshot came from (folder + throttle key).
            ip: Camera IP, used in the folder name and filename.
            jpeg: Original JPEG bytes as fetched over ISAPI.
            ts: Epoch seconds the frame was fetched (from the pipeline clock).
        """
        if not self._throttled(camera_id, ts):
            self._persist(camera_id, ip, jpeg, ts)

    def save_image(
        self, camera_id: int, ip: str, image: np.ndarray, ts: float, quality: int = 95
    ) -> None:
        """JPEG-encode a decoded frame and save it (for RTSP sources); never raises.

        Args:
            camera_id: Camera the frame came from (folder + throttle key).
            ip: Camera IP, used in the folder name and filename.
            image: Decoded BGR frame (HxWx3) as produced by the RTSP capture.
            ts: Epoch seconds the frame was captured (from the pipeline clock).
            quality: JPEG quality (0-100) for the encode.
        """
        if self._throttled(camera_id, ts):
            return
        import cv2  # lazy: heavy/optional dependency (HLD constraint)

        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            self._persist(camera_id, ip, buf.tobytes(), ts)

    def _persist(self, camera_id: int, ip: str, jpeg: bytes, ts: float) -> None:
        """Write ``jpeg`` to the camera/day folder, swallowing OS errors."""
        try:
            self._write(camera_id, ip, jpeg, ts)
        except OSError as exc:
            logger.warning(
                "dataset frame write failed: %s",
                exc,
                extra={"camera_id": camera_id, "event": "dataset_write_error"},
            )

    def _throttled(self, camera_id: int, ts: float) -> bool:
        """Return whether this save is inside the per-camera min interval."""
        if self._min_interval_s <= 0.0:
            return False
        with self._lock:
            last = self._last_saved.get(camera_id)
            if last is not None and (ts - last) < self._min_interval_s:
                return True
            self._last_saved[camera_id] = ts
            return False

    def _write(self, camera_id: int, ip: str, jpeg: bytes, ts: float) -> None:
        """Create the day folder and write the JPEG under a timestamped name."""
        gm = time.gmtime(ts)
        day = time.strftime("%Y-%m-%d", gm)
        safe_ip = ip.replace(":", "_")
        folder = self._base / f"cam{camera_id}_{safe_ip}" / day
        folder.mkdir(parents=True, exist_ok=True)
        micros = int((ts % 1.0) * 1_000_000)
        name = f"{safe_ip}_{time.strftime('%Y%m%d_%H%M%S', gm)}_{micros:06d}.jpg"
        (folder / name).write_bytes(jpeg)
