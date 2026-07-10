"""Per-round timing for the snapshot-pull occupancy path (diagnostics).

Each snapshot camera runs its own capture thread and worker, so a "round" — one
pull of every camera — exists only implicitly. This module reconstructs it: a
round opens when the first camera reports a processed frame and closes once every
expected camera has reported one. Closing logs the wall time from the *earliest
capture* in the round to the *latest finish*, which is what "how long until all
this round's images are processed" actually means.

The per-camera ``latency`` reported here is ``finish_ts - capture_ts``: it spans
queue wait + inference + publish, so a queue backlog shows up as latency far
larger than the detector time. That difference is the whole point — it separates
"the model is slow" from "the frame sat in a buffer".

Opt-in and zero-cost when off: the engine builds a timer only when
``PIPELINE_ROUND_TIMING`` is set (mirroring ``PIPELINE_TRACE_CAMERA``).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Sample:
    """One camera's contribution to a round."""

    capture_ts: float
    finish_ts: float
    detect_seconds: float
    detections: int

    @property
    def latency(self) -> float:
        """Capture -> processed. Includes queue wait, inference and publish."""
        return self.finish_ts - self.capture_ts


class RoundTimer:
    """Accumulates per-camera samples and logs one line per completed round.

    Thread-safe: every camera's worker thread calls :meth:`record` concurrently.

    A camera reporting twice before the round completes means the round can never
    complete (that camera has already moved on), so the partial round is closed
    and a new one opened with the late sample. Rounds are therefore never held
    open by a camera whose fetch is failing.
    """

    def __init__(self, expected_camera_ids: Iterable[int]) -> None:
        """Track a round across ``expected_camera_ids``.

        Args:
            expected_camera_ids: The cameras that must each report once before a
                round is considered complete. Empty disables the timer.
        """
        self._expected: frozenset[int] = frozenset(expected_camera_ids)
        self._lock = threading.Lock()
        self._current: dict[int, _Sample] = {}
        self._round = 0

    @property
    def expected(self) -> frozenset[int]:
        """The camera ids a complete round requires."""
        return self._expected

    def record(
        self,
        camera_id: int,
        capture_ts: float,
        finish_ts: float,
        detect_seconds: float,
        detections: int,
    ) -> None:
        """Register that ``camera_id``'s frame finished processing.

        Args:
            camera_id: The reporting camera.
            capture_ts: When the frame was fetched from the camera.
            finish_ts: When its analytics finished and events were published.
            detect_seconds: Time spent inside the detector for this frame.
            detections: Number of raw detections found.
        """
        if not self._expected:
            return
        with self._lock:
            if camera_id in self._current:
                self._close_locked(complete=False)
            self._current[camera_id] = _Sample(
                capture_ts=capture_ts,
                finish_ts=finish_ts,
                detect_seconds=detect_seconds,
                detections=detections,
            )
            if self._expected.issubset(self._current):
                self._close_locked(complete=True)

    def _close_locked(self, *, complete: bool) -> None:
        """Emit the round summary and reset. Caller holds the lock."""
        samples = self._current
        self._current = {}
        if not samples:
            return
        self._round += 1

        first_capture = min(s.capture_ts for s in samples.values())
        last_finish = max(s.finish_ts for s in samples.values())
        wall = last_finish - first_capture

        slowest_id, slowest = max(samples.items(), key=lambda kv: kv[1].latency)
        detect_total = sum(s.detect_seconds for s in samples.values())
        detections = sum(s.detections for s in samples.values())
        missing = sorted(self._expected - samples.keys())

        # The JSON formatter only forwards a fixed key whitelist, so the
        # per-camera breakdown has to live in the message to be visible at all.
        # Sorted slowest-first: the head of this list is the camera to look at.
        breakdown = " ".join(
            f"cam{cid}={s.latency:.2f}s/detect{s.detect_seconds:.2f}s"
            for cid, s in sorted(samples.items(), key=lambda kv: -kv[1].latency)
        )
        logger.info(
            "round %d %s: %d/%d cameras in %.2fs "
            "(detect %.2fs total, slowest cam %s @ %.2fs latency, %d detections)%s | %s",
            self._round,
            "complete" if complete else "PARTIAL",
            len(samples),
            len(self._expected),
            wall,
            detect_total,
            slowest_id,
            slowest.latency,
            detections,
            f" missing={missing}" if missing else "",
            breakdown,
            extra={
                "event": "round_complete" if complete else "round_partial",
                "round": self._round,
                "cameras_reported": len(samples),
                "cameras_expected": len(self._expected),
                "wall_seconds": round(wall, 4),
                "first_capture_ts": first_capture,
                "last_finish_ts": last_finish,
                "detect_seconds_total": round(detect_total, 4),
                "detect_seconds_max": round(max(s.detect_seconds for s in samples.values()), 4),
                "latency_seconds_max": round(slowest.latency, 4),
                "latency_seconds_min": round(min(s.latency for s in samples.values()), 4),
                "slowest_camera_id": slowest_id,
                "detections_total": detections,
                "missing_camera_ids": missing,
                "per_camera": {
                    str(cid): {
                        "latency_s": round(s.latency, 4),
                        "detect_s": round(s.detect_seconds, 4),
                        "detections": s.detections,
                    }
                    for cid, s in sorted(samples.items())
                },
            },
        )
