"""Multi-object tracking backend (HLD 6.2).

:class:`ByteTrackTracker` wraps the ByteTrack implementation shipped with the
``supervision`` library. ``supervision`` is imported lazily inside the
constructor so this module imports cleanly without it installed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.config.schema import TrackerConfig
from app.domain.interfaces import Tracker
from app.domain.models import BBox, Detection, TrackedDetection
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ByteTrackTracker:
    """ByteTrack tracker implementing the :class:`Tracker` protocol.

    Heavy imports (``supervision``) happen inside ``__init__`` so importing this
    module never requires the tracking library to be installed.
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        """Initialise ByteTrack from ``cfg``.

        Args:
            cfg: Tracker configuration. ``track_thresh``, ``match_thresh``,
                ``track_buffer`` and ``frame_rate`` are forwarded to ByteTrack.
        """
        self._cfg = cfg
        self._tracker = self._new_tracker()

    def _new_tracker(self) -> object:
        """Build a fresh ``supervision`` ByteTrack instance from the config."""
        import supervision as sv  # lazy: heavy optional dependency

        return sv.ByteTrack(
            track_activation_threshold=self._cfg.track_thresh,
            minimum_matching_threshold=self._cfg.match_thresh,
            lost_track_buffer=self._cfg.track_buffer,
            frame_rate=self._cfg.frame_rate,
        )

    def update(
        self, detections: Sequence[Detection], image: np.ndarray | None = None
    ) -> list[TrackedDetection]:
        """Associate ``detections`` against existing tracks for the next frame.

        Args:
            detections: Per-frame detections from the detector.
            image: Unused by ByteTrack (kept for protocol compatibility with
                appearance-based trackers).

        Returns:
            Tracked detections; only detections ByteTrack matched to a track are
            returned, each carrying its ``track_id``.
        """
        import supervision as sv  # lazy: heavy optional dependency

        sv_detections = self._to_sv_detections(detections, sv)
        tracked = self._tracker.update_with_detections(sv_detections)
        return self._from_sv_detections(tracked)

    def reset(self) -> None:
        """Drop all track state and reset ID allocation."""
        # ``sv.ByteTrack`` exposes ``reset``; rebuild defensively if it does not.
        reset = getattr(self._tracker, "reset", None)
        if callable(reset):
            reset()
        else:
            self._tracker = self._new_tracker()

    @staticmethod
    def _to_sv_detections(detections: Sequence[Detection], sv: object) -> object:
        """Convert domain :class:`Detection` objects to ``sv.Detections``."""
        if not detections:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox.as_xyxy() for d in detections], dtype=np.float64)
        confidence = np.array([d.confidence for d in detections], dtype=np.float64)
        class_id = np.array([d.class_id for d in detections], dtype=np.int64)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    @staticmethod
    def _from_sv_detections(tracked: object) -> list[TrackedDetection]:
        """Map tracked ``sv.Detections`` back to :class:`TrackedDetection`."""
        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None or len(tracked) == 0:
            return []

        results: list[TrackedDetection] = []
        for idx in range(len(tracked)):
            track_id = tracker_ids[idx]
            if track_id is None:
                continue
            x1, y1, x2, y2 = tracked.xyxy[idx]
            confidence = (
                float(tracked.confidence[idx])
                if tracked.confidence is not None
                else 0.0
            )
            class_id = (
                int(tracked.class_id[idx]) if tracked.class_id is not None else 0
            )
            results.append(
                TrackedDetection(
                    track_id=int(track_id),
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                    confidence=confidence,
                    class_id=class_id,
                )
            )
        return results


def build_tracker(cfg: TrackerConfig) -> Tracker:
    """Construct the tracker backend for ``cfg``.

    Args:
        cfg: Tracker configuration.

    Returns:
        A :class:`Tracker` (currently always a :class:`ByteTrackTracker`).
    """
    return ByteTrackTracker(cfg)
