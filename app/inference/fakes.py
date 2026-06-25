"""Deterministic fake perception backends for unit/integration tests.

None of these import heavy dependencies. They implement the same protocols as
the real backends (:class:`~app.domain.interfaces.Detector`,
:class:`~app.domain.interfaces.Tracker`,
:class:`~app.domain.interfaces.EmbeddingExtractor`) so the engine and analytics
can be exercised end-to-end without a model runtime installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from app.domain.models import BBox, Detection, TrackedDetection

# A script is either a per-call list of detections, or a function of the call
# index returning the detections for that call.
DetectionScript = list[list[Detection]] | Callable[[int], list[Detection]]


class FakeDetector:
    """Replays a scripted sequence of detections, one entry per ``detect`` call.

    Implements the :class:`~app.domain.interfaces.Detector` protocol.
    """

    def __init__(self, script: DetectionScript) -> None:
        """Configure the detector with a detection script.

        Args:
            script: Either a list whose ``i``-th element is the detections to
                return on call ``i`` (clamped to the last entry once exhausted),
                or a callable mapping a call index to detections.
        """
        self._script = script
        self._call_index = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return the scripted detections for the current call index."""
        detections = self._detections_for(self._call_index)
        self._call_index += 1
        return detections

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Return one scripted detection list per image, advancing per image."""
        return [self.detect(image) for image in images]

    def _detections_for(self, index: int) -> list[Detection]:
        """Resolve the scripted detections for ``index``."""
        if callable(self._script):
            return list(self._script(index))

        if not self._script:
            return []
        clamped = min(index, len(self._script) - 1)
        return list(self._script[clamped])


def _iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes (0.0 when they do not overlap)."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


class FakeTracker:
    """Deterministic greedy-IoU tracker assigning incremental integer track IDs.

    Implements the :class:`~app.domain.interfaces.Tracker` protocol. On each
    frame, detections are greedily matched (highest IoU first) to the previous
    frame's tracks above ``iou_threshold``; unmatched detections start new
    tracks with monotonically increasing IDs.
    """

    def __init__(self, iou_threshold: float = 0.3) -> None:
        """Configure the tracker.

        Args:
            iou_threshold: Minimum IoU for a detection to continue an existing
                track.
        """
        self._iou_threshold = iou_threshold
        self._next_track_id = 1
        # track_id -> last bbox seen for that track.
        self._tracks: dict[int, BBox] = {}

    def update(
        self, detections: Sequence[Detection], image: np.ndarray | None = None
    ) -> list[TrackedDetection]:
        """Associate ``detections`` with prior tracks by greedy IoU.

        Args:
            detections: Per-frame detections.
            image: Unused (kept for protocol compatibility).

        Returns:
            One :class:`TrackedDetection` per input detection, in input order.
        """
        prior = self._tracks
        assignments = self._greedy_match(detections, prior)

        new_tracks: dict[int, BBox] = {}
        results: list[TrackedDetection] = []
        for idx, detection in enumerate(detections):
            track_id = assignments.get(idx)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
            new_tracks[track_id] = detection.bbox
            results.append(
                TrackedDetection(
                    track_id=track_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                )
            )

        self._tracks = new_tracks
        return results

    def reset(self) -> None:
        """Drop all track state and restart ID allocation from 1."""
        self._tracks = {}
        self._next_track_id = 1

    def _greedy_match(
        self, detections: Sequence[Detection], prior: dict[int, BBox]
    ) -> dict[int, int]:
        """Greedily match detection indices to prior track ids by IoU.

        Returns a mapping ``detection_index -> track_id`` for matched pairs only.
        Candidate pairs are sorted by descending IoU; each detection and each
        track is consumed at most once, giving a deterministic assignment.
        """
        candidates: list[tuple[float, int, int]] = []
        for det_idx, detection in enumerate(detections):
            for track_id, last_bbox in prior.items():
                overlap = _iou(detection.bbox, last_bbox)
                if overlap >= self._iou_threshold:
                    candidates.append((overlap, det_idx, track_id))

        # Sort by IoU desc, then by indices for a stable, deterministic order.
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _overlap, det_idx, track_id in candidates:
            if det_idx in assignments or track_id in used_tracks:
                continue
            assignments[det_idx] = track_id
            used_tracks.add(track_id)
        return assignments


class FakeEmbeddingExtractor:
    """Deterministic pseudo-embedding extractor derived from bbox coordinates.

    Implements the :class:`~app.domain.interfaces.EmbeddingExtractor` protocol.
    The same box always yields the same L2-normalised vector, so two cameras
    seeing identical boxes produce matching embeddings — exactly what Re-ID
    tests need.
    """

    def __init__(self, dim: int = 128) -> None:
        """Configure the embedding width.

        Args:
            dim: Embedding dimensionality (must be positive).

        Raises:
            ValueError: If ``dim`` is not positive.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        """Embedding dimensionality produced by :meth:`extract`."""
        return self._dim

    def extract(
        self,
        image: np.ndarray,
        bboxes: Sequence[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Return one deterministic L2-normalised embedding per box.

        Args:
            image: Unused (kept for protocol compatibility).
            bboxes: ``(x1, y1, x2, y2)`` boxes to embed.

        Returns:
            ``(len(bboxes), dim)`` float32 array of unit-norm embeddings.
        """
        if not bboxes:
            return np.empty((0, self._dim), dtype=np.float32)

        rows = [self._embed_bbox(bbox) for bbox in bboxes]
        matrix = np.stack(rows, axis=0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (matrix / norms).astype(np.float32)

    def _embed_bbox(self, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Seed a deterministic vector from a box's coordinates."""
        seed = abs(hash(tuple(round(float(c), 3) for c in bbox))) % (2**32)
        generator = np.random.default_rng(seed)
        return generator.standard_normal(self._dim)
