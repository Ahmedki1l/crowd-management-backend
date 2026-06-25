"""Perception-layer contracts (Protocols).

These abstractions let the engine, analytics, and tests depend on *behaviour*
rather than concrete heavy backends. Real implementations (Ultralytics/OpenVINO/
TensorRT detector, ByteTrack tracker, OSNet embedding extractor) import their
heavy dependencies lazily; fakes used in tests implement the same Protocols with
no heavy deps.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from app.domain.models import Detection, FramePacket, TrackedDetection


@runtime_checkable
class Clock(Protocol):
    """Time source — abstracted so dwell/debounce tests are deterministic."""

    def now(self) -> float:
        """Epoch seconds."""
        ...


@runtime_checkable
class FrameSource(Protocol):
    """A source of decoded frames for one camera (RTSP capture or a recorded clip)."""

    def read(self) -> FramePacket | None:
        """Return the next frame, or ``None`` when the source is exhausted/disconnected."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class Detector(Protocol):
    """Person detection on a single frame or a batch of frames."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        ...

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Detect over a batch; ``len(result) == len(images)``."""
        ...


@runtime_checkable
class Tracker(Protocol):
    """Multi-object tracker assigning stable track IDs across frames (per camera)."""

    def update(
        self, detections: Sequence[Detection], image: np.ndarray | None = None
    ) -> list[TrackedDetection]:
        ...

    def reset(self) -> None:
        """Drop all track state (e.g., after a long stream gap)."""
        ...


@runtime_checkable
class EmbeddingExtractor(Protocol):
    """Appearance embeddings for Re-ID. Returns an (N, D) L2-normalised array."""

    @property
    def dim(self) -> int:
        ...

    def extract(self, image: np.ndarray, bboxes: Sequence[tuple[float, float, float, float]]) -> np.ndarray:
        ...
