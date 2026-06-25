"""Person detection backends (HLD 6.2).

The real backend (:class:`YoloDetector`) wraps Ultralytics YOLO and lazily
imports the heavy ``ultralytics`` package only when a detector is actually
constructed, so this module imports cleanly on the core dependency set used by
the API/tests.

Runtimes
--------
* ``openvino`` — CPU inference with an OpenVINO INT8-quantised export. Selected
  for CPU-only deployments.
* ``tensorrt`` — GPU inference with a TensorRT FP16 engine. Selected when a
  CUDA device is available.

Both runtimes are handed to Ultralytics as an exported model *format* plus a
``device`` hint; Ultralytics loads the matching artefact under
``cfg.model_path``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.config.schema import DetectorConfig
from app.domain.interfaces import Detector
from app.domain.models import BBox, Detection
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Maps the configured runtime to the (Ultralytics export ``format``, ``device``)
# pair. OpenVINO runs INT8 on CPU; TensorRT runs FP16 on GPU.
_RUNTIME_HINTS: dict[str, tuple[str, str]] = {
    "openvino": ("openvino", "cpu"),
    "tensorrt": ("engine", "cuda:0"),
}


class YoloDetector:
    """Ultralytics YOLO person detector implementing the :class:`Detector` protocol.

    Heavy imports (``ultralytics``) happen inside ``__init__`` so importing this
    module never requires the model runtime to be installed.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        """Load YOLO weights for ``cfg.model_path`` using the configured runtime.

        Args:
            cfg: Detector configuration. ``cfg.runtime`` must be one of
                ``"openvino"`` (CPU INT8) or ``"tensorrt"`` (GPU FP16).

        Raises:
            ValueError: If ``cfg.runtime`` is not a known runtime.
        """
        runtime = cfg.runtime.lower()
        if runtime not in _RUNTIME_HINTS:
            raise ValueError(
                f"Unknown detector runtime {cfg.runtime!r}; "
                f"expected one of {sorted(_RUNTIME_HINTS)}"
            )

        self._cfg = cfg
        self._fmt, self._device = _RUNTIME_HINTS[runtime]

        from ultralytics import YOLO  # lazy: heavy optional dependency

        logger.info(
            "Loading YOLO detector",
            extra={"event": "detector.load"},
        )
        # ``task="detect"`` keeps Ultralytics from re-deriving the task from a
        # bare exported artefact (e.g. an OpenVINO directory or .engine file).
        self._model = YOLO(cfg.model_path, task="detect")

    @property
    def cfg(self) -> DetectorConfig:
        """The configuration this detector was built from."""
        return self._cfg

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect people in a single frame.

        Args:
            image: HxWx3 BGR/RGB frame as a numpy array.

        Returns:
            Person detections passing ``cfg.confidence``, ordered as returned by
            the model.
        """
        return self.detect_batch([image])[0]

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Detect people over a batch of frames.

        Args:
            images: Sequence of frames.

        Returns:
            One detection list per input image; ``len(result) == len(images)``.
        """
        if not images:
            return []

        results = self._model.predict(
            list(images),
            conf=self._cfg.confidence,
            iou=self._cfg.iou,
            imgsz=self._cfg.imgsz,
            classes=[self._cfg.person_class],
            device=self._device,
            verbose=False,
        )
        return [self._to_detections(result) for result in results]

    def _to_detections(self, result: object) -> list[Detection]:
        """Map one Ultralytics ``Results`` object to a list of :class:`Detection`.

        Detections are filtered to ``cfg.person_class`` and ``cfg.confidence``;
        the ``classes``/``conf`` predict arguments already do this, but the
        explicit filter guards against backends that ignore those hints.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        # ``.cpu().numpy()`` works for both torch and OpenVINO/TensorRT tensors.
        xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64)
        confs = np.asarray(boxes.conf.cpu().numpy(), dtype=np.float64).ravel()
        class_ids = np.asarray(boxes.cls.cpu().numpy(), dtype=np.int64).ravel()

        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids, strict=False):
            if int(class_id) != self._cfg.person_class:
                continue
            if conf < self._cfg.confidence:
                continue
            detections.append(
                Detection(
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(class_id),
                )
            )
        return detections


def build_detector(cfg: DetectorConfig) -> Detector:
    """Construct the detector backend for ``cfg``.

    Args:
        cfg: Detector configuration.

    Returns:
        A :class:`Detector` (currently always a :class:`YoloDetector`).

    Raises:
        ValueError: If ``cfg.runtime`` is unknown (raised by the backend).
    """
    return YoloDetector(cfg)
