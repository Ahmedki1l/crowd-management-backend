"""Appearance Re-ID: embedding extraction and cross-camera identity (HLD 6.2).

Two collaborators live here:

* :class:`OSNetEmbeddingExtractor` — the heavy backend that turns person crops
  into L2-normalised appearance vectors. It lazily imports its runtime
  (``onnxruntime`` or ``torch``/``torchreid``) plus an image library
  (``cv2``/Pillow) only when constructed/used.
* :class:`ReIDManager` — a **pure numpy** gallery that maps per-camera track
  embeddings to stable cross-camera global IDs by cosine similarity. It has no
  heavy dependencies and is fully unit-testable with synthetic embeddings.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Sequence

import numpy as np

from app.config.schema import TrackerConfig
from app.domain.interfaces import Clock, EmbeddingExtractor
from app.domain.models import TrackedDetection
from app.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_REID_RUNTIMES = ("onnx", "torch")


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm.

    Rows with zero norm are left as zeros (their similarity to anything is 0).
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe


class OSNetEmbeddingExtractor:
    """OSNet appearance-embedding extractor implementing :class:`EmbeddingExtractor`.

    The inference runtime is chosen by ``cfg.reid_runtime``:

    * ``"onnx"`` — runs the OSNet ONNX model under ``onnxruntime``.
    * ``"torch"`` — runs an OSNet model from ``torchreid`` on torch.

    All runtime libraries and image-processing libraries are imported lazily.
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        """Load the OSNet model for the configured Re-ID runtime.

        Args:
            cfg: Tracker configuration carrying the Re-ID model path and runtime.

        Raises:
            ValueError: If ``cfg.reid_runtime`` is not supported.
        """
        runtime = cfg.reid_runtime.lower()
        if runtime not in _SUPPORTED_REID_RUNTIMES:
            raise ValueError(
                f"Unknown reid runtime {cfg.reid_runtime!r}; "
                f"expected one of {list(_SUPPORTED_REID_RUNTIMES)}"
            )

        self._cfg = cfg
        self._runtime = runtime
        # OSNet x1.0 input size and embedding width.
        self._input_hw: tuple[int, int] = (256, 128)
        self._dim = 512
        self._session: object | None = None
        self._model: object | None = None

        if runtime == "onnx":
            self._load_onnx(cfg.reid_model_path)
        else:
            self._load_torch(cfg.reid_model_path)

    def _load_onnx(self, model_path: str) -> None:
        """Create an ``onnxruntime`` session for the OSNet ONNX model."""
        import onnxruntime as ort  # lazy: heavy optional dependency

        logger.info("Loading OSNet ONNX model", extra={"event": "reid.load"})
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        output = self._session.get_outputs()[0]
        # Static embedding width when the model declares it; else keep default.
        if output.shape and isinstance(output.shape[-1], int):
            self._dim = int(output.shape[-1])

    def _load_torch(self, model_path: str) -> None:
        """Load an OSNet torch model via ``torchreid``."""
        import torch  # lazy: heavy optional dependency
        from torchreid.reid.models import build_model
        from torchreid.reid.utils import load_pretrained_weights

        logger.info("Loading OSNet torch model", extra={"event": "reid.load"})
        model = build_model("osnet_x1_0", num_classes=1, pretrained=False)
        load_pretrained_weights(model, model_path)
        model.eval()
        self._model = model
        self._torch = torch
        self._dim = int(getattr(model, "feature_dim", self._dim))

    @property
    def dim(self) -> int:
        """Embedding dimensionality produced by :meth:`extract`."""
        return self._dim

    def extract(
        self,
        image: np.ndarray,
        bboxes: Sequence[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Extract one L2-normalised embedding per bounding box.

        Args:
            image: HxWx3 frame the boxes index into.
            bboxes: ``(x1, y1, x2, y2)`` pixel boxes.

        Returns:
            ``(len(bboxes), dim)`` float32 array of L2-normalised embeddings.
            Returns an empty ``(0, dim)`` array when ``bboxes`` is empty.
        """
        if not bboxes:
            return np.empty((0, self._dim), dtype=np.float32)

        crops = [self._prepare_crop(image, bbox) for bbox in bboxes]
        batch = np.stack(crops, axis=0)

        if self._runtime == "onnx":
            embeddings = self._infer_onnx(batch)
        else:
            embeddings = self._infer_torch(batch)

        return _l2_normalise(embeddings.astype(np.float32))

    def _prepare_crop(
        self, image: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray:
        """Crop, resize and CHW-normalise one box into a model input tensor."""
        height, width = image.shape[:2]
        x1, y1, x2, y2 = bbox
        ix1 = int(max(0, min(width, round(x1))))
        iy1 = int(max(0, min(height, round(y1))))
        ix2 = int(max(0, min(width, round(x2))))
        iy2 = int(max(0, min(height, round(y2))))
        if ix2 <= ix1:
            ix2 = min(width, ix1 + 1)
        if iy2 <= iy1:
            iy2 = min(height, iy1 + 1)

        crop = image[iy1:iy2, ix1:ix2]
        resized = self._resize(crop, self._input_hw)

        # HWC uint8/float -> CHW float32 in [0, 1].
        chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
        return chw

    @staticmethod
    def _resize(crop: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        """Resize ``crop`` to ``target_hw`` using cv2 when available, else Pillow."""
        target_h, target_w = target_hw
        try:
            import cv2  # lazy: heavy optional dependency

            return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image  # lazy: heavy optional dependency

            pil = Image.fromarray(crop.astype(np.uint8))
            pil = pil.resize((target_w, target_h), Image.BILINEAR)
            return np.asarray(pil)

    def _infer_onnx(self, batch: np.ndarray) -> np.ndarray:
        """Run the ONNX session over a CHW batch.

        The constructor builds ``self._session`` for the onnx runtime, so this
        is only reached with a live session.
        """
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: batch.astype(np.float32)})
        return np.asarray(outputs[0])

    def _infer_torch(self, batch: np.ndarray) -> np.ndarray:
        """Run the torch OSNet model over a CHW batch."""
        torch = self._torch
        with torch.no_grad():
            tensor = torch.from_numpy(batch.astype(np.float32))
            features = self._model(tensor)
        return features.cpu().numpy()


@dataclasses.dataclass(slots=True)
class _GalleryEntry:
    """One global identity in the Re-ID gallery."""

    global_id: int
    embedding: np.ndarray
    last_seen: float


class ReIDManager:
    """Pure-numpy cross-camera identity assignment by appearance similarity.

    Maintains a gallery of ``(global_id, embedding, last_seen)`` entries. For
    each tracked detection that carries an embedding, the manager finds the most
    cosine-similar *unused* gallery entry; if the best similarity meets the
    threshold the existing ``global_id`` is reused (and its embedding refreshed),
    otherwise a new ``global_id`` is allocated. Entries unseen for longer than the
    TTL are evicted on each call.

    Two safety properties make it usable as the *shared* identity source across
    several camera-pipeline threads:

    * **Thread-safe** — one lock guards the gallery, so multiple pipelines may
      call :meth:`assign_global_ids` concurrently.
    * **Per-frame mutual exclusion** — within a single call, each gallery entry is
      matched to at most one detection. Two distinct people in the same frame can
      never collapse onto one ``global_id`` (which would undercount occupancy);
      the second falls through to a new identity.

    Detections without an embedding keep ``global_id=None``.
    """

    def __init__(
        self,
        similarity_threshold: float,
        gallery_ttl_seconds: float,
        clock: Clock,
    ) -> None:
        """Initialise an empty gallery.

        Args:
            similarity_threshold: Minimum cosine similarity (in ``[-1, 1]``) to
                treat two embeddings as the same identity.
            gallery_ttl_seconds: Evict gallery entries unseen for longer than
                this many seconds.
            clock: Time source (epoch seconds); abstracted for deterministic
                tests.
        """
        self._threshold = similarity_threshold
        self._ttl = gallery_ttl_seconds
        self._clock = clock
        self._gallery: list[_GalleryEntry] = []
        self._next_global_id = 1
        self._lock = threading.Lock()

    @property
    def gallery_size(self) -> int:
        """Number of live identities currently held in the gallery."""
        with self._lock:
            return len(self._gallery)

    def assign_global_ids(
        self, camera_id: int, tracked: list[TrackedDetection], ts: float
    ) -> list[TrackedDetection]:
        """Assign cross-camera ``global_id`` to each embedded detection.

        Thread-safe: the whole assignment (evict + match + allocate) runs under
        one lock so a shared manager can serve multiple camera pipelines.

        Args:
            camera_id: Source camera (used for logging/diagnostics only — global
                identities span cameras).
            tracked: Per-camera tracked detections; those with ``embedding`` set
                participate in matching.
            ts: Frame timestamp in epoch seconds, used as the ``last_seen`` mark.

        Returns:
            New :class:`TrackedDetection` copies with ``global_id`` populated for
            embedded detections; detections without an embedding are returned
            unchanged (``global_id=None``).
        """
        with self._lock:
            self._evict_expired()
            # Gallery indices already claimed by an earlier detection in THIS
            # frame; excluded from matching so two simultaneous tracks cannot
            # share one identity.
            used_indices: set[int] = set()
            assigned: list[TrackedDetection] = []
            for detection in tracked:
                embedding = detection.embedding
                if embedding is None:
                    assigned.append(detection)
                    continue
                query = self._as_unit_vector(embedding)
                global_id = self._match_or_allocate(query, ts, used_indices)
                assigned.append(dataclasses.replace(detection, global_id=global_id))

        if assigned:
            logger.debug(
                "Re-ID assignment",
                extra={"event": "reid.assign", "camera_id": camera_id},
            )
        return assigned

    def _match_or_allocate(
        self, query: np.ndarray, ts: float, used_indices: set[int]
    ) -> int:
        """Return the best-matching global id, allocating a new one if needed.

        Only gallery entries not in ``used_indices`` are eligible; the chosen
        index (matched or newly allocated) is added to ``used_indices``.
        """
        best_id, best_sim, best_idx = self._best_match(query, used_indices)

        if best_id is not None and best_sim >= self._threshold:
            entry = self._gallery[best_idx]
            entry.embedding = query
            entry.last_seen = ts
            used_indices.add(best_idx)
            return best_id

        return self._allocate(query, ts, used_indices)

    def _best_match(
        self, query: np.ndarray, used_indices: set[int]
    ) -> tuple[int | None, float, int]:
        """Find the most cosine-similar *unused* gallery entry to ``query``.

        Returns ``(global_id, similarity, index)``; ``global_id`` is ``None``
        when no eligible entry exists.
        """
        if not self._gallery:
            return None, -1.0, -1

        gallery_matrix = np.stack([entry.embedding for entry in self._gallery], axis=0)
        # Embeddings are unit vectors, so the dot product is the cosine similarity.
        similarities = gallery_matrix @ query
        if used_indices:
            similarities = similarities.copy()
            for idx in used_indices:
                similarities[idx] = -np.inf
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])
        if best_sim == -np.inf:  # every entry already claimed this frame
            return None, -1.0, -1
        return self._gallery[best_idx].global_id, best_sim, best_idx

    def _allocate(self, query: np.ndarray, ts: float, used_indices: set[int]) -> int:
        """Append a new identity to the gallery and return its id."""
        global_id = self._next_global_id
        self._next_global_id += 1
        self._gallery.append(
            _GalleryEntry(global_id=global_id, embedding=query, last_seen=ts)
        )
        used_indices.add(len(self._gallery) - 1)
        return global_id

    def _evict_expired(self) -> None:
        """Drop gallery entries unseen for longer than the TTL."""
        cutoff = self._clock.now() - self._ttl
        self._gallery = [entry for entry in self._gallery if entry.last_seen >= cutoff]

    @staticmethod
    def _as_unit_vector(embedding: np.ndarray) -> np.ndarray:
        """Return a 1-D L2-normalised copy of ``embedding`` as float64."""
        vector = np.asarray(embedding, dtype=np.float64).ravel()
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


def build_embedding_extractor(cfg: TrackerConfig) -> EmbeddingExtractor | None:
    """Construct the embedding extractor for ``cfg``, or ``None`` if disabled.

    Args:
        cfg: Tracker configuration. Returns ``None`` when ``cfg.reid_enabled``
            is false.

    Returns:
        An :class:`EmbeddingExtractor`, or ``None`` when Re-ID is disabled.
    """
    if not cfg.reid_enabled:
        return None
    return OSNetEmbeddingExtractor(cfg)
