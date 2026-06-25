"""Evidence-snapshot I/O with path-traversal protection (HLD 6.5, 8.4).

Snapshots are JPEG frames captured for safety alerts. They live under a
configured base directory (``snapshots.dir``) and are addressed by the
**relative** path ``cam<camera_id>/<int ts>.jpg`` — the exact value stored in
``app.db.models.alerts.Snapshot.path``. All filesystem access funnels through
:func:`safe_join`, which guarantees a caller-supplied path can never escape the
base directory via ``..`` segments, absolute paths, or symlinks.

OpenCV (``cv2``) is imported lazily inside :func:`save_snapshot` so this module
stays importable with only core dependencies installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)


def safe_join(base_dir: str | Path, user_path: str) -> Path:
    """Resolve ``user_path`` under ``base_dir`` without allowing escape.

    The returned path is the absolute, resolved location of ``user_path``
    interpreted relative to ``base_dir``. Traversal attempts (``..``),
    absolute ``user_path`` values, and symlinks that would land outside the
    base are rejected.

    Args:
        base_dir: Trusted root directory that the result must stay within.
        user_path: Untrusted relative path supplied by a caller or client.

    Returns:
        The resolved absolute path located inside ``base_dir``.

    Raises:
        ValueError: If ``user_path`` is absolute or resolves outside
            ``base_dir``.
    """
    candidate = Path(user_path)
    if candidate.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {user_path!r}")

    base = Path(base_dir).resolve()
    resolved = (base / candidate).resolve()

    # ``is_relative_to`` (3.9+) is the precise, separator-aware containment
    # check; a naive ``startswith`` would let ``/base-evil`` pass for ``/base``.
    if not resolved.is_relative_to(base):
        raise ValueError(
            f"path traversal detected: {user_path!r} escapes base {str(base)!r}"
        )
    return resolved


def save_snapshot(
    image: np.ndarray,
    base_dir: str | Path,
    camera_id: int,
    ts: float,
) -> str:
    """Encode ``image`` as JPEG and write it under the snapshot base directory.

    The file is written to ``<base_dir>/cam<camera_id>/<int ts>.jpg``; the
    camera subdirectory is created if needed. The float epoch ``ts`` is
    truncated to whole seconds for the filename.

    Args:
        image: Frame to persist, as an OpenCV-compatible ``numpy`` array
            (BGR ``HxWx3`` or grayscale ``HxW``).
        base_dir: Snapshot root directory (``snapshots.dir``).
        camera_id: Source camera identifier; selects the ``cam<id>`` subdir.
        ts: Capture time as float epoch seconds.

    Returns:
        The path relative to ``base_dir`` (``cam<camera_id>/<int ts>.jpg``),
        ready to be stored in ``Snapshot.path``.

    Raises:
        ValueError: If JPEG encoding fails.
        OSError: If the directory cannot be created or the file cannot be
            written.
    """
    import cv2  # lazy: heavy/optional backend, not needed to import this module

    rel_path = f"cam{camera_id}/{int(ts)}.jpg"
    abs_path = safe_join(base_dir, rel_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        logger.error(
            "snapshot JPEG encode failed",
            extra={"camera_id": camera_id, "event": "snapshot_encode_failed"},
        )
        raise ValueError(f"failed to JPEG-encode snapshot for camera {camera_id}")

    abs_path.write_bytes(buffer.tobytes())
    return rel_path


def resolve_snapshot(base_dir: str | Path, rel_path: str) -> Path:
    """Resolve a stored relative snapshot path to an existing absolute file.

    Args:
        base_dir: Snapshot root directory (``snapshots.dir``).
        rel_path: Relative path as returned by :func:`save_snapshot` /
            stored in ``Snapshot.path``.

    Returns:
        The absolute path to the snapshot file.

    Raises:
        ValueError: If ``rel_path`` escapes ``base_dir`` (see
            :func:`safe_join`).
        FileNotFoundError: If no file exists at the resolved location.
    """
    abs_path = safe_join(base_dir, rel_path)
    if not abs_path.is_file():
        raise FileNotFoundError(f"snapshot not found: {rel_path!r}")
    return abs_path
