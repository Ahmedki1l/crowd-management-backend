"""Synthetic frame helpers for tests.

Frames are plain ``numpy`` arrays (BGR-style ``uint8``) so the fakes and
``RecordedClipSource`` can be driven without OpenCV or any heavy dependency.
The fakes ignore pixel content (they replay scripted detections), so all-zero
frames of the right shape are sufficient and deterministic.
"""

from __future__ import annotations

import numpy as np


def synthetic_frame(w: int = 1280, h: int = 720) -> np.ndarray:
    """Return a single all-zero BGR frame of shape ``(h, w, 3)`` uint8."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_clip(n: int, w: int = 1280, h: int = 720) -> list[np.ndarray]:
    """Return a list of ``n`` independent synthetic frames.

    Each frame is a distinct array (not aliased) so a source/consumer mutating
    one cannot affect another.
    """
    return [synthetic_frame(w, h) for _ in range(n)]
