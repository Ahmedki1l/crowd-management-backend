"""Inference-layer factory (HLD 6.2).

Single import surface for building perception backends from configuration. The
underlying builders lazily import their heavy runtimes, so importing this module
never pulls in ``ultralytics``/``supervision``/``onnxruntime``/``torch``.
"""

from __future__ import annotations

from app.inference.detector import build_detector
from app.inference.reid import build_embedding_extractor
from app.inference.tracker import build_tracker

__all__ = [
    "build_detector",
    "build_tracker",
    "build_embedding_extractor",
]
