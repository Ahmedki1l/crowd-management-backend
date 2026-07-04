"""The engine wires snapshot-pull vs RTSP capture per camera role.

Asserts :meth:`CameraPipeline._build_capture` returns a
:class:`SnapshotCaptureThread` for a role listed in
``processing.snapshot_pull.roles`` and a :class:`RtspCaptureThread` otherwise.
Construction only — no threads are started and no network occurs.
"""

from __future__ import annotations

from app.config.schema import AppConfig
from app.domain.models import CameraRole, CameraSpec
from app.engine.camera_pipeline import CameraPipeline
from app.events.event_bus import InMemoryEventBus
from app.inference.fakes import FakeDetector, FakeTracker
from app.ingestion.capture import RtspCaptureThread, SnapshotCaptureThread
from app.ingestion.frame_queue import BoundedFrameQueue
from app.services.state_store import StateStore
from app.utils.clock import FakeClock


def _pipeline(cfg: AppConfig) -> CameraPipeline:
    spec = CameraSpec(
        id=1,
        name="cam",
        area="lobby",
        ip="10.0.0.9",
        port=554,
        username="admin",
        roles=(CameraRole.OCCUPANCY,),
        stream_channel_sub=102,
        stream_channel_main=101,
    )
    return CameraPipeline(
        spec=spec,
        password="pw",
        detector=FakeDetector([[]]),
        tracker=FakeTracker(),
        embedding_extractor=None,
        bus=InMemoryEventBus(),
        clock=FakeClock(),
        cfg=cfg,
        store=StateStore(),
    )


def test_occupancy_role_in_snapshot_roles_builds_snapshot_capture() -> None:
    cfg = AppConfig()
    cfg.processing.snapshot_pull.roles = ["occupancy"]

    capture = _pipeline(cfg)._build_capture(BoundedFrameQueue(2))

    assert isinstance(capture, SnapshotCaptureThread)


def test_default_config_builds_rtsp_capture() -> None:
    cfg = AppConfig()  # snapshot_pull.roles defaults to [] -> RTSP for everyone

    capture = _pipeline(cfg)._build_capture(BoundedFrameQueue(2))

    assert isinstance(capture, RtspCaptureThread)
