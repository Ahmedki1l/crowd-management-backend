"""The engine must build a tracker only for cameras that actually track.

Snapshot-occupancy cameras count detections in-zone directly and return before
``tracker.update`` is ever reached. Building a tracker for them is not merely tidy-up:
:class:`~app.inference.tracker.ByteTrackTracker` eagerly imports ``supervision``, which
pulls in matplotlib/scipy/PIL — measured at ~80 MB of RSS and ~0.7 s of startup on this
project's venv — to construct objects nothing ever calls.

The engine and :class:`~app.engine.camera_pipeline.CameraPipeline` must also agree on
*which* cameras those are, because a camera built with ``tracker=None`` that then took
the tracked branch would dereference ``None``. Both read the same predicate
(:meth:`SnapshotPullConfig.counts_from_detections`); these tests pin that agreement.
"""

from __future__ import annotations

import pytest

from app.config.schema import AppConfig, ProcessingConfig, SnapshotPullConfig
from app.domain.models import CameraRole
from app.engine.engine import _build_runtimes
from tests.fixtures.specs import make_camera_spec

_SENTINEL_DETECTOR = object()
_SENTINEL_TRACKER = object()
_SENTINEL_EXTRACTOR = object()


@pytest.fixture
def snapshot_occupancy_config() -> AppConfig:
    """Config matching the deployment: occupancy cameras count detections directly."""
    return AppConfig(
        processing=ProcessingConfig(
            snapshot_pull=SnapshotPullConfig(
                roles=[CameraRole.OCCUPANCY.value],
                count_from_detections=True,
            )
        )
    )


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy perception factories; this is about *whether* they are called."""
    import app.inference.factory as factory

    monkeypatch.setattr(factory, "build_detector", lambda _cfg: _SENTINEL_DETECTOR)
    monkeypatch.setattr(factory, "build_tracker", lambda _cfg: _SENTINEL_TRACKER)
    monkeypatch.setattr(
        factory, "build_embedding_extractor", lambda _cfg: _SENTINEL_EXTRACTOR
    )


def test_snapshot_occupancy_camera_gets_no_tracker(
    snapshot_occupancy_config: AppConfig,
) -> None:
    spec = make_camera_spec(id=1, roles=(CameraRole.OCCUPANCY,))

    runtime = _build_runtimes([(spec, "pw")], snapshot_occupancy_config)[0]

    assert runtime.tracker is None
    assert runtime.extractor is None
    assert runtime.detector is _SENTINEL_DETECTOR  # the detector is always needed


def test_entry_exit_camera_still_gets_a_tracker(
    snapshot_occupancy_config: AppConfig,
) -> None:
    """The gate decodes RTSP and needs the tracked path, so it keeps its tracker."""
    spec = make_camera_spec(id=27, roles=(CameraRole.ENTRY_EXIT,))

    runtime = _build_runtimes([(spec, "pw")], snapshot_occupancy_config)[0]

    assert runtime.tracker is _SENTINEL_TRACKER
    assert runtime.extractor is _SENTINEL_EXTRACTOR


def test_occupancy_camera_keeps_its_tracker_when_counting_from_detections_is_off() -> None:
    """With the flag off, occupancy runs the tracked path — so it must still get one."""
    cfg = AppConfig(
        processing=ProcessingConfig(
            snapshot_pull=SnapshotPullConfig(
                roles=[CameraRole.OCCUPANCY.value],
                count_from_detections=False,
            )
        )
    )
    spec = make_camera_spec(id=1, roles=(CameraRole.OCCUPANCY,))

    runtime = _build_runtimes([(spec, "pw")], cfg)[0]

    assert runtime.tracker is _SENTINEL_TRACKER
