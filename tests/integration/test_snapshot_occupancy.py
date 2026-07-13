"""End-to-end tests for the snapshot-pull occupancy path — the one production runs.

The deployed config (``config/config.local.yaml``) sets
``processing.snapshot_pull.roles: ["occupancy"]`` and ``count_from_detections: true``,
which selects a different branch of :class:`~app.engine.camera_pipeline.CameraPipeline`
from the one :mod:`tests.integration.test_camera_pipeline` covers: detections are counted
in-zone directly and :meth:`OccupancyCalculator.process_counts` publishes the result,
while the whole tracked half of the pipeline (tracker, Re-ID, presence machine, line
crossing, entry/exit, safety, waiting, heatmap) is skipped by an early return in
``_process_frame``.

``count_from_detections`` defaults to ``False`` in the schema, so that branch is only
reachable from config — which is why it needs its own coverage rather than riding on the
default-path tests.

These drive the genuine threaded pipeline (capture thread -> bounded queue -> worker ->
detect -> count -> publish -> project) through the ``capture_factory`` seam, with
perception replaced by the deterministic fakes and time by a
:class:`~app.utils.clock.FakeClock`. No HTTP, no camera, no model runtime.

Determinism note: the snapshot worker drains the queue with
:meth:`BoundedFrameQueue.get_latest`, which *drops any backlog by design* — a stale frame
answers a question about the past. A source that pumps frames as fast as it can therefore
races the worker and most frames simply vanish. Production does not hit this because
``SnapshotCaptureThread`` paces at ``snapshot_pull.interval_s``; :class:`_PacedClipSource`
reproduces that back-pressure by holding the next frame until the worker has taken the
previous one, so every scripted frame is observed exactly once.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.config.schema import AppConfig, ProcessingConfig, SnapshotPullConfig
from app.domain.models import BBox, CameraRole, Detection, FramePacket, ZoneType
from app.engine.camera_pipeline import CameraPipeline
from app.events.event_bus import InMemoryEventBus
from app.events.events import OccupancyUpdate
from app.inference.fakes import FakeDetector, FakeTracker
from app.ingestion.capture import FrameSourceCaptureThread
from app.ingestion.frame_queue import BoundedFrameQueue
from app.services.projectors import StateProjector
from app.services.state_store import StateStore
from app.utils.clock import FakeClock
from tests.fixtures.collector import EventCollector
from tests.fixtures.specs import make_camera_spec, make_zone_spec

_CAMERA_ID = 1
_ZONE_ID = 11
_BASE_TS = 1000.0
_FPS = 2.0
_EVENT_TIMEOUT_S = 10.0

# Ground points inside tests.fixtures.specs.DEFAULT_POLYGON (x 400-800, y 300-600).
_INSIDE = ((500.0, 400.0), (600.0, 450.0), (700.0, 500.0))
# Ground point clear of the polygon — must never be counted.
_OUTSIDE = (100.0, 650.0)


class _PacedClipSource:
    """A :class:`~app.domain.interfaces.FrameSource` that yields to the consumer.

    Holds the next frame until the worker has drained the previous one, so the queue
    never holds a backlog for ``get_latest`` to discard. This is back-pressure, not a
    fixed sleep: the poll spins on the real queue depth and stops as soon as it clears.
    """

    # A frame the worker never takes means the pipeline is wedged; fail fast rather
    # than block the capture thread's join forever.
    _DRAIN_TIMEOUT_S = 5.0
    _POLL_S = 0.001

    def __init__(self, queue: BoundedFrameQueue, n_frames: int) -> None:
        self._queue = queue
        self._n_frames = n_frames
        self._next = 0
        self._closed = False

    def read(self) -> FramePacket | None:
        if self._closed or self._next >= self._n_frames:
            return None

        deadline = time.monotonic() + self._DRAIN_TIMEOUT_S
        while self._queue.qsize() > 0:
            if time.monotonic() > deadline:
                return None  # worker is not consuming; end the clip
            time.sleep(self._POLL_S)

        packet = FramePacket(
            camera_id=_CAMERA_ID,
            frame_idx=self._next,
            ts=_BASE_TS + self._next / _FPS,
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            role=CameraRole.OCCUPANCY,
        )
        self._next += 1
        return packet

    def close(self) -> None:
        self._closed = True


class _CountingTracker(FakeTracker):
    """A tracker that records whether the pipeline ever called it.

    The snapshot occupancy path must never invoke the tracker. Counting calls rather than
    raising keeps a failure legible: the worker swallows per-frame exceptions, so a raise
    here would surface only as a confusing timeout.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def update(self, detections, image=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().update(detections, image)


def _person_at(cx: float, cy: float) -> Detection:
    """A person detection whose ground point (bbox bottom-centre) is ``(cx, cy)``."""
    return Detection(
        bbox=BBox(x1=cx - 30.0, y1=cy - 160.0, x2=cx + 30.0, y2=cy), confidence=0.9
    )


def _config() -> AppConfig:
    """An :class:`AppConfig` mirroring the deployed snapshot-pull occupancy setup."""
    return AppConfig(
        processing=ProcessingConfig(
            queue_maxsize=256,
            snapshot_pull=SnapshotPullConfig(
                roles=[CameraRole.OCCUPANCY.value],
                count_from_detections=True,
            ),
        ),
    )


def _changing_occupancy_script() -> list[list[Detection]]:
    """Two people in-zone, then three, then one — with a decoy outside the polygon.

    Occupancy is published only when a zone's count *changes*, so the three plateaux
    yield exactly three OccupancyUpdates: 2, 3, 1.
    """
    two = [_person_at(*_INSIDE[0]), _person_at(*_INSIDE[1])]
    three = [_person_at(*p) for p in _INSIDE]
    one_plus_decoy = [_person_at(*_INSIDE[1]), _person_at(*_OUTSIDE)]
    return [two, two, three, three, one_plus_decoy, one_plus_decoy]


def _run_pipeline(
    script: list[list[Detection]], await_occupancy_updates: int
) -> tuple[EventCollector, StateStore, _CountingTracker]:
    """Drive the real pipeline over ``script`` on the snapshot-occupancy branch.

    Wires a real :class:`StateProjector` to the bus — as ``RuntimeWiring`` does in
    production — so the assertions can read the live occupancy the API would serve.
    Blocks (bounded) until ``await_occupancy_updates`` OccupancyUpdates arrive rather
    than sleeping for a fixed time.
    """
    clock = FakeClock(start=_BASE_TS)
    bus = InMemoryEventBus()
    store = StateStore()
    tracker = _CountingTracker()
    collector = EventCollector()
    bus.subscribe_sync(StateProjector(store).handle)
    bus.subscribe_sync(collector)

    spec = make_camera_spec(
        id=_CAMERA_ID,
        roles=(CameraRole.OCCUPANCY,),
        zones=(
            make_zone_spec(id=_ZONE_ID, camera_id=_CAMERA_ID, type=ZoneType.OCCUPANCY),
        ),
    )

    def _capture_factory(queue: BoundedFrameQueue) -> FrameSourceCaptureThread:
        source = _PacedClipSource(queue, len(script))
        return FrameSourceCaptureThread(source, queue, clock)

    pipeline = CameraPipeline(
        spec=spec,
        password="unused-for-recorded-source",
        detector=FakeDetector(script),
        tracker=tracker,
        embedding_extractor=None,
        bus=bus,
        clock=clock,
        cfg=_config(),
        store=store,
        capture_factory=_capture_factory,
    )

    pipeline.start()
    try:
        reached = collector.wait_for_count(
            OccupancyUpdate, await_occupancy_updates, timeout=_EVENT_TIMEOUT_S
        )
        assert reached, (
            f"snapshot pipeline did not publish {await_occupancy_updates} "
            f"OccupancyUpdate(s) within {_EVENT_TIMEOUT_S}s"
        )
    finally:
        pipeline.stop()

    return collector, store, tracker


@pytest.fixture
def snapshot_run() -> tuple[EventCollector, StateStore, _CountingTracker]:
    """Drive the scripted scene through the snapshot path once, for its assertions."""
    return _run_pipeline(_changing_occupancy_script(), await_occupancy_updates=3)


def test_snapshot_path_publishes_a_count_per_change_ignoring_people_outside_the_zone(
    snapshot_run: tuple[EventCollector, StateStore, _CountingTracker],
) -> None:
    """Counts are the in-zone detections, emitted once per change.

    The last plateau carries a decoy outside the polygon, so a broken zone filter would
    show up here as a trailing 2 rather than 1.
    """
    collector, _, _ = snapshot_run

    updates = collector.of_type(OccupancyUpdate)
    assert [u.count for u in updates] == [2, 3, 1]
    assert {u.zone_id for u in updates} == {_ZONE_ID}


def test_snapshot_occupancy_reaches_the_live_state_store(
    snapshot_run: tuple[EventCollector, StateStore, _CountingTracker],
) -> None:
    """The published count is projected into the cache ``GET /occupancy`` reads."""
    _, store, _ = snapshot_run

    assert store.get_occupancy(_ZONE_ID).count == 1


def test_snapshot_path_never_runs_the_tracked_half(
    snapshot_run: tuple[EventCollector, StateStore, _CountingTracker],
) -> None:
    """The early return in ``_process_frame`` must skip the tracked half entirely.

    ``tracker.update`` is the first statement after that return, so a call count of zero
    is sufficient: nothing downstream of it (Re-ID, presence machine, line crossing,
    entry/exit, safety, waiting, heatmap) can have run either. That inertness is the
    premise the removal work rests on — if this ever fails, the premise is gone.
    """
    _, _, tracker = snapshot_run

    assert tracker.calls == 0
