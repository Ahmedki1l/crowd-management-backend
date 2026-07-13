"""Integration tests for the real :class:`app.engine.camera_pipeline.CameraPipeline`.

These drive the genuine, threaded pipeline end-to-end over a deterministic
recorded clip — the same wiring production uses (capture thread -> bounded queue
-> worker -> detect -> track -> [embed + Re-ID] -> localise -> analyse ->
publish/persist) — with only the perception backends replaced by the
deterministic fakes (:mod:`app.inference.fakes`) and the frame producer replaced
by a :class:`~app.ingestion.recorded.RecordedClipSource` pumped through a real
:class:`~app.ingestion.capture.FrameSourceCaptureThread`. No RTSP camera, no
model runtime, no wall-clock sleeps.

Determinism: a :class:`~app.utils.clock.FakeClock` is the pipeline time source
and the :class:`RecordedClipSource` stamps frame ``i`` at ``base_ts + i / fps``,
so dwell/debounce measurements are reproducible. The drop-oldest frame queue is
sized at least as large as the clip, so no frame is dropped and the whole script
is processed. The test never sleeps for a fixed time: it subscribes a sync
collector to the bus and blocks on a :class:`threading.Event` that the collector
sets when the terminal :class:`~app.events.events.OccupancyUpdate` is published,
with a bounded timeout, before stopping the pipeline.

Each test owns its own :class:`~app.events.event_bus.InMemoryEventBus`,
:class:`~app.services.state_store.StateStore`, config and clip, so they are fully
hermetic. The autouse ``_isolated_environment`` conftest fixture provides the
fresh in-memory database the alert-persistence assertion reads back.
"""

from __future__ import annotations

import pytest

from app.config.schema import AppConfig, ProcessingConfig, StateMachineConfig
from app.domain.models import (
    BBox,
    CameraRole,
    Detection,
    ZoneType,
)
from app.engine.camera_pipeline import CameraPipeline
from app.events.event_bus import InMemoryEventBus
from app.events.events import (
    CameraHealth,
    CountUpdate,
    CrossingEvent,
    OccupancyUpdate,
)
from app.inference.fakes import FakeDetector, FakeEmbeddingExtractor, FakeTracker
from app.inference.reid import ReIDManager
from app.ingestion.capture import FrameSourceCaptureThread
from app.ingestion.recorded import RecordedClipSource
from app.services.state_store import StateStore
from app.utils.clock import FakeClock
from tests.fixtures.collector import EventCollector
from tests.fixtures.frames import make_clip
from tests.fixtures.specs import (
    make_camera_spec,
    make_line_spec,
    make_zone_spec,
    walk_across_line,
)

# --------------------------------------------------------------------------- #
# Deterministic timing / geometry constants.
# --------------------------------------------------------------------------- #
_CAMERA_ID = 1
_BASE_TS = 1000.0
_FPS = 2.0  # frame i is stamped at _BASE_TS + i * 0.5

# Small debounce windows keep the scripted clips short while still exercising
# the real hysteresis state machine.
_CONFIRM_ENTER = 2
_CONFIRM_LEAVE = 2

# Zone / line ids used across the scenarios.
_OCCUPANCY_ZONE_ID = 11
_LINE_ID = 20
_AREA_ID = "area-1"

# A bounded wait for the terminal event — generous enough to be reliable on a
# loaded CI box, but never a fixed sleep: the test returns as soon as the event
# fires. The frozen FakeClock means the worker spins on its queue-get timeout.
_EVENT_TIMEOUT_S = 10.0


def _person_bbox_at(cx: float, cy: float, w: float = 60.0, h: float = 160.0) -> BBox:
    """A person box whose ``bottom_center`` ground point is ``(cx, cy)``."""
    return BBox(x1=cx - w / 2.0, y1=cy - h, x2=cx + w / 2.0, y2=cy)


def _person_at(cx: float, cy: float) -> Detection:
    """A person detection whose ground point (bbox bottom-centre) is ``(cx, cy)``."""
    return Detection(bbox=_person_bbox_at(cx, cy), confidence=0.9)


def _config() -> AppConfig:
    """Build an :class:`AppConfig` tuned for short, deterministic clips.

    ``queue_maxsize`` is set well above any clip length so the drop-oldest queue
    drops nothing and the whole script is processed.
    """
    return AppConfig(
        processing=ProcessingConfig(queue_maxsize=256),
        state_machine=StateMachineConfig(
            confirm_enter_frames=_CONFIRM_ENTER,
            confirm_leave_frames=_CONFIRM_LEAVE,
        ),
    )


def _capture_factory(clock: FakeClock, n_frames: int, role: CameraRole):
    """Build a capture-thread factory feeding a recorded clip of ``n_frames``.

    The factory matches the ``Callable[[BoundedFrameQueue], CaptureThread]``
    signature the pipeline expects; it wraps a :class:`RecordedClipSource` of
    synthetic (all-zero) frames in a real :class:`FrameSourceCaptureThread`.
    """

    def _build(queue) -> FrameSourceCaptureThread:
        source = RecordedClipSource(
            camera_id=_CAMERA_ID,
            fps=_FPS,
            role=role,
            frames=make_clip(n_frames),
            base_ts=_BASE_TS,
        )
        return FrameSourceCaptureThread(source, queue, clock)

    return _build


def _run_pipeline(
    *,
    spec,
    script: list[list[Detection]],
    role: CameraRole,
    await_event: type,
    await_count: int = 1,
    embedding_extractor: FakeEmbeddingExtractor | None = None,
    reid_manager: ReIDManager | None = None,
) -> tuple[EventCollector, StateStore]:
    """Drive the real pipeline over ``script`` and return the collected events.

    Subscribes the collector to the bus *before* starting so no event is missed,
    starts the pipeline, blocks (bounded) until ``await_count`` events of
    ``await_event`` have been published, then stops the pipeline and joins its
    threads. The wait — not a fixed sleep — is what makes the test deterministic
    despite the worker running on its own thread.
    """
    clock = FakeClock(start=_BASE_TS)
    bus = InMemoryEventBus()
    store = StateStore()
    collector = EventCollector()
    bus.subscribe_sync(collector)

    pipeline = CameraPipeline(
        spec=spec,
        password="unused-for-recorded-source",
        detector=FakeDetector(script),
        tracker=FakeTracker(),
        embedding_extractor=embedding_extractor,
        bus=bus,
        clock=clock,
        cfg=_config(),
        store=store,
        reid_manager=reid_manager,
        capture_factory=_capture_factory(clock, len(script), role),
    )

    pipeline.start()
    try:
        reached = collector.wait_for_count(
            await_event, await_count, timeout=_EVENT_TIMEOUT_S
        )
        assert reached, (
            f"pipeline did not publish {await_count} {await_event.__name__} "
            f"within {_EVENT_TIMEOUT_S}s"
        )
    finally:
        pipeline.stop()

    return collector, store


# --------------------------------------------------------------------------- #
# Scenario A — walker crossing a line + dweller in occupancy/waiting zones.
# --------------------------------------------------------------------------- #
def _walker_and_dweller_script() -> list[list[Detection]]:
    """Script one walker crossing the line and one dweller entering then leaving.

    Walker (person A) sweeps left-to-right at ``y=250`` (above the zone polygon,
    so it only crosses the vertical line at ``x=640``, never joining a zone),
    crossing once in the IN direction. The walker advances in ~18px steps so
    consecutive boxes overlap above the IoU tracker threshold and keep one stable
    track id (a wider step would mint a new id each frame and hide the crossing).
    Dweller (person B) holds ground point ``(600, 450)`` inside the default
    polygon long enough to confirm entry, then moves straight down clear of the
    polygon to confirm a leave — taking the zone count 0 -> 1 -> 0.
    """
    walker = walk_across_line(n=10, y=250.0, start_x=560.0, end_x=720.0)
    # 450 (inside) x6, then down and clear of the polygon (cy >= 713 breaks both
    # the ground-point test and the >=0.3 overlap fallback).
    dweller_y = [450.0, 450.0, 450.0, 450.0, 450.0, 450.0, 720.0, 800.0, 880.0, 880.0]
    assert len(walker) == len(dweller_y)
    return [
        [walker[i][0], _person_at(600.0, dweller_y[i])]
        for i in range(len(dweller_y))
    ]


@pytest.fixture
def scenario_a():
    """Run scenario A once and share the result across its assertions."""
    occupancy_zone = make_zone_spec(
        id=_OCCUPANCY_ZONE_ID, camera_id=_CAMERA_ID, type=ZoneType.OCCUPANCY
    )
    line = make_line_spec(id=_LINE_ID, camera_id=_CAMERA_ID, area_id=_AREA_ID)
    spec = make_camera_spec(
        id=_CAMERA_ID,
        roles=(CameraRole.ENTRY_EXIT,),
        zones=(occupancy_zone,),
        lines=(line,),
    )
    # The dweller drives the zone count 0 -> 1 -> 0, so the third OccupancyUpdate is
    # the terminal event of the clip: awaiting it guarantees the walker's crossing
    # (mid-clip) has already been processed.
    collector, store = _run_pipeline(
        spec=spec,
        script=_walker_and_dweller_script(),
        role=CameraRole.ENTRY_EXIT,
        await_event=OccupancyUpdate,
        await_count=3,
    )
    return collector, store


def test_pipeline_publishes_positive_occupancy_update(scenario_a) -> None:
    collector, _ = scenario_a

    occupancy = collector.of_type(OccupancyUpdate)

    assert any(e.count > 0 for e in occupancy)


def test_pipeline_publishes_single_inbound_crossing(scenario_a) -> None:
    collector, _ = scenario_a

    crossings = collector.of_type(CrossingEvent)

    assert [c.direction.value for c in crossings] == ["in"]


def test_pipeline_publishes_count_update_with_net_change(scenario_a) -> None:
    collector, _ = scenario_a

    final_count = collector.of_type(CountUpdate)[-1]

    assert final_count.area_id == _AREA_ID
    assert final_count.net == 1


def test_pipeline_publishes_camera_health(scenario_a) -> None:
    collector, _ = scenario_a

    health = collector.of_type(CameraHealth)

    assert health, "expected at least one CameraHealth heartbeat"
    assert health[-1].camera_id == _CAMERA_ID


def test_pipeline_writes_camera_health_to_state_store(scenario_a) -> None:
    _, store = scenario_a

    health = store.get_camera_health(_CAMERA_ID)

    assert health is not None
    assert health.camera_id == _CAMERA_ID
