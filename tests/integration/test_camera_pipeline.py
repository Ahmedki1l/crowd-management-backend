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
sets when the terminal :class:`~app.events.events.DwellClosed` is published,
with a bounded timeout, before stopping the pipeline.

Each test owns its own :class:`~app.events.event_bus.InMemoryEventBus`,
:class:`~app.services.state_store.StateStore`, config and clip, so they are fully
hermetic. The autouse ``_isolated_environment`` conftest fixture provides the
fresh in-memory database the alert-persistence assertion reads back.
"""

from __future__ import annotations

import threading

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
    AlertRaised,
    CameraHealth,
    CountUpdate,
    CrossingEvent,
    DwellClosed,
    Event,
    OccupancyUpdate,
)
from app.inference.fakes import FakeDetector, FakeEmbeddingExtractor, FakeTracker
from app.inference.reid import ReIDManager
from app.ingestion.capture import FrameSourceCaptureThread
from app.ingestion.recorded import RecordedClipSource
from app.services.state_store import StateStore
from app.utils.clock import FakeClock
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
# the real hysteresis state machine and alert debounce.
_CONFIRM_ENTER = 2
_CONFIRM_LEAVE = 2
_ALERT_DEBOUNCE = 2

# Zone / line ids used across the scenarios.
_WAITING_ZONE_ID = 10
_OCCUPANCY_ZONE_ID = 11
_RESTRICTED_ZONE_ID = 12
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


class _Collector:
    """Thread-safe sync bus subscriber that records every published event.

    The pipeline publishes from its worker thread, so collection must be guarded.
    A :class:`threading.Condition` lets a test block until a predicate over the
    collected events holds (e.g. "a DwellClosed has arrived") instead of sleeping
    for a fixed time; the condition is notified on every published event.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._events: list[Event] = []

    def __call__(self, event: Event) -> None:
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    def of_type(self, event_type: type) -> list[Event]:
        """Return a snapshot of the collected events of ``event_type``."""
        with self._cond:
            return [e for e in self._events if isinstance(e, event_type)]

    def wait_for_count(
        self, event_type: type, count: int, timeout: float
    ) -> bool:
        """Block until ``count`` events of ``event_type`` have been collected.

        Returns ``True`` once the threshold is reached, ``False`` on timeout. The
        wait is driven by the bus condition (woken on each event), so it returns
        as soon as the condition is met — never a fixed sleep. The predicate runs
        while the condition lock is held, so it reads ``self._events`` directly
        rather than re-entering :meth:`of_type`.
        """

        def _reached() -> bool:
            return sum(isinstance(e, event_type) for e in self._events) >= count

        with self._cond:
            return self._cond.wait_for(_reached, timeout=timeout)


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
            alert_debounce_frames=_ALERT_DEBOUNCE,
            alert_cooldown_seconds=30.0,
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
) -> tuple[_Collector, StateStore]:
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
    collector = _Collector()
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
    polygon to confirm a leave (closing a dwell session).
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
    waiting_zone = make_zone_spec(
        id=_WAITING_ZONE_ID, camera_id=_CAMERA_ID, type=ZoneType.WAITING
    )
    occupancy_zone = make_zone_spec(
        id=_OCCUPANCY_ZONE_ID, camera_id=_CAMERA_ID, type=ZoneType.OCCUPANCY
    )
    line = make_line_spec(id=_LINE_ID, camera_id=_CAMERA_ID, area_id=_AREA_ID)
    spec = make_camera_spec(
        id=_CAMERA_ID,
        roles=(CameraRole.WAITING,),
        zones=(waiting_zone, occupancy_zone),
        lines=(line,),
    )
    collector, store = _run_pipeline(
        spec=spec,
        script=_walker_and_dweller_script(),
        role=CameraRole.WAITING,
        await_event=DwellClosed,
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


def test_pipeline_closes_dwell_with_positive_duration(scenario_a) -> None:
    collector, _ = scenario_a

    closed = collector.of_type(DwellClosed)

    assert len(closed) == 1
    assert closed[0].zone_id == _WAITING_ZONE_ID
    assert closed[0].dwell_s > 0.0


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


# --------------------------------------------------------------------------- #
# Scenario B — intrusion alert in a RESTRICTED zone is published AND persisted.
# --------------------------------------------------------------------------- #
def _intruder_script(n: int = 8) -> list[list[Detection]]:
    """One person standing inside the restricted zone for every frame.

    The fixed ground point ``(600, 450)`` sits inside the default polygon, so the
    person confirms presence and — held past the alert debounce window — trips an
    intrusion alert.
    """
    return [[_person_at(600.0, 450.0)] for _ in range(n)]


def _make_camera_and_zone_rows(session, zone_type: str) -> None:
    """Insert matching camera and zone rows so the persisted alert's FKs resolve.

    The pipeline persists alerts via ``session_scope`` against the per-test
    in-memory database; inserting the referenced camera/zone keeps the row
    realistic (real foreign keys) rather than orphaned.
    """
    from app.db.models import Camera, Zone

    session.add(
        Camera(
            id=_CAMERA_ID,
            name="cam-1",
            area="lobby",
            ip="10.0.0.1",
            port=554,
            username="admin",
            roles=["occupancy"],
            stream_channel_sub=102,
            stream_channel_main=101,
            enabled=True,
        )
    )
    session.add(
        Zone(
            id=_RESTRICTED_ZONE_ID,
            camera_id=_CAMERA_ID,
            name="vault",
            type=zone_type,
            polygon=[[400.0, 300.0], [800.0, 300.0], [800.0, 600.0], [400.0, 600.0]],
        )
    )
    session.commit()


def test_pipeline_raises_and_persists_intrusion_alert(session) -> None:
    """A person dwelling in a RESTRICTED zone yields a published + persisted alert."""
    _make_camera_and_zone_rows(session, zone_type="restricted")

    restricted = make_zone_spec(
        id=_RESTRICTED_ZONE_ID,
        camera_id=_CAMERA_ID,
        name="vault",
        type=ZoneType.RESTRICTED,
    )
    spec = make_camera_spec(
        id=_CAMERA_ID, roles=(CameraRole.OCCUPANCY,), zones=(restricted,)
    )

    collector, _ = _run_pipeline(
        spec=spec,
        script=_intruder_script(),
        role=CameraRole.OCCUPANCY,
        await_event=AlertRaised,
    )

    alerts = collector.of_type(AlertRaised)
    assert alerts, "expected an AlertRaised to be published"
    assert alerts[0].zone_id == _RESTRICTED_ZONE_ID

    from app.db.repositories.alert_repo import AlertRepository

    rows = AlertRepository(session).list(zone_id=_RESTRICTED_ZONE_ID)
    assert len(rows) >= 1
    assert rows[0].type == "intrusion"
    assert rows[0].camera_id == _CAMERA_ID


# --------------------------------------------------------------------------- #
# Scenario C — Re-ID gives one identity across a leave/re-enter (FakeEmbedding).
# --------------------------------------------------------------------------- #
def _leave_and_reenter_script() -> list[list[Detection]]:
    """One person who dwells, fully disappears, then re-enters at the same box.

    While the person is absent (empty detection frames) the IoU tracker forgets
    the track, so on return it would mint a *new* ``track_id``. The dweller
    re-enters at the identical bounding box, so the appearance embedding is
    identical and the Re-ID gallery must collapse both appearances onto one
    ``global_id`` — which the pipeline promotes to the effective identity. The
    person finally leaves so a second dwell session closes.
    """
    inside = [_person_at(600.0, 450.0)]
    empty: list[Detection] = []
    gone = [_person_at(600.0, 880.0)]  # clear of the polygon, breaks membership
    # enter+confirm, leave+confirm, gap, re-enter+confirm, leave+confirm.
    return [
        inside,  # 0
        inside,  # 1 enter confirmed (B in zone)
        inside,  # 2
        gone,  # 3
        gone,  # 4 leave confirmed -> DwellClosed #1
        empty,  # 5 person fully out of frame: tracker drops the track
        empty,  # 6
        inside,  # 7 re-enters at SAME box -> Re-ID reuses the global id
        inside,  # 8 enter confirmed again
        inside,  # 9
        gone,  # 10
        gone,  # 11 leave confirmed -> DwellClosed #2
    ]


def test_reid_keeps_one_dwell_identity_across_leave_and_reenter() -> None:
    """With Re-ID on, a leave/re-enter keeps a single appearance-stable identity."""
    clock = FakeClock(start=_BASE_TS)
    reid = ReIDManager(
        similarity_threshold=0.6, gallery_ttl_seconds=600.0, clock=clock
    )
    waiting_zone = make_zone_spec(
        id=_WAITING_ZONE_ID, camera_id=_CAMERA_ID, type=ZoneType.WAITING
    )
    spec = make_camera_spec(
        id=_CAMERA_ID, roles=(CameraRole.WAITING,), zones=(waiting_zone,)
    )

    collector, _ = _run_pipeline(
        spec=spec,
        script=_leave_and_reenter_script(),
        role=CameraRole.WAITING,
        await_event=DwellClosed,
        await_count=2,
        embedding_extractor=FakeEmbeddingExtractor(),
        reid_manager=reid,
    )
    closed = collector.of_type(DwellClosed)

    assert len(closed) == 2, "expected two closed dwell sessions"
    # Both sessions reference the SAME identity (the Re-ID global id), proving
    # the appearance-stable identity — not the renumbered tracker id — is what
    # the dwell session keys on across the leave/re-enter.
    assert closed[0].track_ref == closed[1].track_ref


def test_reid_disabled_renumbers_identity_across_reenter() -> None:
    """Control: without Re-ID the tracker renumbers the re-entrant track.

    This is the contrast that makes the Re-ID test meaningful — it shows the two
    dwell sessions would otherwise carry *different* identities, so the previous
    test's shared ``track_ref`` is genuinely the Re-ID gallery's doing.
    """
    waiting_zone = make_zone_spec(
        id=_WAITING_ZONE_ID, camera_id=_CAMERA_ID, type=ZoneType.WAITING
    )
    spec = make_camera_spec(
        id=_CAMERA_ID, roles=(CameraRole.WAITING,), zones=(waiting_zone,)
    )

    collector, _ = _run_pipeline(
        spec=spec,
        script=_leave_and_reenter_script(),
        role=CameraRole.WAITING,
        await_event=DwellClosed,
        await_count=2,
        embedding_extractor=None,
        reid_manager=None,
    )
    closed = collector.of_type(DwellClosed)

    assert len(closed) == 2
    assert closed[0].track_ref != closed[1].track_ref
