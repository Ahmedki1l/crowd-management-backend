"""End-to-end pipeline contract test over recorded frames (BRD acceptance).

Proves the BRD acceptance path — *live frames produce metrics* — without any
RTSP camera or heavy inference stack. A deterministic clip of scripted
:class:`~app.domain.models.Detection` frames (one person walking across a
counting line, another dwelling then leaving an occupancy/waiting zone) is driven
through the whole chain::

    RecordedClipSource -> FakeDetector -> FakeTracker -> ZoneEvaluator
        -> ZonePresenceTracker -> {OccupancyCalculator, LineCrossingDetector
        -> EntryExitCalculator, WaitingCalculator}

The emitted analytics events are then fanned through :class:`StateProjector` into
a :class:`StateStore`, and the resulting ``/state``-style read model is asserted.
Every step uses the real geometry / state-machine / calculators — only the
perception backends are fakes — so this exercises the genuine pipeline.

Timing is fully deterministic: frame ``i`` is stamped ``base_ts + i / fps`` by the
:class:`RecordedClipSource`, so dwell measurements are reproducible with no real
clock and no sleeps.
"""

from __future__ import annotations

import pytest

from app.analytics.entry_exit import EntryExitCalculator
from app.analytics.occupancy import OccupancyCalculator
from app.analytics.waiting import WaitingCalculator
from app.domain.models import BBox, CameraRole, Detection, ZoneType
from app.events.events import (
    CountUpdate,
    CrossingEvent,
    DwellClosed,
    Event,
    OccupancyUpdate,
)
from app.ingestion.recorded import RecordedClipSource
from app.localisation.lines import LineCrossingDetector
from app.localisation.state_machine import ZonePresenceTracker
from app.localisation.zones import ZoneEvaluator
from app.services.projectors import StateProjector
from app.services.state_store import StateStore
from app.utils.clock import FakeClock
from tests.fixtures.frames import make_clip
from tests.fixtures.specs import make_line_spec, make_zone_spec

# Pipeline timing constants. fps=2 means frame i is stamped at base_ts + i*0.5.
_BASE_TS = 1000.0
_FPS = 2.0
_CONFIRM_FRAMES = 2  # consecutive frames to confirm an enter / a leave


def _person(cx: float, cy: float) -> Detection:
    """A person detection whose ground point (bbox bottom-centre) is ``(cx, cy)``.

    Box is 60x160 px so two consecutive same-person boxes overlap enough (IoU >=
    the FakeTracker threshold) to keep one stable track across small moves.
    """
    return Detection(
        bbox=BBox(x1=cx - 30.0, y1=cy - 160.0, x2=cx + 30.0, y2=cy),
        confidence=0.9,
    )


def _scripted_frames() -> list[list[Detection]]:
    """Two-person script: one walks across the line, one dwells then leaves.

    Person A (the walker) sweeps left-to-right from x=500 to x=780 at y=400,
    crossing the vertical line at x=640 once in the IN direction, then parks on
    the right. Person B (the dweller) holds the ground point (600, 450) inside the
    default polygon, then walks straight down out of it (y past 720, where the box
    no longer overlaps the polygon) and stays gone long enough to confirm a leave.
    The per-frame steps keep each person's IoU above the tracker threshold so both
    keep a single stable track id throughout.
    """
    walker_x = [500.0 + (780.0 - 500.0) / 11.0 * i for i in range(12)] + [780.0] * 4
    dweller_y = [450.0] * 9 + [510.0, 570.0, 630.0, 690.0, 750.0, 810.0, 810.0]
    assert len(walker_x) == len(dweller_y)
    return [
        [_person(walker_x[i], 400.0), _person(600.0, dweller_y[i])]
        for i in range(len(walker_x))
    ]


def _run_pipeline() -> tuple[list[Event], StateStore]:
    """Drive the scripted clip through the full pipeline into a fresh StateStore.

    Returns the ordered list of analytics events the calculators emitted and the
    :class:`StateStore` after the :class:`StateProjector` has consumed them all.
    """
    from app.inference.fakes import FakeDetector, FakeTracker

    script = _scripted_frames()
    n_frames = len(script)

    zone = make_zone_spec(id=1, type=ZoneType.WAITING)
    line = make_line_spec(id=1)
    clock = FakeClock(start=_BASE_TS)

    detector = FakeDetector(script)
    tracker = FakeTracker()
    zone_evaluator = ZoneEvaluator([zone])
    presence = ZonePresenceTracker(
        [zone],
        confirm_enter_frames=_CONFIRM_FRAMES,
        confirm_leave_frames=_CONFIRM_FRAMES,
    )
    line_detector = LineCrossingDetector([line])
    occupancy = OccupancyCalculator(camera_id=1, zones=[zone], clock=clock)
    entry_exit = EntryExitCalculator(lines=[line], clock=clock)
    waiting = WaitingCalculator(zones=[zone], clock=clock)

    source = RecordedClipSource(
        camera_id=1,
        fps=_FPS,
        role=CameraRole.WAITING,
        frames=make_clip(n_frames),
        base_ts=_BASE_TS,
    )

    events: list[Event] = []
    while True:
        packet = source.read()
        if packet is None:
            break
        detections = detector.detect(packet.image)
        tracked = tracker.update(detections, packet.image)

        membership = zone_evaluator.membership(tracked)
        presence_result = presence.update(membership, packet.ts)

        events.extend(occupancy.process(presence_result.confirmed, packet.ts))
        crossings = line_detector.update(tracked, packet.ts)
        events.extend(entry_exit.process(crossings, packet.ts))
        events.extend(waiting.process(presence_result.transitions, packet.ts))

    store = StateStore()
    projector = StateProjector(store)
    for event in events:
        projector.handle(event)
    return events, store


# --------------------------------------------------------------------------- #
# Events produced by the pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_emits_positive_occupancy_count() -> None:
    events, _ = _run_pipeline()

    occupancy_counts = [
        e.count for e in events if isinstance(e, OccupancyUpdate)
    ]

    assert max(occupancy_counts) > 0


def test_pipeline_emits_inbound_crossing() -> None:
    events, _ = _run_pipeline()

    crossings = [e for e in events if isinstance(e, CrossingEvent)]

    assert [c.direction.value for c in crossings] == ["in"]


def test_pipeline_count_update_has_positive_net_change() -> None:
    events, _ = _run_pipeline()

    final_count = [e for e in events if isinstance(e, CountUpdate)][-1]

    assert final_count.net == 1


def test_pipeline_closes_dwell_session_with_positive_dwell() -> None:
    events, _ = _run_pipeline()

    closed = [e for e in events if isinstance(e, DwellClosed)]

    assert len(closed) == 1
    assert closed[0].dwell_s > 0.0


# --------------------------------------------------------------------------- #
# Read model after projection (the /state-style view)
# --------------------------------------------------------------------------- #
def test_state_store_reflects_final_occupancy_after_projection() -> None:
    _, store = _run_pipeline()

    occupancy = store.get_occupancy(1)

    assert occupancy is not None
    assert occupancy.count == 1


def test_state_store_reflects_net_count_after_projection() -> None:
    _, store = _run_pipeline()

    counts = store.get_counts("area-1")

    assert counts is not None
    assert counts.net == 1
    assert counts.in_count == 1
    assert counts.out_count == 0


def test_state_store_reflects_dwell_average_after_projection() -> None:
    _, store = _run_pipeline()

    waiting = store.get_waiting(1)

    assert waiting is not None
    assert waiting.avg_dwell_s > 0.0


def test_dwell_average_matches_closed_session_duration() -> None:
    events, store = _run_pipeline()

    closed = [e for e in events if isinstance(e, DwellClosed)][0]
    waiting = store.get_waiting(1)

    assert waiting is not None
    assert waiting.avg_dwell_s == pytest.approx(closed.dwell_s, abs=0.01)
