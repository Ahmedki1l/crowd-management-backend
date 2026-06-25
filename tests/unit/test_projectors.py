"""Tests for the event projectors (HLD 6.6).

``StateProjector`` is exercised synchronously against the real ``StateStore``.
``PersistenceProjector`` is exercised end-to-end against the per-test in-memory
DB: events are enqueued, the worker drains them on ``stop()`` (which joins the
queue), and the resulting rows are read back through a session.

The persistence worker writes through ``session_scope()``, which is bound to the
same in-memory engine configured by the ``_isolated_environment`` fixture (a
SQLite ``StaticPool`` shared across threads), so the worker thread and the test
session see the same database.
"""

from __future__ import annotations

from datetime import UTC

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.timeseries import CrossingEvent as CrossingRow
from app.db.models.timeseries import DwellSession as DwellRow
from app.db.models.timeseries import OccupancySample
from app.domain.models import CrossingDirection
from app.events.events import (
    CameraHealth,
    CountUpdate,
    CrossingEvent,
    DwellClosed,
    OccupancyUpdate,
    WaitingUpdate,
)
from app.services.projectors import PersistenceProjector, StateProjector
from app.services.state_store import StateStore
from app.utils.timeutil import to_epoch


# --------------------------------------------------------------------------- #
# StateProjector — synchronous current-state writes
# --------------------------------------------------------------------------- #
def test_state_projector_records_occupancy_in_store() -> None:
    store = StateStore()
    StateProjector(store).handle(
        OccupancyUpdate(ts=1000.0, zone_id=7, camera_id=3, count=4, dt_space_id="space-7")
    )

    state = store.get_occupancy(7)
    assert state is not None
    assert state.count == 4
    assert state.dt_space_id == "space-7"


def test_state_projector_records_area_counts_in_store() -> None:
    store = StateStore()
    StateProjector(store).handle(
        CountUpdate(ts=1000.0, area_id="lobby", in_count=9, out_count=4, net=5, line_id=2)
    )

    state = store.get_counts("lobby")
    assert state is not None
    assert (state.in_count, state.out_count, state.net) == (9, 4, 5)


def test_state_projector_attributes_counts_to_triggering_line() -> None:
    store = StateStore()
    StateProjector(store).handle(
        CountUpdate(ts=1000.0, area_id="lobby", in_count=9, out_count=4, net=5, line_id=2)
    )

    state = store.get_counts("lobby")
    assert state is not None
    assert state.lines[2].in_count == 9
    assert state.lines[2].out_count == 4


def test_state_projector_records_waiting_in_store() -> None:
    store = StateStore()
    StateProjector(store).handle(
        WaitingUpdate(ts=1000.0, zone_id=5, current_waits=3, avg_dwell_s=12.5, dt_space_id="q")
    )

    state = store.get_waiting(5)
    assert state is not None
    assert state.current_waits == 3
    assert state.avg_dwell_s == 12.5


def test_state_projector_records_camera_health_in_store() -> None:
    store = StateStore()
    StateProjector(store).handle(
        CameraHealth(
            ts=1000.0, camera_id=11, fps=24.0, last_frame_age_s=0.2, queue_depth=1, healthy=False
        )
    )

    state = store.get_camera_health(11)
    assert state is not None
    assert state.fps == 24.0
    assert state.healthy is False


def test_state_projector_ignores_unprojected_event_types() -> None:
    store = StateStore()
    # A crossing is persisted by the PersistenceProjector, not projected to state.
    StateProjector(store).handle(
        CrossingEvent(
            ts=1000.0, line_id=1, area_id="lobby", direction=CrossingDirection.IN, track_ref=8
        )
    )

    assert store.all_counts() == []
    assert store.all_occupancy() == []


# --------------------------------------------------------------------------- #
# PersistenceProjector — async durable writes
# --------------------------------------------------------------------------- #
def test_persistence_projector_persists_crossing_event_row(
    session: Session, make_camera, make_line
) -> None:
    camera = make_camera(session)
    line = make_line(session, camera.id, area_id="lobby")

    projector = PersistenceProjector(persist_interval=0.0)
    projector.start()
    projector.handle(
        CrossingEvent(
            ts=1000.0,
            line_id=line.id,
            area_id="lobby",
            direction=CrossingDirection.OUT,
            track_ref=42,
        )
    )
    projector.stop()  # drains the queue and joins the worker

    rows = session.scalars(select(CrossingRow)).all()
    assert len(rows) == 1
    assert rows[0].line_id == line.id
    assert rows[0].direction == CrossingDirection.OUT.value
    assert rows[0].track_ref == 42


def test_persistence_projector_persists_dwell_session_row(
    session: Session, make_camera, make_zone
) -> None:
    camera = make_camera(session)
    zone = make_zone(session, camera.id)

    projector = PersistenceProjector(persist_interval=0.0)
    projector.start()
    projector.handle(
        DwellClosed(
            ts=1030.0,
            zone_id=zone.id,
            track_ref=7,
            enter_ts=1000.0,
            leave_ts=1030.0,
            dwell_s=30.0,
        )
    )
    projector.stop()

    rows = session.scalars(select(DwellRow)).all()
    assert len(rows) == 1
    assert rows[0].zone_id == zone.id
    assert rows[0].track_ref == 7
    assert rows[0].dwell_s == 30.0


def test_persistence_projector_persists_dwell_enter_and_leave_timestamps(
    session: Session, make_camera, make_zone
) -> None:
    camera = make_camera(session)
    zone = make_zone(session, camera.id)

    projector = PersistenceProjector(persist_interval=0.0)
    projector.start()
    projector.handle(
        DwellClosed(
            ts=1030.0,
            zone_id=zone.id,
            track_ref=7,
            enter_ts=1000.0,
            leave_ts=1030.0,
            dwell_s=30.0,
        )
    )
    projector.stop()

    row = session.scalars(select(DwellRow)).one()
    # SQLite drops tzinfo on round-trip; the DB contract is UTC, so compare epochs.
    assert to_epoch(row.enter_ts.replace(tzinfo=UTC)) == 1000.0
    assert to_epoch(row.leave_ts.replace(tzinfo=UTC)) == 1030.0


def test_persistence_projector_throttles_rapid_occupancy_samples_for_one_zone(
    session: Session, make_camera, make_zone
) -> None:
    camera = make_camera(session)
    zone = make_zone(session, camera.id)

    # interval=5s: two updates 1s apart for the same zone must write at most one.
    projector = PersistenceProjector(persist_interval=5.0)
    projector.start()
    projector.handle(OccupancyUpdate(ts=1000.0, zone_id=zone.id, camera_id=camera.id, count=2))
    projector.handle(OccupancyUpdate(ts=1001.0, zone_id=zone.id, camera_id=camera.id, count=3))
    projector.stop()

    rows = session.scalars(
        select(OccupancySample).where(OccupancySample.zone_id == zone.id)
    ).all()
    assert len(rows) == 1
    assert rows[0].count == 2  # the first sample wins; the rapid second is dropped


def test_persistence_projector_persists_occupancy_again_after_interval_elapses(
    session: Session, make_camera, make_zone
) -> None:
    camera = make_camera(session)
    zone = make_zone(session, camera.id)

    projector = PersistenceProjector(persist_interval=5.0)
    projector.start()
    projector.handle(OccupancyUpdate(ts=1000.0, zone_id=zone.id, camera_id=camera.id, count=2))
    # 6s later — past the throttle interval, so this one is persisted too.
    projector.handle(OccupancyUpdate(ts=1006.0, zone_id=zone.id, camera_id=camera.id, count=5))
    projector.stop()

    rows = session.scalars(
        select(OccupancySample)
        .where(OccupancySample.zone_id == zone.id)
        .order_by(OccupancySample.ts)
    ).all()
    assert [r.count for r in rows] == [2, 5]
