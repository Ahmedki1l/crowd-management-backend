"""Tests for the event projectors (HLD 6.6).

``StateProjector`` is exercised synchronously against the real ``StateStore``.
``PersistenceProjector`` is exercised end-to-end against the per-test in-memory
DB: events are enqueued, the worker drains them on ``stop()`` (which joins the
queue), and the resulting rows are read back through a session.

Occupancy history is deliberately absent here: it is no longer written by a projector.
A change-driven event stream cannot yield a time-weighted average, so occupancy is
sampled at a fixed rate instead — see ``tests/unit/test_occupancy_sampler.py``.

The persistence worker writes through ``session_scope()``, which is bound to the
same in-memory engine configured by the ``_isolated_environment`` fixture (a
SQLite ``StaticPool`` shared across threads), so the worker thread and the test
session see the same database.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.timeseries import CrossingEvent as CrossingRow
from app.domain.models import CrossingDirection
from app.events.events import (
    CameraHealth,
    CountUpdate,
    CrossingEvent,
    OccupancyUpdate,
)
from app.services.projectors import PersistenceProjector, StateProjector
from app.services.state_store import StateStore


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

    projector = PersistenceProjector()
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
