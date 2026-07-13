"""Drive the occupancy history writers on their **real background threads**.

Every other test calls ``sampler.tick()`` / ``worker.run_once()`` directly, and the autouse
conftest fixture stops the threads from launching at all — deliberately, because a 1 Hz DB
writer racing a test's own session on a shared in-memory connection makes every DB test
flaky. The cost of that isolation is that the thread loops themselves — ``_run``, the
minute-boundary flush, the stop-and-flush on shutdown — would otherwise never execute in
any test, even though they are exactly what runs in production.

This is the one test that opts back into them. It restores the real ``start`` methods,
runs the loops for a couple of seconds against the real ``RuntimeWiring``, and asserts that
rows actually land. It is bounded and polls for its result rather than sleeping a fixed
time, so it stays fast and does not flake.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from app.db.repositories.occupancy_repo import Grain, OccupancyRepository, RollupRow
from app.events.event_bus import InMemoryEventBus
from app.services.history_worker import HistoryWorker
from app.services.occupancy_sampler import OccupancySampler
from app.services.runtime_wiring import RuntimeWiring
from app.services.state_store import CameraHealthState, StateStore
from tests.conftest import REAL_HISTORY_WORKER_START, REAL_SAMPLER_START

# The sampler ticks at 1 Hz, so a couple of ticks is all we need. Generous enough not to
# flake on a loaded CI box; the test returns as soon as a row appears, never on a timer.
_DEADLINE_S = 8.0
_POLL_S = 0.2


@pytest.fixture
def real_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the conftest neutering: this test wants the genuine thread loops."""
    monkeypatch.setattr(OccupancySampler, "start", REAL_SAMPLER_START)
    monkeypatch.setattr(HistoryWorker, "start", REAL_HISTORY_WORKER_START)


def _rollup_rows(session: Session) -> list[RollupRow]:
    session.rollback()  # drop the read snapshot so we see the writer thread's commit
    return OccupancyRepository(session).query(
        Grain.MINUTE,
        datetime.now(tz=UTC) - timedelta(hours=1),
        datetime.now(tz=UTC) + timedelta(hours=1),
    )


def _await_samples(sampler: OccupancySampler, wanted: int) -> int:
    """Block until the thread has taken ``wanted`` samples into its open bucket.

    Polls the accumulator rather than the DB: the sampler only *writes* at a minute
    boundary or on ``stop()``, so waiting for a row here would wait up to a minute. This
    waits for the loop to have actually run, which is the thing under test.
    """
    deadline = time.monotonic() + _DEADLINE_S
    while time.monotonic() < deadline:
        taken = max((acc.n for acc in sampler._buckets.values()), default=0)
        if taken >= wanted:
            return taken
        time.sleep(_POLL_S)
    return 0


def test_the_real_sampler_thread_writes_a_space_rollup(
    real_threads: None, session: Session, make_camera, make_zone
) -> None:
    """The production path end to end: threads up -> live state -> a row on disk.

    Asserts the two things the thread loop owns and no other test reaches: that the loop
    actually samples on its own, and that ``stop()`` flushes the bucket still in progress
    instead of discarding it.
    """
    cam_a = make_camera(session, name="a", ip="10.0.0.1", floor="B1")
    cam_b = make_camera(session, name="b", ip="10.0.0.2", floor="B1")
    zone_a = make_zone(session, camera_id=cam_a.id, dt_space_id="b1-waiting-area")
    zone_b = make_zone(session, camera_id=cam_b.id, dt_space_id="b1-waiting-area")
    session.commit()

    store = StateStore()
    now = datetime.now(tz=UTC).timestamp()
    for camera_id in (cam_a.id, cam_b.id):
        store.set_camera_health(
            CameraHealthState(camera_id=camera_id, healthy=True, ts=now)
        )
    store.set_occupancy(zone_a.id, 3, "b1-waiting-area", now)
    store.set_occupancy(zone_b.id, 2, "b1-waiting-area", now)  # space total = 5

    wiring = RuntimeWiring(InMemoryEventBus(), store)
    wiring.start()
    try:
        taken = _await_samples(wiring.occupancy_sampler, wanted=2)
        assert taken >= 2, "the sampler thread never ran its loop"
        # Nothing is on disk yet: the bucket is still open.
        assert _rollup_rows(session) == []
    finally:
        wiring.close_sync()  # this is what must flush the open minute

    (row,) = _rollup_rows(session)
    assert row.space_id == "b1-waiting-area"
    assert row.floor == "B1"
    assert row.peak == 5  # the two cameras' zones, summed into one space
    assert row.samples >= 2
    assert (row.cameras_healthy, row.cameras_total) == (2, 2)
