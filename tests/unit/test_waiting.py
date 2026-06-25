"""Unit tests for :class:`app.analytics.waiting.WaitingCalculator`.

The calculator opens a dwell session on an ``enter`` transition and closes it on a
``leave``, emitting a ``DwellClosed`` with the measured dwell and a per-zone
``WaitingUpdate`` carrying the open-session count and rolling-average dwell. The
dwell length is ``leave_frame_ts - enter_transition_ts``; tests use explicit
timestamps (deterministic, no real sleeps) and the ``FakeClock`` time source.
"""

from __future__ import annotations

from app.analytics.waiting import WaitingCalculator
from app.domain.models import ZoneType
from app.events.events import DwellClosed, WaitingUpdate
from app.localisation.state_machine import Transition
from tests.fixtures.specs import make_zone_spec


def _enter(zone_id, track_id, ts):
    return Transition(zone_id=zone_id, track_id=track_id, kind="enter", ts=ts)


def _leave(zone_id, track_id, ts):
    return Transition(zone_id=zone_id, track_id=track_id, kind="leave", ts=ts)


def test_enter_then_leave_after_T_seconds_closes_dwell_of_T(fake_clock):
    zone = make_zone_spec(id=4, type=ZoneType.WAITING)
    calc = WaitingCalculator(zones=[zone], clock=fake_clock)
    enter_ts = fake_clock.now()
    T = 45.0

    calc.process([_enter(4, 1, ts=enter_ts)], ts=enter_ts)
    fake_clock.advance(T)
    leave_events = calc.process([_leave(4, 1, ts=fake_clock.now())], ts=fake_clock.now())

    (dwell,) = [e for e in leave_events if isinstance(e, DwellClosed)]
    assert dwell.dwell_s == T


def test_closed_dwell_emits_dwell_closed_with_zone_and_track(fake_clock):
    zone = make_zone_spec(id=4, type=ZoneType.WAITING)
    calc = WaitingCalculator(zones=[zone], clock=fake_clock)

    calc.process([_enter(4, 7, ts=1000.0)], ts=1000.0)
    events = calc.process([_leave(4, 7, ts=1020.0)], ts=1020.0)

    (dwell,) = [e for e in events if isinstance(e, DwellClosed)]
    assert isinstance(dwell, DwellClosed)
    assert (dwell.zone_id, dwell.track_ref, dwell.enter_ts, dwell.leave_ts) == (
        4,
        7,
        1000.0,
        1020.0,
    )


def test_current_waits_reflects_open_sessions(fake_clock):
    zone = make_zone_spec(id=4, type=ZoneType.WAITING)
    calc = WaitingCalculator(zones=[zone], clock=fake_clock)

    events = calc.process(
        [_enter(4, 1, ts=1000.0), _enter(4, 2, ts=1000.0)], ts=1000.0
    )

    (update,) = [e for e in events if isinstance(e, WaitingUpdate)]
    assert update.current_waits == 2


def test_current_waits_drops_when_a_session_closes(fake_clock):
    zone = make_zone_spec(id=4, type=ZoneType.WAITING)
    calc = WaitingCalculator(zones=[zone], clock=fake_clock)

    calc.process([_enter(4, 1, ts=1000.0), _enter(4, 2, ts=1000.0)], ts=1000.0)
    events = calc.process([_leave(4, 1, ts=1010.0)], ts=1010.0)

    (update,) = [e for e in events if isinstance(e, WaitingUpdate)]
    assert update.current_waits == 1


def test_avg_dwell_is_rolling_average_of_closed_sessions(fake_clock):
    zone = make_zone_spec(id=4, type=ZoneType.WAITING)
    calc = WaitingCalculator(zones=[zone], clock=fake_clock, avg_window_seconds=600.0)

    # Session A dwells 20s, session B dwells 40s -> average 30s.
    calc.process([_enter(4, 1, ts=1000.0), _enter(4, 2, ts=1000.0)], ts=1000.0)
    calc.process([_leave(4, 1, ts=1020.0)], ts=1020.0)
    events = calc.process([_leave(4, 2, ts=1040.0)], ts=1040.0)

    (update,) = [e for e in events if isinstance(e, WaitingUpdate)]
    assert update.avg_dwell_s == 30.0


def test_leave_without_matching_enter_emits_no_dwell(fake_clock):
    zone = make_zone_spec(id=4, type=ZoneType.WAITING)
    calc = WaitingCalculator(zones=[zone], clock=fake_clock)

    events = calc.process([_leave(4, 99, ts=1000.0)], ts=1000.0)

    assert not [e for e in events if isinstance(e, DwellClosed)]
