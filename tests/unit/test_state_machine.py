"""Unit tests for the debounced zone-presence state machine (HLD 5.7 / 6.3).

:class:`ZonePresenceTracker` smooths noisy per-frame membership with hysteresis:
a track is confirmed present only after a run of consecutive present frames and
confirmed absent only after a run of consecutive absent frames. These tests feed
it raw ``zone_id -> {track_id}`` membership directly (the real per-frame output
shape) and assert the observable confirmations, transitions, and derived
``ZoneState`` — no mocking of the state machine itself.

All timestamps come from the deterministic :class:`FakeClock` fixture so the
``ts`` stamped on transitions is reproducible without wall-clock time.
"""

from __future__ import annotations

from app.domain.models import ZoneState
from app.localisation.state_machine import ZonePresenceTracker
from app.utils.clock import FakeClock
from tests.fixtures.specs import make_zone_spec

_ZONE_ID = 1
_TRACK_ID = 7


def _tracker(enter: int, leave: int) -> ZonePresenceTracker:
    zone = make_zone_spec(id=_ZONE_ID)
    return ZonePresenceTracker(
        [zone], confirm_enter_frames=enter, confirm_leave_frames=leave
    )


def _present() -> dict[int, set[int]]:
    return {_ZONE_ID: {_TRACK_ID}}


def _absent() -> dict[int, set[int]]:
    return {_ZONE_ID: set()}


# --------------------------------------------------------------------------- #
# Confirmed entry
# --------------------------------------------------------------------------- #
def test_track_not_confirmed_before_enough_present_frames(
    fake_clock: FakeClock,
) -> None:
    tracker = _tracker(enter=3, leave=2)

    result = tracker.update(_present(), ts=fake_clock.now())
    fake_clock.advance(1.0)
    result = tracker.update(_present(), ts=fake_clock.now())

    # Two present frames with a threshold of three: not yet confirmed.
    assert result.confirmed[_ZONE_ID] == set()


def test_track_confirmed_after_consecutive_present_frames_emits_single_enter(
    fake_clock: FakeClock,
) -> None:
    tracker = _tracker(enter=3, leave=2)
    enter_transitions: list = []

    for _ in range(3):
        result = tracker.update(_present(), ts=fake_clock.now())
        enter_transitions.extend(
            t for t in result.transitions if t.kind == "enter"
        )
        fake_clock.advance(1.0)

    assert result.confirmed[_ZONE_ID] == {_TRACK_ID}
    assert len(enter_transitions) == 1


def test_enter_transition_carries_zone_track_and_timestamp(
    fake_clock: FakeClock,
) -> None:
    tracker = _tracker(enter=2, leave=2)
    fake_clock.set(2000.0)

    tracker.update(_present(), ts=fake_clock.now())
    fake_clock.advance(5.0)
    confirm_ts = fake_clock.now()
    result = tracker.update(_present(), ts=confirm_ts)

    assert len(result.transitions) == 1
    transition = result.transitions[0]
    assert (
        transition.zone_id,
        transition.track_id,
        transition.kind,
        transition.ts,
    ) == (_ZONE_ID, _TRACK_ID, "enter", confirm_ts)


# --------------------------------------------------------------------------- #
# Confirmed leave
# --------------------------------------------------------------------------- #
def test_track_leaves_only_after_consecutive_absent_frames_emits_single_leave(
    fake_clock: FakeClock,
) -> None:
    tracker = _tracker(enter=1, leave=3)
    # Confirm presence first (enter=1 -> confirmed on first present frame).
    tracker.update(_present(), ts=fake_clock.now())
    fake_clock.advance(1.0)

    leave_transitions: list = []
    last_result = None
    for _ in range(3):
        last_result = tracker.update(_absent(), ts=fake_clock.now())
        leave_transitions.extend(
            t for t in last_result.transitions if t.kind == "leave"
        )
        fake_clock.advance(1.0)

    assert last_result is not None
    assert last_result.confirmed[_ZONE_ID] == set()
    assert len(leave_transitions) == 1


def test_track_still_confirmed_before_enough_absent_frames(
    fake_clock: FakeClock,
) -> None:
    tracker = _tracker(enter=1, leave=3)
    tracker.update(_present(), ts=fake_clock.now())
    fake_clock.advance(1.0)

    # Two absent frames against a leave threshold of three: still confirmed.
    tracker.update(_absent(), ts=fake_clock.now())
    fake_clock.advance(1.0)
    result = tracker.update(_absent(), ts=fake_clock.now())

    assert result.confirmed[_ZONE_ID] == {_TRACK_ID}


# --------------------------------------------------------------------------- #
# Flicker rejection
# --------------------------------------------------------------------------- #
def test_single_frame_dropout_does_not_flicker_leave_or_enter(
    fake_clock: FakeClock,
) -> None:
    tracker = _tracker(enter=1, leave=3)
    all_transitions: list = []

    def step(membership: dict[int, set[int]]) -> None:
        result = tracker.update(membership, ts=fake_clock.now())
        all_transitions.extend(result.transitions)
        fake_clock.advance(1.0)

    step(_present())  # frame 0: confirmed enter (1 transition)
    step(_present())  # frame 1
    step(_absent())   # frame 2: single-frame dropout (< leave threshold)
    step(_present())  # frame 3: present again, already confirmed
    last_result = tracker.update(_present(), ts=fake_clock.now())

    # Only the original enter ever fired; the dropout produced no leave and the
    # re-appearance produced no second enter.
    kinds = [t.kind for t in all_transitions] + [
        t.kind for t in last_result.transitions
    ]
    assert kinds == ["enter"]
    assert last_result.confirmed[_ZONE_ID] == {_TRACK_ID}


# --------------------------------------------------------------------------- #
# ZoneState lifecycle
# --------------------------------------------------------------------------- #
def test_zone_state_progresses_vacant_entering_occupied_leaving_vacant(
    fake_clock: FakeClock,
) -> None:
    tracker = _tracker(enter=2, leave=2)
    states: list[ZoneState] = []

    def step(membership: dict[int, set[int]]) -> ZoneState:
        result = tracker.update(membership, ts=fake_clock.now())
        fake_clock.advance(1.0)
        return result.zone_state[_ZONE_ID]

    states.append(step(_present()))  # 1st present: entering (streak 1 < 2)
    states.append(step(_present()))  # 2nd present: confirmed -> occupied
    states.append(step(_absent()))   # 1st absent on confirmed: leaving
    states.append(step(_absent()))   # 2nd absent: confirmed cleared -> vacant

    assert states == [
        ZoneState.ENTERING,
        ZoneState.OCCUPIED,
        ZoneState.LEAVING,
        ZoneState.VACANT,
    ]


def test_zone_state_is_vacant_with_no_tracks(fake_clock: FakeClock) -> None:
    tracker = _tracker(enter=2, leave=2)
    result = tracker.update(_absent(), ts=fake_clock.now())
    assert result.zone_state[_ZONE_ID] is ZoneState.VACANT
