"""Debounced zone-presence state machine (HLD 5.7 / 6.3).

Raw per-frame membership (from :class:`app.localisation.zones.ZoneEvaluator`) is
noisy: a track may flicker in and out of a polygon across consecutive frames.
:class:`ZonePresenceTracker` smooths this with hysteresis — a track only becomes
*confirmed present* after a run of consecutive present frames and *confirmed
absent* after a run of consecutive absent frames — and derives a per-zone
:class:`~app.domain.models.ZoneState`.

The tracker is pure and deterministic: ``update`` mutates only the tracker's own
counters and returns an immutable :class:`PresenceResult` describing the frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.domain.models import ZoneSpec, ZoneState


@dataclass(frozen=True, slots=True)
class Transition:
    """A confirmed presence change for one track in one zone."""

    zone_id: int
    track_id: int
    kind: str  # "enter" | "leave"
    ts: float


@dataclass(frozen=True, slots=True)
class PresenceResult:
    """Outcome of one :meth:`ZonePresenceTracker.update` call.

    Attributes:
        confirmed: ``zone_id -> set of confirmed-present track_ids`` after this
            frame.
        transitions: Confirmed enter/leave events that fired on this frame.
        zone_state: Derived :class:`ZoneState` for every configured zone.
    """

    confirmed: dict[int, set[int]]
    transitions: list[Transition]
    zone_state: dict[int, ZoneState]


@dataclass(slots=True)
class _TrackCounters:
    """Mutable hysteresis counters for one (zone, track) pair.

    ``present_streak`` / ``absent_streak`` count consecutive frames; only one is
    non-zero at a time. ``confirmed`` reflects the debounced membership decision.
    """

    present_streak: int = 0
    absent_streak: int = 0
    confirmed: bool = False


class ZonePresenceTracker:
    """Hysteresis-debounced presence tracker across all zones of one camera.

    A track is confirmed present once it has been observed for
    ``confirm_enter_frames`` consecutive frames inside a zone, and confirmed
    absent once an already-present track has been missing for
    ``confirm_leave_frames`` consecutive frames. Each confirmation emits a
    :class:`Transition`.
    """

    def __init__(
        self,
        zones: Sequence[ZoneSpec],
        confirm_enter_frames: int,
        confirm_leave_frames: int,
    ) -> None:
        """Initialise the tracker.

        Args:
            zones: Zone specifications for the camera. May be empty.
            confirm_enter_frames: Consecutive present frames required to confirm
                entry (clamped to a minimum of 1).
            confirm_leave_frames: Consecutive absent frames required to confirm a
                departure (clamped to a minimum of 1).
        """
        self._zone_ids: tuple[int, ...] = tuple(zone.id for zone in zones)
        self._confirm_enter_frames = max(1, confirm_enter_frames)
        self._confirm_leave_frames = max(1, confirm_leave_frames)
        # zone_id -> track_id -> counters
        self._counters: dict[int, dict[int, _TrackCounters]] = {
            zone_id: {} for zone_id in self._zone_ids
        }

    def update(
        self, raw_membership: dict[int, set[int]], ts: float
    ) -> PresenceResult:
        """Advance the state machine by one frame.

        Args:
            raw_membership: ``zone_id -> set of track_ids`` from the per-frame
                zone evaluator. Missing zones are treated as empty.
            ts: Epoch-seconds timestamp of the frame (stamped onto transitions).

        Returns:
            A :class:`PresenceResult` for this frame.
        """
        transitions: list[Transition] = []
        confirmed: dict[int, set[int]] = {}
        zone_state: dict[int, ZoneState] = {}

        for zone_id in self._zone_ids:
            present_ids = raw_membership.get(zone_id, set())
            zone_counters = self._counters[zone_id]
            zone_transitions = self._advance_zone(
                zone_id, zone_counters, present_ids, ts
            )
            transitions.extend(zone_transitions)
            confirmed[zone_id] = {
                track_id
                for track_id, counters in zone_counters.items()
                if counters.confirmed
            }
            zone_state[zone_id] = self._derive_state(zone_counters)

        return PresenceResult(
            confirmed=confirmed, transitions=transitions, zone_state=zone_state
        )

    def _advance_zone(
        self,
        zone_id: int,
        zone_counters: dict[int, _TrackCounters],
        present_ids: set[int],
        ts: float,
    ) -> list[Transition]:
        """Update counters for one zone and collect any confirmed transitions."""
        transitions: list[Transition] = []

        # Step every track currently present in the zone this frame.
        for track_id in present_ids:
            counters = zone_counters.get(track_id)
            if counters is None:
                counters = _TrackCounters()
                zone_counters[track_id] = counters
            counters.present_streak += 1
            counters.absent_streak = 0
            if (
                not counters.confirmed
                and counters.present_streak >= self._confirm_enter_frames
            ):
                counters.confirmed = True
                transitions.append(
                    Transition(zone_id, track_id, "enter", ts)
                )

        # Step every known track that was NOT present this frame.
        for track_id in list(zone_counters.keys()):
            if track_id in present_ids:
                continue
            counters = zone_counters[track_id]
            counters.absent_streak += 1
            counters.present_streak = 0
            if (
                counters.confirmed
                and counters.absent_streak >= self._confirm_leave_frames
            ):
                counters.confirmed = False
                transitions.append(
                    Transition(zone_id, track_id, "leave", ts)
                )
                # Fully gone: drop the counter so memory does not grow unbounded.
                del zone_counters[track_id]
            elif (
                not counters.confirmed
                and counters.absent_streak >= self._confirm_leave_frames
            ):
                # Never confirmed and now absent long enough — discard noise.
                del zone_counters[track_id]

        return transitions

    @staticmethod
    def _derive_state(zone_counters: dict[int, _TrackCounters]) -> ZoneState:
        """Map the zone's per-track counters to a single :class:`ZoneState`.

        Precedence (most-active wins): a confirmed member accumulating absence
        marks the zone ``LEAVING``; otherwise any confirmed member makes it
        ``OCCUPIED``; otherwise a track accumulating toward entry makes it
        ``ENTERING``; otherwise ``VACANT``.
        """
        has_confirmed = False
        has_leaving = False
        has_entering = False
        for counters in zone_counters.values():
            if counters.confirmed:
                has_confirmed = True
                if counters.absent_streak > 0:
                    has_leaving = True
            elif counters.present_streak > 0:
                has_entering = True

        if has_leaving:
            return ZoneState.LEAVING
        if has_confirmed:
            return ZoneState.OCCUPIED
        if has_entering:
            return ZoneState.ENTERING
        return ZoneState.VACANT
