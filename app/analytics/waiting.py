"""Waiting-time / dwell calculator (HLD 7).

Consumes confirmed :class:`app.localisation.state_machine.Transition` events for
waiting zones and tracks how long each track stays in a zone:

* An ``"enter"`` transition opens a dwell session for ``(zone, track)``.
* A ``"leave"`` transition closes it, emitting a
  :class:`app.events.events.DwellClosed` with the measured dwell and recording
  that dwell into a time-windowed deque per zone for the rolling average.

After processing a frame's transitions, one
:class:`app.events.events.WaitingUpdate` is emitted per affected zone carrying
the current number of open sessions and the rolling average dwell over the
configured window. :meth:`tick` lets the engine emit a periodic update for a
quiet zone (no transitions) so the rolling average keeps trending down as old
samples age out of the window.
"""

from __future__ import annotations

from collections import deque

from app.analytics.base import BaseCalculator
from app.domain.interfaces import Clock
from app.domain.models import ZoneSpec
from app.events.events import DwellClosed, Event, WaitingUpdate
from app.localisation.state_machine import Transition
from app.utils.logging import get_logger

logger = get_logger(__name__)


class WaitingCalculator(BaseCalculator):
    """Tracks dwell sessions and rolling average waiting time per zone.

    Holds, per zone, the set of currently-open dwell sessions (keyed by track)
    and a time-ordered deque of recently-closed dwell samples used for the
    rolling average. Samples older than ``avg_window_seconds`` relative to the
    latest seen timestamp are evicted before each average is computed.
    """

    def __init__(
        self,
        zones: list[ZoneSpec],
        clock: Clock,
        avg_window_seconds: float = 600.0,
    ) -> None:
        """Initialise the calculator.

        Args:
            zones: Waiting zone specifications for the camera. May be empty.
            clock: Shared pipeline time source.
            avg_window_seconds: Rolling-average window in seconds. Closed dwell
                samples older than this (relative to the newest observed
                timestamp) are dropped before averaging.
        """
        super().__init__(clock)
        self._zones: dict[int, ZoneSpec] = {zone.id: zone for zone in zones}
        self._avg_window_seconds = avg_window_seconds
        # zone_id -> track_id -> enter timestamp (open sessions).
        self._open: dict[int, dict[int, float]] = {
            zone_id: {} for zone_id in self._zones
        }
        # zone_id -> deque of (closed_at_ts, dwell_seconds) within the window.
        self._samples: dict[int, deque[tuple[float, float]]] = {
            zone_id: deque() for zone_id in self._zones
        }

    def process(self, transitions: list[Transition], ts: float) -> list[Event]:
        """Process confirmed presence transitions for one frame.

        Args:
            transitions: Confirmed enter/leave transitions for this frame.
                Transitions for zones not configured on this calculator are
                ignored.
            ts: Frame timestamp (float epoch seconds), stamped on emitted events.

        Returns:
            Zero or more :class:`DwellClosed` events (one per closed session),
            followed by one :class:`WaitingUpdate` per affected zone.
        """
        events: list[Event] = []
        affected: set[int] = set()

        for transition in transitions:
            zone_id = transition.zone_id
            if zone_id not in self._zones:
                continue
            affected.add(zone_id)
            if transition.kind == "enter":
                self._open[zone_id][transition.track_id] = transition.ts
            elif transition.kind == "leave":
                closed = self._close_session(zone_id, transition, ts)
                if closed is not None:
                    events.append(closed)
            else:
                logger.warning(
                    "ignoring transition with unknown kind",
                    extra={"zone_id": zone_id, "event": transition.kind},
                )

        for zone_id in affected:
            events.append(self._build_update(zone_id, ts))
        return events

    def tick(self, ts: float) -> list[WaitingUpdate]:
        """Emit a periodic update per zone with no transitions this interval.

        Lets the rolling average decay as samples age out even when no track
        enters or leaves. Intended to be called by the engine on a timer.

        Args:
            ts: Current timestamp (float epoch seconds).

        Returns:
            One :class:`WaitingUpdate` per configured zone.
        """
        return [self._build_update(zone_id, ts) for zone_id in self._zones]

    def _close_session(
        self, zone_id: int, transition: Transition, ts: float
    ) -> DwellClosed | None:
        """Close an open dwell session, recording its sample. ``None`` if none open."""
        enter_ts = self._open[zone_id].pop(transition.track_id, None)
        if enter_ts is None:
            # A leave without a matching enter (e.g. started occupied) — nothing
            # to measure; do not invent a dwell.
            return None
        dwell_s = max(0.0, ts - enter_ts)
        self._samples[zone_id].append((ts, dwell_s))
        return DwellClosed(
            ts=ts,
            zone_id=zone_id,
            track_ref=transition.track_id,
            enter_ts=enter_ts,
            leave_ts=ts,
            dwell_s=dwell_s,
        )

    def _build_update(self, zone_id: int, ts: float) -> WaitingUpdate:
        """Construct a :class:`WaitingUpdate` for a zone at time ``ts``."""
        avg = self._rolling_average(zone_id, ts)
        return WaitingUpdate(
            ts=ts,
            zone_id=zone_id,
            current_waits=len(self._open[zone_id]),
            avg_dwell_s=avg,
            dt_space_id=self._zones[zone_id].dt_space_id,
        )

    def _rolling_average(self, zone_id: int, ts: float) -> float:
        """Average dwell over the window, evicting samples older than the window."""
        samples = self._samples[zone_id]
        cutoff = ts - self._avg_window_seconds
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        if not samples:
            return 0.0
        return sum(dwell for _, dwell in samples) / len(samples)
