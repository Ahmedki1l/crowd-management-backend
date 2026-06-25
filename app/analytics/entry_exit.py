"""Entry/exit counting calculator (HLD 7).

Consumes :class:`app.localisation.lines.Crossing` events from the line-crossing
detector and maintains running IN/OUT totals per area and per line across calls.
For every crossing it emits a :class:`app.events.events.CrossingEvent`, then one
:class:`app.events.events.CountUpdate` summarising each area affected on this
frame. Totals are cumulative for the lifetime of the calculator instance.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analytics.base import BaseCalculator
from app.domain.interfaces import Clock
from app.domain.models import CrossingDirection, LineSpec
from app.events.events import CountUpdate, CrossingEvent, Event
from app.localisation.lines import Crossing
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class _Counts:
    """Mutable IN/OUT tally for one area or one line."""

    in_count: int = 0
    out_count: int = 0

    @property
    def net(self) -> int:
        return self.in_count - self.out_count

    def add(self, direction: CrossingDirection) -> None:
        """Increment the tally for one crossing in the given direction."""
        if direction is CrossingDirection.IN:
            self.in_count += 1
        else:
            self.out_count += 1


class EntryExitCalculator(BaseCalculator):
    """Maintains cumulative entry/exit counts per area and per line.

    Holds running per-area and per-line IN/OUT totals. Each call processes the
    crossings detected on one frame, emitting a crossing event for each and a
    single count update per affected area. The per-line totals are kept as state
    and exposed via :meth:`line_counts` for callers that need a finer breakdown.
    """

    def __init__(self, lines: list[LineSpec], clock: Clock) -> None:
        """Initialise the calculator.

        Args:
            lines: Counting-line specifications. Used to resolve a line's
                ``dt_space_id`` for emitted area updates. May be empty.
            clock: Shared pipeline time source.
        """
        super().__init__(clock)
        self._line_dt_space: dict[int, str | None] = {
            line.id: line.dt_space_id for line in lines
        }
        # area_id -> cumulative IN/OUT; line_id -> cumulative IN/OUT.
        self._area_counts: dict[str, _Counts] = {}
        self._line_counts: dict[int, _Counts] = {}

    def process(self, crossings: list[Crossing], ts: float) -> list[Event]:
        """Process the crossings detected on one frame.

        Args:
            crossings: Crossings from the line-crossing detector for this frame.
            ts: Frame timestamp (float epoch seconds), stamped on emitted events.

        Returns:
            A list interleaving one :class:`CrossingEvent` per crossing followed
            by one :class:`CountUpdate` per area touched this frame (empty when
            there were no crossings).
        """
        events: list[Event] = []
        affected_areas: dict[str, str | None] = {}

        for crossing in crossings:
            self._area_counts.setdefault(crossing.area_id, _Counts()).add(
                crossing.direction
            )
            self._line_counts.setdefault(crossing.line_id, _Counts()).add(
                crossing.direction
            )
            affected_areas[crossing.area_id] = crossing.dt_space_id
            events.append(
                CrossingEvent(
                    ts=ts,
                    line_id=crossing.line_id,
                    area_id=crossing.area_id,
                    direction=crossing.direction,
                    track_ref=crossing.track_id,
                    dt_space_id=crossing.dt_space_id,
                )
            )

        for area_id, dt_space_id in affected_areas.items():
            counts = self._area_counts[area_id]
            events.append(
                CountUpdate(
                    ts=ts,
                    area_id=area_id,
                    in_count=counts.in_count,
                    out_count=counts.out_count,
                    net=counts.net,
                    line_id=None,
                    dt_space_id=dt_space_id,
                )
            )
        return events

    def area_counts(self, area_id: str) -> tuple[int, int, int]:
        """Return ``(in, out, net)`` cumulative totals for an area.

        Args:
            area_id: Area identifier.

        Returns:
            ``(in_count, out_count, net)``; zeros if the area has no crossings yet.
        """
        counts = self._area_counts.get(area_id)
        if counts is None:
            return (0, 0, 0)
        return (counts.in_count, counts.out_count, counts.net)

    def line_counts(self, line_id: int) -> tuple[int, int, int]:
        """Return ``(in, out, net)`` cumulative totals for a single line.

        Args:
            line_id: Counting-line identifier.

        Returns:
            ``(in_count, out_count, net)``; zeros if the line has no crossings yet.
        """
        counts = self._line_counts.get(line_id)
        if counts is None:
            return (0, 0, 0)
        return (counts.in_count, counts.out_count, counts.net)
