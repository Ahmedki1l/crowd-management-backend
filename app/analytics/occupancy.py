"""Occupancy calculator (HLD 7).

Turns the debounced per-zone confirmed membership produced upstream by
:class:`app.localisation.state_machine.ZonePresenceTracker` into
:class:`app.events.events.OccupancyUpdate` events. An update is emitted for a
zone only when its confirmed count differs from the last value emitted for that
zone, so the wire carries one event per real change rather than one per frame.
"""

from __future__ import annotations

from app.analytics.base import BaseCalculator
from app.domain.interfaces import Clock
from app.domain.models import ZoneSpec
from app.events.events import OccupancyUpdate
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OccupancyCalculator(BaseCalculator):
    """Emits occupancy counts per zone on change only.

    The calculator holds the zone specifications for one camera and the last
    emitted count per zone. Debouncing of raw membership flicker happens upstream
    in the presence state machine; this calculator additionally suppresses
    no-change frames so consumers see one event per genuine count transition.
    """

    def __init__(self, camera_id: int, zones: list[ZoneSpec], clock: Clock) -> None:
        """Initialise the calculator.

        Args:
            camera_id: Camera these zones belong to.
            zones: Occupancy zone specifications for the camera. May be empty.
            clock: Shared pipeline time source.
        """
        super().__init__(clock)
        self._camera_id = camera_id
        self._zones: dict[int, ZoneSpec] = {zone.id: zone for zone in zones}
        # zone_id -> last count emitted for that zone (None until first emit).
        self._last_count: dict[int, int] = {}

    def process(
        self, confirmed: dict[int, set[int]], ts: float
    ) -> list[OccupancyUpdate]:
        """Compute occupancy updates for the current frame.

        Args:
            confirmed: ``zone_id -> set of confirmed-present track_ids`` from the
                presence state machine. Zones absent from the mapping are treated
                as empty.
            ts: Frame timestamp (float epoch seconds), stamped on emitted events.

        Returns:
            One :class:`OccupancyUpdate` per zone whose confirmed count changed
            since its previous emission (possibly empty).
        """
        counts = {
            zone_id: len(confirmed.get(zone_id, set())) for zone_id in self._zones
        }
        return self.process_counts(counts, ts)

    def process_counts(
        self, counts: dict[int, int], ts: float
    ) -> list[OccupancyUpdate]:
        """Emit occupancy updates from per-zone counts directly (no tracking).

        The snapshot-pull path counts detections in-zone (via
        :meth:`~app.localisation.zones.ZoneEvaluator.count_in_zones`) and feeds the
        counts here. Emission is change-only, exactly like :meth:`process`.

        Args:
            counts: ``zone_id -> people count`` for this frame. Zones absent from
                the mapping are treated as ``0``.
            ts: Frame timestamp stamped on emitted events.

        Returns:
            One :class:`OccupancyUpdate` per zone whose count changed.
        """
        updates: list[OccupancyUpdate] = []
        for zone_id, zone in self._zones.items():
            count = counts.get(zone_id, 0)
            if self._last_count.get(zone_id) == count:
                continue
            self._last_count[zone_id] = count
            updates.append(
                OccupancyUpdate(
                    ts=ts,
                    zone_id=zone_id,
                    camera_id=self._camera_id,
                    count=count,
                    dt_space_id=zone.dt_space_id,
                )
            )
        return updates
