"""Safety calculator: intrusion and overcrowding alerts (HLD 7).

Watches the debounced per-zone confirmed membership and counts and raises
:class:`app.events.events.AlertRaised` events:

* **Intrusion** — a :class:`~app.domain.models.ZoneType.RESTRICTED` zone has at
  least one confirmed member for ``debounce_frames`` consecutive frames.
* **Overcrowding** — a zone with a configured ``safe_limit`` has a count strictly
  above that limit for ``debounce_frames`` consecutive frames.

A per-``(zone, alert_type)`` cooldown of ``cooldown_seconds`` suppresses repeat
alerts so a sustained condition does not produce an alert storm. The calculator
is pure: it never persists anything and never fetches a snapshot. Emitted alerts
carry ``alert_id=None`` and ``snapshot_url=None``; the engine enriches them with
a captured snapshot and the persisted id.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.analytics.base import BaseCalculator
from app.domain.interfaces import Clock
from app.domain.models import AlertType, ZoneSpec, ZoneType
from app.events.events import AlertRaised
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class _ConditionState:
    """Per-(zone, alert_type) debounce and cooldown bookkeeping."""

    streak: int = 0
    last_fired_ts: float | None = None


class SafetyCalculator(BaseCalculator):
    """Raises intrusion and overcrowding alerts for one camera's zones.

    Holds per-zone debounce streak counters and last-fired timestamps. A
    condition must hold for ``debounce_frames`` consecutive frames before it
    fires, and once fired it cannot fire again for the same zone/type until
    ``cooldown_seconds`` have elapsed.
    """

    def __init__(
        self,
        camera_id: int,
        zones: list[ZoneSpec],
        clock: Clock,
        debounce_frames: int,
        cooldown_seconds: float,
    ) -> None:
        """Initialise the calculator.

        Args:
            camera_id: Camera these zones belong to.
            zones: Zone specifications for the camera. Restricted zones drive
                intrusion alerts; zones with a ``safe_limit`` drive overcrowding.
            clock: Shared pipeline time source.
            debounce_frames: Consecutive condition frames required before an alert
                fires (clamped to a minimum of 1).
            cooldown_seconds: Minimum seconds between two alerts of the same type
                for the same zone.
        """
        super().__init__(clock)
        self._camera_id = camera_id
        self._zones: dict[int, ZoneSpec] = {zone.id: zone for zone in zones}
        self._debounce_frames = max(1, debounce_frames)
        self._cooldown_seconds = cooldown_seconds
        # (zone_id, alert_type) -> condition state.
        self._states: dict[tuple[int, AlertType], _ConditionState] = {}

    def process(
        self,
        confirmed: dict[int, set[int]],
        counts: dict[int, int],
        ts: float,
    ) -> list[AlertRaised]:
        """Evaluate alert conditions for the current frame.

        Args:
            confirmed: ``zone_id -> set of confirmed-present track_ids`` from the
                presence state machine.
            counts: ``zone_id -> confirmed count`` for the frame. Used for the
                overcrowding comparison.
            ts: Frame timestamp (float epoch seconds), stamped on emitted alerts.

        Returns:
            The alerts that fired on this frame (possibly empty).
        """
        alerts: list[AlertRaised] = []
        for zone_id, zone in self._zones.items():
            if zone.type is ZoneType.RESTRICTED:
                members = confirmed.get(zone_id)
                active = members is not None and len(members) >= 1
                detail = self._intrusion_detail(zone, members)
                alert = self._evaluate(
                    zone, AlertType.INTRUSION, active, detail, ts
                )
                if alert is not None:
                    alerts.append(alert)

            if zone.safe_limit is not None:
                count = counts.get(zone_id, 0)
                active = count > zone.safe_limit
                detail = self._overcrowding_detail(zone, count)
                alert = self._evaluate(
                    zone, AlertType.OVERCROWDING, active, detail, ts
                )
                if alert is not None:
                    alerts.append(alert)
        return alerts

    def _evaluate(
        self,
        zone: ZoneSpec,
        alert_type: AlertType,
        active: bool,
        detail: str,
        ts: float,
    ) -> AlertRaised | None:
        """Advance one condition's debounce/cooldown and emit if it fires.

        Returns the alert when the condition has just satisfied the debounce
        window and is outside its cooldown, else ``None``. The streak is reset
        when the condition is not active so it must rebuild from scratch.
        """
        state = self._states.setdefault((zone.id, alert_type), _ConditionState())
        if not active:
            state.streak = 0
            return None

        state.streak += 1
        if state.streak < self._debounce_frames:
            return None

        if state.last_fired_ts is not None and (
            ts - state.last_fired_ts < self._cooldown_seconds
        ):
            return None

        state.last_fired_ts = ts
        logger.info(
            "safety alert fired",
            extra={
                "camera_id": self._camera_id,
                "zone_id": zone.id,
                "event": alert_type.value,
            },
        )
        return AlertRaised(
            ts=ts,
            alert_type=alert_type,
            zone_id=zone.id,
            camera_id=self._camera_id,
            detail=detail,
            snapshot_url=None,
            dt_space_id=zone.dt_space_id,
            alert_id=None,
        )

    @staticmethod
    def _intrusion_detail(zone: ZoneSpec, members: set[int] | None) -> str:
        """Human-readable description for an intrusion alert."""
        count = len(members) if members is not None else 0
        return f"Intrusion in restricted zone '{zone.name}': {count} person(s) present"

    @staticmethod
    def _overcrowding_detail(zone: ZoneSpec, count: int) -> str:
        """Human-readable description for an overcrowding alert."""
        return (
            f"Overcrowding in zone '{zone.name}': {count} present "
            f"exceeds safe limit of {zone.safe_limit}"
        )
