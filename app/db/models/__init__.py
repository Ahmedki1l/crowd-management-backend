"""ORM models (HLD 9). Import order ensures all tables are registered on ``Base``."""

from app.db.base import Base
from app.db.models.alerts import Alert, Snapshot
from app.db.models.camera import Camera
from app.db.models.config import ConfigRow
from app.db.models.geometry import Line, Zone
from app.db.models.timeseries import (
    CrossingEvent,
    DwellSession,
    OccupancySample,
)

__all__ = [
    "Base",
    "Camera",
    "Zone",
    "Line",
    "OccupancySample",
    "CrossingEvent",
    "DwellSession",
    "Alert",
    "Snapshot",
    "ConfigRow",
]
