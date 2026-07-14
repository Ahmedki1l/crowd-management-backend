"""Zone and counting-line API schemas (HLD 8.1)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.schemas.common import PointArray
from app.domain.models import ZoneType


class ZoneCreate(BaseModel):
    camera_id: int
    name: str
    type: ZoneType
    polygon: list[PointArray] = Field(..., description="[[x,y], ...] >= 3 points")
    safe_limit: int | None = None
    # Required, and non-empty. Occupancy history is keyed by space, so a zone with no
    # dt_space_id contributes to nothing and is invisible to every history query — it is
    # counted live and then silently forgotten. That is not a hypothetical: 31 of the 44
    # zones in this deployment were created without one, and none of them has a single
    # row of history. The drawing tool only warned about it in JavaScript, which is a
    # dialog to click through, not a constraint.
    dt_space_id: str = Field(
        ..., min_length=1, description="Logical space this zone feeds, e.g. b1-waiting-area"
    )

    @field_validator("polygon")
    @classmethod
    def _at_least_triangle(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError("polygon needs at least 3 points")
        return v


class ZoneUpdate(BaseModel):
    name: str | None = None
    type: ZoneType | None = None
    polygon: list[PointArray] | None = None
    safe_limit: int | None = None
    # ``None`` means "leave it alone" (the PATCH convention), not "clear it" — there is
    # deliberately no way to strip a zone's space and orphan it from history.
    dt_space_id: str | None = Field(default=None, min_length=1)


class ZoneOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: int
    name: str
    type: str
    polygon: list[list[float]]
    safe_limit: int | None = None
    dt_space_id: str | None = None


class LineCreate(BaseModel):
    camera_id: int
    name: str
    points: list[PointArray] = Field(..., min_length=2, max_length=2)
    in_direction: PointArray = Field(..., description="[dx,dy] normal that counts as IN")
    area_id: str
    dt_space_id: str | None = None


class LineUpdate(BaseModel):
    name: str | None = None
    points: list[PointArray] | None = Field(default=None, min_length=2, max_length=2)
    in_direction: PointArray | None = None
    area_id: str | None = None
    dt_space_id: str | None = None


class LineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    camera_id: int
    name: str
    points: list[list[float]]
    in_direction: list[float]
    area_id: str
    dt_space_id: str | None = None
