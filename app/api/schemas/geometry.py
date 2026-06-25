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
    dt_space_id: str | None = None

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
    dt_space_id: str | None = None


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
