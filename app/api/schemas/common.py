"""Shared API schema primitives."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# A point is [x, y] in image pixels.
PointArray = Annotated[list[float], Field(min_length=2, max_length=2)]


class Message(BaseModel):
    message: str


class HealthOut(BaseModel):
    status: str
    detail: dict | None = None
