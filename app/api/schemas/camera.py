"""Camera API schemas.

Registry responses keep the password **write-only** (HLD 8.1 / 14): it is
accepted on create/update and never returned by any GET. The dedicated,
authenticated server-to-server credential-resolution POST uses the separate
``CameraCredentialsResolveOut`` response model. Its distinct name prevents it
from being confused with richer internal camera projections during merges.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import CameraRole


class CameraCreate(BaseModel):
    name: str
    area: str
    floor: str | None = None
    ip: str
    port: int = 554
    username: str
    password: str = Field(..., description="RTSP password; stored encrypted, never returned")
    roles: list[CameraRole] = Field(default_factory=lambda: [CameraRole.OCCUPANCY])
    imgsz: int | None = None
    stream_channel_sub: int = 102
    stream_channel_main: int = 101
    enabled: bool = True


class CameraUpdate(BaseModel):
    name: str | None = None
    area: str | None = None
    floor: str | None = None
    ip: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = Field(default=None, description="Set to rotate; never returned")
    roles: list[CameraRole] | None = None
    imgsz: int | None = None
    stream_channel_sub: int | None = None
    stream_channel_main: int | None = None
    enabled: bool | None = None


class CameraCredentialsByIp(BaseModel):
    """Server-to-server request for one camera's credentials by IP address."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ip: str = Field(min_length=1, max_length=64)


class CameraCredentialsResolveOut(BaseModel):
    """Sensitive response returned only by the credential-resolution POST."""

    model_config = ConfigDict(hide_input_in_errors=True)

    ip: str
    username: str
    password: str = Field(repr=False)


class CameraOut(BaseModel):
    """Response model — note: NO password field, by design."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    area: str
    floor: str | None = None
    ip: str
    port: int
    username: str
    roles: list[str]
    imgsz: int | None = None
    stream_channel_sub: int
    stream_channel_main: int
    enabled: bool
    has_password: bool = False
    updated_at: datetime | None = None


class CameraCredentialsOut(CameraOut):
    """Internal camera projection; never mount on a public-authenticated route."""

    model_config = ConfigDict(from_attributes=True, hide_input_in_errors=True)

    password: str = Field(repr=False)


class CameraTestResult(BaseModel):
    reachable: bool
    codec: str | None = None
    resolution: str | None = None
    fps: float | None = None
    latency_ms: float | None = None
    error: str | None = None
