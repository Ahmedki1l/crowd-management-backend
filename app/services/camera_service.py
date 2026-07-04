"""Camera registry service (HLD 6.5, 5.8).

Orchestrates camera CRUD on top of :class:`~app.db.repositories.camera_repo`
while owning the two concerns the repository deliberately does not: credential
encryption at rest (:class:`~app.services.credentials.CredentialCipher`) and
assembling the immutable :class:`~app.domain.models.CameraSpec` the engine runs.

Plaintext passwords exist only transiently inside this service. They are
encrypted before reaching the database, never written to a response model
(``CameraOut`` exposes only ``has_password``), and only decrypted by
:meth:`CameraService.resolve_password` for the engine that actually connects to
the stream.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy.orm import Session

from app.api.schemas.camera import (
    CameraCreate,
    CameraOut,
    CameraTestResult,
    CameraUpdate,
)
from app.db.models.camera import Camera
from app.db.repositories.camera_repo import CameraRepository
from app.db.repositories.line_repo import LineRepository
from app.db.repositories.mappers import to_camera_spec
from app.db.repositories.zone_repo import ZoneRepository
from app.domain.models import CameraSpec
from app.ingestion.stream_url import redact, sub_stream_url
from app.services.credentials import CredentialCipher
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Seconds to wait when probing a camera stream in test_camera before giving up.
_PROBE_TIMEOUT_S = 5.0


class CameraService:
    """Camera CRUD, credential handling and spec assembly for one DB session."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a ``session``; it owns no transaction itself."""
        self._session = session
        self._cameras = CameraRepository(session)
        self._zones = ZoneRepository(session)
        self._lines = LineRepository(session)

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def create_camera(self, payload: CameraCreate) -> Camera:
        """Create a camera, encrypting its password before it touches the DB.

        Args:
            payload: Validated create request including the plaintext password.

        Returns:
            The persisted :class:`Camera` row (still attached to the session).
        """
        cipher = CredentialCipher.from_env()
        data: dict[str, Any] = {
            "name": payload.name,
            "area": payload.area,
            "floor": payload.floor,
            "ip": payload.ip,
            "port": payload.port,
            "username": payload.username,
            "roles": [role.value for role in payload.roles],
            "imgsz": payload.imgsz,
            "stream_channel_sub": payload.stream_channel_sub,
            "stream_channel_main": payload.stream_channel_main,
            "enabled": payload.enabled,
        }
        camera = self._cameras.create(data)
        # Set the ciphertext separately: the repository never touches the
        # encrypted column (HLD 5.8) — the service owns it.
        camera.password_encrypted = cipher.encrypt(payload.password)
        self._session.flush()
        logger.info(
            "camera created",
            extra={"camera_id": camera.id, "event": "camera_created"},
        )
        return camera

    def update_camera(self, id: int, payload: CameraUpdate) -> Camera | None:
        """Apply a partial update, re-encrypting the password only if provided.

        Args:
            id: Camera id to update.
            payload: Partial update; unset fields are left untouched. A non-null
                ``password`` rotates the stored credential.

        Returns:
            The updated :class:`Camera`, or ``None`` if no camera has ``id``.
        """
        data = self._update_columns(payload)
        camera = self._cameras.update(id, data) if data else self._cameras.get(id)
        if camera is None:
            return None
        if payload.password is not None:
            cipher = CredentialCipher.from_env()
            camera.password_encrypted = cipher.encrypt(payload.password)
            self._session.flush()
        logger.info(
            "camera updated",
            extra={"camera_id": camera.id, "event": "camera_updated"},
        )
        return camera

    def delete_camera(self, id: int) -> bool:
        """Delete a camera (cascading to its zones/lines). Return whether it existed."""
        removed = self._cameras.delete(id)
        if removed:
            logger.info(
                "camera deleted", extra={"camera_id": id, "event": "camera_deleted"}
            )
        return removed

    def get(self, id: int) -> Camera | None:
        """Return the camera with ``id`` or ``None``."""
        return self._cameras.get(id)

    def list(self) -> list[Camera]:
        """Return all cameras ordered by id."""
        return self._cameras.list()

    # ------------------------------------------------------------------ #
    # Presentation
    # ------------------------------------------------------------------ #
    def to_out(self, camera: Camera) -> CameraOut:
        """Map a camera row to its response model, never exposing the password.

        ``has_password`` reflects whether a credential is stored; the ciphertext
        and plaintext are both withheld by design (HLD 8.1 / 14).
        """
        return CameraOut(
            id=camera.id,
            name=camera.name,
            area=camera.area,
            floor=camera.floor,
            ip=camera.ip,
            port=camera.port,
            username=camera.username,
            roles=list(camera.roles),
            imgsz=camera.imgsz,
            stream_channel_sub=camera.stream_channel_sub,
            stream_channel_main=camera.stream_channel_main,
            enabled=camera.enabled,
            has_password=camera.password_encrypted is not None,
            updated_at=camera.updated_at,
        )

    # ------------------------------------------------------------------ #
    # Engine-facing
    # ------------------------------------------------------------------ #
    def build_camera_spec(self, camera_id: int) -> CameraSpec | None:
        """Assemble the immutable :class:`CameraSpec` the engine runs one camera with.

        Loads the camera with its zones and lines and maps them to domain value
        objects. Credentials are intentionally excluded — the engine resolves the
        password separately via :meth:`resolve_password`.

        Args:
            camera_id: Camera id to build a spec for.

        Returns:
            The :class:`CameraSpec`, or ``None`` if no camera has ``camera_id``.
        """
        camera = self._cameras.get(camera_id)
        if camera is None:
            return None
        zones = self._zones.list_by_camera(camera_id)
        lines = self._lines.list_by_camera(camera_id)
        return to_camera_spec(camera, zones, lines)

    def resolve_password(self, camera_id: int) -> str:
        """Decrypt and return a camera's plaintext password for the engine.

        Args:
            camera_id: Camera id whose credential to resolve.

        Returns:
            The decrypted plaintext password.

        Raises:
            KeyError: If no camera has ``camera_id``.
            ValueError: If the camera has no stored credential, or the stored
                ciphertext is invalid/tampered (surfaced by the cipher).
        """
        camera = self._cameras.get(camera_id)
        if camera is None:
            raise KeyError(f"camera {camera_id} not found")
        if camera.password_encrypted is None:
            raise ValueError(f"camera {camera_id} has no stored credential")
        cipher = CredentialCipher.from_env()
        return cipher.decrypt(camera.password_encrypted)

    # ------------------------------------------------------------------ #
    # Connectivity probe
    # ------------------------------------------------------------------ #
    def test_camera(self, camera_id: int) -> CameraTestResult:
        """Best-effort connectivity probe against a camera's sub stream.

        Opens the sub-stream RTSP URL with OpenCV, grabs a single frame and
        reports the observed codec, resolution, fps and round-trip latency. Any
        failure (missing camera, bad credential, unreachable stream, OpenCV
        error) is reported as ``reachable=False`` with an ``error`` message — the
        probe never raises so callers can surface the result directly.

        Args:
            camera_id: Camera id to probe.

        Returns:
            A :class:`CameraTestResult` describing reachability and stream
            properties, or the failure reason.
        """
        spec = self.build_camera_spec(camera_id)
        if spec is None:
            return CameraTestResult(reachable=False, error=f"camera {camera_id} not found")
        try:
            password = self.resolve_password(camera_id)
        except (KeyError, ValueError) as exc:
            return CameraTestResult(reachable=False, error=str(exc))

        url = sub_stream_url(spec, password)
        return self._probe_stream(camera_id, url)

    def _probe_stream(self, camera_id: int, url: str) -> CameraTestResult:
        """Open ``url`` with OpenCV and grab one frame, reporting stream metadata.

        ``cv2`` is imported lazily so this service imports cleanly with only core
        dependencies installed (HLD constraint). The capture handle is always
        released, even on failure.
        """
        import cv2  # lazy: heavy/optional dependency (HLD constraint)

        redacted = redact(url)
        started = time.monotonic()
        capture: Any = None
        try:
            capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                logger.warning(
                    "camera probe could not open stream %s",
                    redacted,
                    extra={"camera_id": camera_id, "event": "camera_probe_unreachable"},
                )
                return CameraTestResult(reachable=False, error="stream could not be opened")

            ok, frame = capture.read()
            if not ok or frame is None:
                return CameraTestResult(
                    reachable=False, error="stream opened but no frame could be read"
                )

            latency_ms = (time.monotonic() - started) * 1000.0
            return CameraTestResult(
                reachable=True,
                codec=self._read_codec(capture, cv2),
                resolution=self._read_resolution(frame),
                fps=self._read_fps(capture, cv2),
                latency_ms=round(latency_ms, 1),
            )
        except cv2.error as exc:  # type: ignore[attr-defined]
            # OpenCV/FFmpeg surfaces backend faults as cv2.error; report the
            # reason rather than crashing the probe endpoint.
            logger.warning(
                "camera probe raised: %s",
                exc,
                extra={"camera_id": camera_id, "event": "camera_probe_error"},
            )
            return CameraTestResult(reachable=False, error=f"opencv error: {exc}")
        finally:
            if capture is not None:
                capture.release()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _update_columns(payload: CameraUpdate) -> dict[str, Any]:
        """Build the column map for an update from the set, non-password fields.

        The password is handled separately (it must be encrypted), so it is
        excluded here. Only fields the caller actually provided are included so
        the update stays partial.
        """
        data = payload.model_dump(exclude_unset=True, exclude={"password"})
        if "roles" in data and data["roles"] is not None:
            data["roles"] = [role.value for role in payload.roles]
        return data

    @staticmethod
    def _read_codec(capture: Any, cv2: Any) -> str | None:
        """Decode the FourCC codec tag from an open capture, or ``None``."""
        fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))
        if fourcc_int <= 0:
            return None
        chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
        codec = "".join(c for c in chars if c.isprintable()).strip()
        return codec or None

    @staticmethod
    def _read_resolution(frame: Any) -> str | None:
        """Format a decoded frame's resolution as ``"<width>x<height>"``."""
        if getattr(frame, "ndim", 0) < 2:
            return None
        height, width = frame.shape[0], frame.shape[1]
        return f"{int(width)}x{int(height)}"

    @staticmethod
    def _read_fps(capture: Any, cv2: Any) -> float | None:
        """Read the advertised stream fps, or ``None`` if unavailable."""
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return round(fps, 2) if fps > 0 else None
