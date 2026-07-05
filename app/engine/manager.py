"""Process-wide engine handle so the API can switch what the engine runs at runtime.

The :class:`EngineManager` owns the single live :class:`~app.engine.engine.Engine`
and lets callers swap the set of cameras it runs — everything, only the
``entry_exit`` cameras, or only the ``occupancy`` cameras — without restarting the
process. Both the ``--api`` entry point (:mod:`app.main`) and the engine-control
router (:mod:`app.api.routers.engine`) share the one instance via
:func:`get_engine_manager`, so a button pressed over HTTP controls the same engine
the process started.

A single lock serialises switches: each switch stops the current engine and builds
a fresh one for the requested scope, so two concurrent presses can't interleave. A
switch reloads the perception backends, so it takes a few seconds — this is an ops
control, not a hot path.

``build_engine`` (and the heavy perception backends it pulls) is imported lazily
inside the methods, so importing this module — and therefore the router and the
FastAPI app — stays on the lightweight core dependency set.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from app.config.settings import get_settings, reset_settings_cache
from app.db.session import session_scope
from app.domain.models import CameraRole
from app.services.camera_service import CameraService
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.engine import Engine

logger = get_logger(__name__)

# Public scope names accepted by :meth:`EngineManager.switch`, mapped to the role
# each one filters cameras by (``None`` == every enabled/allowlisted camera).
_SCOPE_ROLES: dict[str, CameraRole | None] = {
    "all": None,
    "entry_exit": CameraRole.ENTRY_EXIT,
    "occupancy": CameraRole.OCCUPANCY,
}


class EngineManager:
    """Owns the live engine and swaps the camera scope it runs on demand."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._mode: str = "stopped"
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """Return the current scope, whether it's running, and pipeline count."""
        with self._lock:
            return {
                "mode": self._mode,
                "running": self._engine is not None,
                "pipeline_count": self._engine.pipeline_count if self._engine else 0,
            }

    # ------------------------------------------------------------------ #
    # Controls
    # ------------------------------------------------------------------ #
    def start(self, mode: str = "all") -> dict[str, Any]:
        """Boot the engine on ``mode`` (used by the ``--api`` entry point)."""
        return self.switch(mode)

    def switch(self, mode: str) -> dict[str, Any]:
        """Stop the current engine and run only the cameras for ``mode``.

        Args:
            mode: One of ``"all"``, ``"entry_exit"``, ``"occupancy"``.

        Returns:
            The new :meth:`status`.

        Raises:
            ValueError: If ``mode`` is not a known scope.
        """
        if mode not in _SCOPE_ROLES:
            raise ValueError(f"unknown engine mode {mode!r}; expected one of {sorted(_SCOPE_ROLES)}")
        camera_ids = camera_ids_for_role(_SCOPE_ROLES[mode])
        with self._lock:
            self._apply(mode, camera_ids)
        return self.status()

    def reset(self) -> dict[str, Any]:
        """Reload config (``.env`` + YAML) and restart the engine on every camera.

        Use after editing configuration or the camera registry to apply the
        changes without restarting the process. Always returns to the ``"all"``
        scope.
        """
        from dotenv import load_dotenv

        load_dotenv(override=True)  # re-read .env into the environment
        reset_settings_cache()  # drop cached AppConfig + Secrets
        camera_ids = camera_ids_for_role(None)
        with self._lock:
            self._apply("all", camera_ids)
        logger.info("engine reset (config reloaded)", extra={"event": "engine_reset"})
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop every pipeline and leave the engine idle (the ``"stopped"`` scope)."""
        with self._lock:
            if self._engine is not None:
                self._engine.stop()
                self._engine = None
            self._mode = "stopped"
        return self.status()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _apply(self, mode: str, camera_ids: list[int] | None) -> None:
        """Replace the running engine with a fresh one for ``camera_ids`` (holds lock).

        ``camera_ids=None`` means "every enabled camera" — :func:`build_engine`
        then applies the IP allowlist itself. An empty list means "no cameras".
        """
        from app.engine.engine import build_engine

        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        engine = build_engine(camera_ids=camera_ids)
        engine.start()
        self._engine = engine
        self._mode = mode
        logger.info(
            "engine scope switched",
            extra={"event": "engine_switched", "mode": mode, "cameras": engine.pipeline_count},
        )


def camera_ids_for_role(role: CameraRole | None) -> list[int] | None:
    """Resolve the enabled, allowlisted camera ids that carry ``role``.

    Returns ``None`` for ``role is None`` so the "all" path defers to
    :func:`build_engine`'s own allowlist handling. Otherwise returns the explicit
    id list (possibly empty) of enabled cameras carrying ``role``, narrowed to the
    IP allowlist when one is configured.
    """
    if role is None:
        return None
    allowlist = set(get_settings().processing.camera_allowlist_ips)
    with session_scope() as session:
        cameras = [c for c in CameraService(session).list() if c.enabled]
        if allowlist:
            cameras = [c for c in cameras if c.ip in allowlist]
        return [c.id for c in cameras if role.value in list(c.roles)]


_manager: EngineManager | None = None


def get_engine_manager() -> EngineManager:
    """Return the process-wide :class:`EngineManager` singleton."""
    global _manager
    if _manager is None:
        _manager = EngineManager()
    return _manager


def reset_engine_manager() -> None:
    """Stop any running engine and drop the singleton (used by tests)."""
    global _manager
    if _manager is not None:
        _manager.stop()
    _manager = None
