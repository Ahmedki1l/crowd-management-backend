"""Runtime configuration service (HLD 6.5, 8.1).

The effective runtime config is the YAML/env baseline
(:func:`~app.config.settings.get_settings`) overlaid with the per-section
overrides stored in the ``config`` table
(:class:`~app.db.repositories.config_repo.ConfigRepository`). This service is the
single place those two sources are merged.

Overrides are applied at *read* time: :meth:`ConfigService.get_runtime_config`
recomputes the merge on every call. Workers therefore pick up a change only on
their next rebuild (when they re-read the config) — updating a section does not
hot-reload running workers. Each tunable section is stored under its own key so
sections can be overridden independently without clobbering one another.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.schemas.config import RuntimeConfigOut, RuntimeConfigUpdate
from app.config.schema import (
    DetectorConfig,
    ProcessingConfig,
    StateMachineConfig,
    TrackerConfig,
)
from app.config.settings import get_settings
from app.db.repositories.config_repo import ConfigRepository
from app.utils.logging import get_logger

logger = get_logger(__name__)

# The runtime-tunable sections and the pydantic model each one validates against.
# This is the single mapping from a config-table key to its schema; both reading
# (overlay) and writing (validate-then-upsert) iterate it, so a new tunable
# section is added in exactly one place.
_SECTION_MODELS: dict[str, type[BaseModel]] = {
    "processing": ProcessingConfig,
    "detector": DetectorConfig,
    "tracker": TrackerConfig,
    "state_machine": StateMachineConfig,
}


class ConfigService:
    """Read and update the effective runtime config for one DB session."""

    def __init__(self, session: Session) -> None:
        """Bind the service to a ``session``; it owns no transaction itself."""
        self._config = ConfigRepository(session)

    def get_runtime_config(self) -> RuntimeConfigOut:
        """Return the baseline config overlaid with stored per-section overrides.

        Starts from the YAML/env settings and, for each tunable section, replaces
        it with the validated override row when one exists. A malformed override
        row is logged and ignored so a bad stored value can never break reads.

        Returns:
            The merged :class:`RuntimeConfigOut`.
        """
        overrides = self._config.get_all()
        merged = self._merge(overrides)
        return RuntimeConfigOut(**merged)

    def update_runtime_config(self, payload: RuntimeConfigUpdate) -> RuntimeConfigOut:
        """Persist the provided sections and return the recomputed merged config.

        Only sections present (non-``None``) in ``payload`` are written; each is
        stored under its own key as its ``model_dump`` so it overlays the baseline
        on the next read. Sections left unset keep whatever override (or baseline)
        they already had.

        Args:
            payload: Partial update carrying the sections to override.

        Returns:
            The merged :class:`RuntimeConfigOut` after the upserts.
        """
        provided = payload.model_dump(exclude_unset=True, exclude_none=True)
        for key, value in provided.items():
            self._config.upsert(key, value)
            logger.info(
                "runtime config section updated",
                extra={"event": "config_updated", "section": key},
            )
        return self.get_runtime_config()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _merge(self, overrides: dict[str, dict[str, Any]]) -> dict[str, BaseModel]:
        """Overlay stored override rows onto the baseline settings, per section."""
        settings = get_settings()
        merged: dict[str, BaseModel] = {}
        for key, model in _SECTION_MODELS.items():
            baseline: BaseModel = getattr(settings, key)
            merged[key] = self._apply_override(key, model, baseline, overrides.get(key))
        return merged

    @staticmethod
    def _apply_override(
        key: str,
        model: type[BaseModel],
        baseline: BaseModel,
        override: dict[str, Any] | None,
    ) -> BaseModel:
        """Validate and apply one section's override, or fall back to ``baseline``.

        The override is merged *onto* the baseline (partial overrides keep
        unspecified fields), then re-validated through the section model. A row
        that fails validation is logged and skipped rather than propagated, so a
        corrupt stored override degrades to the baseline instead of failing reads.
        """
        if not override:
            return baseline
        try:
            return model(**{**baseline.model_dump(), **override})
        except (ValueError, TypeError) as exc:
            logger.warning(
                "ignoring invalid config override for section %s: %s",
                key,
                exc,
                extra={"event": "config_override_invalid", "section": key},
            )
            return baseline
