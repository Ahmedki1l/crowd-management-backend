"""Runtime configuration router (HLD 8.1).

Exposes the effective runtime config (YAML/env baseline overlaid with stored
per-section overrides) and lets an operator update individual tunable sections.
All merge/validation/persistence logic lives in
:class:`~app.services.config_service.ConfigService`; this router only wires HTTP
to it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import AuthDep, db_session
from app.api.schemas.config import RuntimeConfigOut, RuntimeConfigUpdate
from app.services.config_service import ConfigService

router = APIRouter(prefix="/config", tags=["config"], dependencies=[AuthDep])


def _get_service(session: Session = Depends(db_session)) -> ConfigService:
    """Provide a :class:`ConfigService` bound to the request-scoped session."""
    return ConfigService(session)


@router.get("", response_model=RuntimeConfigOut)
def get_config(service: ConfigService = Depends(_get_service)) -> RuntimeConfigOut:
    """Return the baseline config overlaid with any stored section overrides."""
    return service.get_runtime_config()


@router.put("", response_model=RuntimeConfigOut)
def update_config(
    payload: RuntimeConfigUpdate, service: ConfigService = Depends(_get_service)
) -> RuntimeConfigOut:
    """Persist the provided sections and return the recomputed merged config.

    Only sections present in ``payload`` are written; unset sections keep their
    existing override or baseline. Workers pick up changes on their next rebuild.
    """
    return service.update_runtime_config(payload)
