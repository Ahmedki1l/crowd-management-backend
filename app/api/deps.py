"""Shared FastAPI dependencies: auth, DB session, event bus, state store."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config.schema import AppConfig
from app.config.settings import get_secrets, get_settings
from app.db.session import get_session
from app.events.event_bus import InMemoryEventBus, get_event_bus
from app.services.state_store import StateStore, get_state_store
from app.utils.security import JWTError, decode_jwt


def db_session() -> Iterator[Session]:
    yield from get_session()


def config() -> AppConfig:
    return get_settings()


def event_bus() -> InMemoryEventBus:
    return get_event_bus()


def state_store() -> StateStore:
    return get_state_store()


def require_auth(authorization: str | None = Header(default=None)) -> dict:
    """Authenticate via Bearer token (HLD 8). Scheme is ``jwt`` or ``api_key``."""
    settings = get_settings()
    secrets = get_secrets()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    if settings.api.auth_scheme == "api_key":
        # constant-time compare against the configured key
        import hmac

        if not hmac.compare_digest(token, secrets.api_auth_secret):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")
        return {"sub": "api-key"}
    try:
        return decode_jwt(token, secrets.api_auth_secret)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc


AuthDep = Depends(require_auth)
