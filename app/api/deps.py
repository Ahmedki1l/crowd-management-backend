"""Shared FastAPI dependencies: auth, DB session, event bus, state store."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config.schema import AppConfig
from app.config.settings import get_secrets, get_settings
from app.db.session import get_session
from app.events.event_bus import InMemoryEventBus, get_event_bus
from app.services.state_store import StateStore, get_state_store
from app.utils.security import JWTError, decode_jwt

# Registering the scheme (rather than reading a raw header) is what makes the
# Swagger UI render a global "Authorize" button and attach the Bearer token to
# every request. auto_error=False so we keep our own 401 message/handling.
_bearer_scheme = HTTPBearer(auto_error=False, description="JWT or API key")


def db_session() -> Iterator[Session]:
    yield from get_session()


def config() -> AppConfig:
    return get_settings()


def event_bus() -> InMemoryEventBus:
    return get_event_bus()


def state_store() -> StateStore:
    return get_state_store()


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """Authenticate via Bearer token (HLD 8). Scheme is ``jwt`` or ``api_key``.

    The token is supplied as ``Authorization: Bearer <token>`` — via the Swagger
    "Authorize" button or any HTTP client. Missing/blank credentials yield 401.
    """
    settings = get_settings()
    secrets = get_secrets()
    if credentials is None or not credentials.credentials.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials.strip()
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
