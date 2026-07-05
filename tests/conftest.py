"""Shared test foundation: deterministic env, in-memory DB, auth, and factories.

Every test runs against a fresh in-memory SQLite database with cleared settings,
state-store, and event-bus singletons, so tests are fully isolated and need no
real cameras, models, or wall-clock time. Heavy inference deps are never
imported here — fakes (``app.inference.fakes``) and ``RecordedClipSource`` cover
the perception layer.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.app import create_app
from app.config.settings import reset_settings_cache
from app.db.models import Camera, Line, Zone
from app.db.session import (
    configure_engine,
    get_engine,
    get_session_factory,
    init_db,
)
from app.engine.manager import reset_engine_manager
from app.events.event_bus import reset_event_bus
from app.services.state_store import reset_state_store
from app.utils.clock import FakeClock
from app.utils.security import encode_jwt

_TEST_DB_URL = "sqlite:///:memory:"
_TEST_AUTH_SECRET = "test-secret"
_CONFIG_PATH = "config/config.example.yaml"


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Per-test isolation: deterministic env, fresh singletons, in-memory DB.

    Sets the secrets/config env the app reads, clears the cached settings,
    state-store and event-bus, then binds a fresh in-memory engine and creates
    the schema. Disposes the engine on teardown so no state leaks between tests.
    """
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)
    monkeypatch.setenv("API_AUTH_SECRET", _TEST_AUTH_SECRET)
    monkeypatch.setenv("CAMERA_CREDENTIALS_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("CONFIG_PATH", _CONFIG_PATH)

    reset_settings_cache()
    reset_state_store()
    reset_event_bus()

    configure_engine(_TEST_DB_URL)
    init_db()

    yield

    reset_engine_manager()  # stop any engine a test's /engine/* call started
    get_engine().dispose()
    reset_settings_cache()
    reset_state_store()
    reset_event_bus()


@pytest.fixture
def fake_clock() -> FakeClock:
    """Deterministic clock starting at t=1000.0 for dwell/debounce/cooldown tests."""
    return FakeClock(start=1000.0)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Authorization header carrying a valid JWT for subject ``test``."""
    token = encode_jwt({"sub": "test"}, _TEST_AUTH_SECRET, 3600)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A ``TestClient`` with the app lifespan run (projectors + bus wired).

    Created after ``_isolated_environment`` so the in-memory engine and projectors
    share the same database. Routes requiring auth need the ``auth_headers``
    fixture, e.g. ``client.get(url, headers=auth_headers)``.
    """
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def session() -> Iterator[Session]:
    """A SQLAlchemy session bound to the per-test in-memory engine."""
    sess = get_session_factory()()
    try:
        yield sess
    finally:
        sess.close()


@pytest.fixture
def make_camera():
    """Factory inserting a :class:`Camera` row and returning the committed ORM object."""

    def _make(session: Session, **over) -> Camera:
        defaults: dict = {
            "name": "cam-1",
            "area": "lobby",
            "ip": "10.0.0.1",
            "port": 554,
            "username": "admin",
            "roles": ["occupancy"],
            "stream_channel_sub": 102,
            "stream_channel_main": 101,
            "enabled": True,
        }
        defaults.update(over)
        camera = Camera(**defaults)
        session.add(camera)
        session.commit()
        session.refresh(camera)
        return camera

    return _make


@pytest.fixture
def make_zone():
    """Factory inserting a :class:`Zone` row and returning the committed ORM object."""

    def _make(
        session: Session,
        camera_id: int,
        type: str = "occupancy",
        polygon: list[list[float]] | None = None,
        **over,
    ) -> Zone:
        defaults: dict = {
            "camera_id": camera_id,
            "name": "zone-1",
            "type": type,
            "polygon": polygon
            if polygon is not None
            else [[400.0, 300.0], [800.0, 300.0], [800.0, 600.0], [400.0, 600.0]],
            "safe_limit": None,
            "dt_space_id": None,
        }
        defaults.update(over)
        zone = Zone(**defaults)
        session.add(zone)
        session.commit()
        session.refresh(zone)
        return zone

    return _make


@pytest.fixture
def make_line():
    """Factory inserting a :class:`Line` row and returning the committed ORM object."""

    def _make(session: Session, camera_id: int, **over) -> Line:
        defaults: dict = {
            "camera_id": camera_id,
            "name": "line-1",
            "points": [[640.0, 200.0], [640.0, 600.0]],
            "in_direction": [1.0, 0.0],
            "area_id": "area-1",
            "dt_space_id": None,
        }
        defaults.update(over)
        line = Line(**defaults)
        session.add(line)
        session.commit()
        session.refresh(line)
        return line

    return _make
