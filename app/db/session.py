"""Engine + session management (sync SQLAlchemy 2.0).

DB-bound API routes are plain ``def`` handlers so FastAPI runs them in a thread
pool; only the SSE endpoint is async. This keeps one consistent DB story across
SQL Server (prod) and SQLite (tests) without an async ODBC driver.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings
from app.db.base import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine(url: str, echo: bool = False) -> Engine:
    connect_args: dict = {}
    kwargs: dict = {"echo": echo, "future": True, "pool_pre_ping": True}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in url:
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
    return create_engine(url, connect_args=connect_args, **kwargs)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        cfg = get_settings().database
        _engine = _build_engine(cfg.url, cfg.echo)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _session_factory


def configure_engine(url: str, echo: bool = False) -> Engine:
    """Rebind the engine (tests / explicit reconfiguration)."""
    global _engine, _session_factory
    _engine = _build_engine(url, echo)
    _session_factory = sessionmaker(
        bind=_engine, autoflush=False, expire_on_commit=False, future=True
    )
    return _engine


def init_db() -> None:
    """Create all tables (dev/tests). Production uses Alembic migrations."""
    import app.db.models  # noqa: F401  (register all tables on Base.metadata)

    Base.metadata.create_all(bind=get_engine())


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yields a session, commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for worker / service code outside the request cycle."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
