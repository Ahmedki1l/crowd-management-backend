"""Runtime config-override data access (HLD 6.5, 9).

Key/value JSON rows that ``app.services.config_service`` merges over the YAML
defaults. Values are arbitrary JSON objects keyed by string. ``upsert`` performs
a dialect-agnostic get-then-insert/update (no ``ON CONFLICT``/``MERGE``) so the
same code runs on SQL Server and SQLite. Repositories flush but never commit.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.config import ConfigRow


class ConfigRepository:
    """Access to the ``config`` key/value table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Return every override row as a ``{key: value}`` mapping."""
        rows = self._session.scalars(select(ConfigRow)).all()
        return {row.key: row.value for row in rows}

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the override value for ``key`` or ``None`` if unset."""
        row = self._session.get(ConfigRow, key)
        return row.value if row is not None else None

    def upsert(self, key: str, value: dict[str, Any]) -> None:
        """Insert ``key`` or overwrite its value if it already exists."""
        row = self._session.get(ConfigRow, key)
        if row is None:
            self._session.add(ConfigRow(key=key, value=value))
        else:
            row.value = value
        self._session.flush()
