#!/usr/bin/env python
"""Migrate data from SQLite to SQL Server.

Copies cameras, zones, lines, and config from a SQLite database into an already-
initialised SQL Server database.  Time-series tables (occupancy_samples,
crossing_events, dwell_sessions) and alerts/snapshots are skipped
by default; pass --include-timeseries / --include-alerts to copy them too.

Usage
-----
  # defaults: src=sqlite:///./camera_analytics.db, dst from DATABASE_URL in .env
  python scripts/migrate_sqlite_to_mssql.py

  # explicit URLs
  python scripts/migrate_sqlite_to_mssql.py \\
      --src sqlite:///./camera_analytics.db \\
      --dst "mssql+pyodbc://sa:Password123@db:1433/camera_analytics?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"

  # also copy historical data
  python scripts/migrate_sqlite_to_mssql.py --include-timeseries --include-alerts
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make sure the project root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import MetaData, Table, create_engine, inspect, text
from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# Migration order matters: parent tables before children (FK constraints).
# ---------------------------------------------------------------------------
_CORE_TABLES = [
    "cameras",
    "config",
    "zones",
    "lines",
]
_ALERT_TABLES = [
    "alerts",
    "snapshots",
]
_TIMESERIES_TABLES = [
    "occupancy_samples",
    "crossing_events",
    "dwell_sessions",
]


def _build_engine(url: str) -> Engine:
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    elif url.startswith("mssql"):
        kwargs["fast_executemany"] = True
    return create_engine(url, **kwargs)


def _reflect_table(engine: Engine, name: str) -> Table | None:
    """Reflect a single table from *engine*; return None if it doesn't exist."""
    meta = MetaData()
    insp = inspect(engine)
    if name not in insp.get_table_names():
        return None
    meta.reflect(bind=engine, only=[name])
    return meta.tables[name]


def _copy_table(
    src_engine: Engine,
    dst_engine: Engine,
    table_name: str,
    chunk_size: int = 500,
) -> int:
    """Copy all rows from *table_name* in src to dst.  Returns rows copied."""
    src_table = _reflect_table(src_engine, table_name)
    if src_table is None:
        print(f"  [skip] {table_name!r} — not found in source")
        return 0

    dst_table = _reflect_table(dst_engine, table_name)
    if dst_table is None:
        print(f"  [skip] {table_name!r} — not found in destination (run the app once to create tables)")
        return 0

    with src_engine.connect() as src_conn:
        rows = src_conn.execute(src_table.select()).fetchall()

    if not rows:
        print(f"  {table_name}: 0 rows (empty)")
        return 0

    columns = [c.name for c in src_table.columns]
    dicts = [dict(zip(columns, row)) for row in rows]

    # SQL Server IDENTITY columns reject explicit values unless this flag is set.
    has_identity = any(
        c.autoincrement is True or str(c.autoincrement) == "auto"
        for c in dst_table.primary_key.columns
    )
    is_mssql = dst_engine.dialect.name == "mssql"

    with dst_engine.begin() as dst_conn:
        if is_mssql and has_identity:
            dst_conn.execute(text(f"SET IDENTITY_INSERT [{table_name}] ON"))

        # Insert in chunks to avoid parameter-list limits.
        for i in range(0, len(dicts), chunk_size):
            chunk = dicts[i : i + chunk_size]
            dst_conn.execute(dst_table.insert(), chunk)

        if is_mssql and has_identity:
            dst_conn.execute(text(f"SET IDENTITY_INSERT [{table_name}] OFF"))

    print(f"  {table_name}: {len(rows)} row(s) copied")
    return len(rows)


def migrate(
    src_url: str,
    dst_url: str,
    include_timeseries: bool,
    include_alerts: bool,
) -> None:
    print(f"Source : {src_url}")
    print(f"Target : {dst_url}")
    print()

    src_engine = _build_engine(src_url)
    dst_engine = _build_engine(dst_url)

    tables = list(_CORE_TABLES)
    if include_alerts:
        tables += _ALERT_TABLES
    if include_timeseries:
        tables += _TIMESERIES_TABLES

    total = 0
    for table in tables:
        total += _copy_table(src_engine, dst_engine, table)

    print()
    print(f"Done — {total} total row(s) migrated.")


def main() -> int:
    default_dst = os.environ.get(
        "DATABASE_URL",
        "mssql+pyodbc://sa:Password123@db:1433/camera_analytics"
        "?driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes",
    )

    parser = argparse.ArgumentParser(description="Migrate SQLite → SQL Server")
    parser.add_argument(
        "--src",
        default="sqlite:///./camera_analytics.db",
        help="Source SQLite URL (default: sqlite:///./camera_analytics.db)",
    )
    parser.add_argument(
        "--dst",
        default=default_dst,
        help="Destination SQL Server URL (default: DATABASE_URL from .env)",
    )
    parser.add_argument(
        "--include-timeseries",
        action="store_true",
        help="Also copy occupancy_samples, crossing_events, dwell_sessions",
    )
    parser.add_argument(
        "--include-alerts",
        action="store_true",
        help="Also copy alerts and snapshots",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.src == args.dst:
        print("ERROR: source and destination are the same URL", file=sys.stderr)
        return 1
    if not args.src.startswith("sqlite"):
        print("WARNING: source does not look like a SQLite URL — proceed anyway? [y/N] ", end="")
        if input().strip().lower() != "y":
            return 1

    migrate(args.src, args.dst, args.include_timeseries, args.include_alerts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
