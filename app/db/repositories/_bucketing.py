"""Dialect-agnostic time-bucket aggregation helpers (HLD 6.5).

Time-series queries bucket rows in Python rather than via per-dialect SQL date
functions, so the same code runs against SQL Server (prod) and SQLite (tests).
The repositories fetch the raw rows ordered by timestamp and feed them through
these helpers. Keeping the grouping logic here removes copy-paste drift between
the occupancy, crossing, and dwell repositories.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from typing import TypeVar

from app.utils.timeutil import floor_bucket

_Row = TypeVar("_Row")


def group_by_bucket(
    rows: Iterable[_Row],
    ts_of: Callable[[_Row], datetime],
    bucket_seconds: int,
) -> Iterator[tuple[datetime, list[_Row]]]:
    """Group ``rows`` into ``bucket_seconds``-wide buckets keyed by bucket start.

    Rows must be supplied already sorted ascending by their timestamp (the
    repositories order the query by ``ts``). Empty buckets in the range are not
    emitted — only buckets that contain at least one row.

    :param rows: Rows ordered ascending by timestamp.
    :param ts_of: Extracts the aware ``datetime`` to bucket a row by.
    :param bucket_seconds: Bucket width in seconds; must be positive.
    :yields: ``(bucket_start, rows_in_bucket)`` pairs in ascending bucket order.
    :raises ValueError: If ``bucket_seconds`` is not positive.
    """
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")

    current_key: datetime | None = None
    current: list[_Row] = []
    for row in rows:
        key = floor_bucket(ts_of(row), bucket_seconds)
        if current_key is None:
            current_key = key
        elif key != current_key:
            yield current_key, current
            current_key = key
            current = []
        current.append(row)
    if current_key is not None:
        yield current_key, current
