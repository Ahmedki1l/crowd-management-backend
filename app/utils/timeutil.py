"""Epoch-second <-> timezone-aware datetime conversions (HLD 6.5).

The pipeline carries timestamps as float epoch seconds (``FramePacket.ts`` and
the various ``*Event.ts`` fields). DB ``ts`` columns are timezone-aware UTC
datetimes. Repositories and services convert at the persistence boundary using
these helpers so the conversion lives in exactly one place.
"""

from __future__ import annotations

from datetime import UTC, datetime


def to_datetime(epoch: float) -> datetime:
    """Convert epoch seconds to a timezone-aware UTC ``datetime``.

    :param epoch: Seconds since the Unix epoch (UTC).
    :returns: An aware ``datetime`` in UTC.
    """
    return datetime.fromtimestamp(epoch, tz=UTC)


def to_epoch(dt: datetime) -> float:
    """Convert a ``datetime`` to epoch seconds (UTC).

    Naive datetimes are assumed to already be expressed in UTC; this matches the
    DB contract where all stored timestamps are UTC even if a dialect (e.g.
    SQLite) drops the tzinfo on round-trip.

    :param dt: An aware or naive ``datetime`` (naive treated as UTC).
    :returns: Seconds since the Unix epoch.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def floor_bucket(dt: datetime, seconds: int) -> datetime:
    """Floor ``dt`` to the start of its ``seconds``-wide bucket.

    Buckets are aligned to the Unix epoch so the same wall-clock instant always
    maps to the same bucket boundary regardless of the query window. The result
    is timezone-aware in UTC.

    :param dt: The timestamp to floor (naive treated as UTC).
    :param seconds: Bucket width in seconds; must be positive.
    :returns: The aware UTC ``datetime`` at the start of the bucket.
    :raises ValueError: If ``seconds`` is not positive.
    """
    if seconds <= 0:
        raise ValueError("bucket width 'seconds' must be positive")
    epoch = to_epoch(dt)
    floored = (int(epoch) // seconds) * seconds
    return datetime.fromtimestamp(floored, tz=UTC)
