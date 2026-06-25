"""Query-parameter time parsing shared by the metric/history routers (HLD 8.3).

Live-metric and history endpoints accept ``from``/``to`` window bounds and a
``bucket`` width as flexible query strings. To keep a single, consistent parsing
contract across every router (DRY), the conversions live here:

* a window bound may be epoch seconds (e.g. ``1718200000`` or ``1718200000.5``)
  or an ISO-8601 timestamp (e.g. ``2024-06-12T13:00:00Z``);
* a bucket width may be epoch seconds (``"3600"``) or a compact duration such as
  ``"30s"``, ``"15m"``, ``"1h"`` or ``"7d"``.

All returned datetimes are timezone-aware UTC, matching the DB ``ts`` contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException, status

_DURATION_UNITS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def parse_instant(value: str) -> datetime:
    """Parse a window bound into a timezone-aware UTC ``datetime``.

    Accepts either epoch seconds (int or float) or an ISO-8601 timestamp. A
    trailing ``Z`` is treated as UTC. Naive ISO inputs are assumed to be UTC.

    Args:
        value: The raw query-string value.

    Returns:
        The parsed instant as an aware UTC ``datetime``.

    Raises:
        HTTPException: ``422`` if the value is neither epoch seconds nor ISO.
    """
    epoch = _try_parse_epoch(value)
    if epoch is not None:
        return datetime.fromtimestamp(epoch, tz=UTC)

    parsed = _try_parse_iso(value)
    if parsed is not None:
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"invalid timestamp: {value!r} (expected epoch seconds or ISO-8601)",
    )


def parse_bucket_seconds(value: str) -> int:
    """Parse a bucket width into a positive whole number of seconds.

    Accepts bare seconds (``"3600"``) or a compact duration with a unit suffix
    (``"30s"``, ``"15m"``, ``"1h"``, ``"7d"``).

    Args:
        value: The raw ``bucket`` query-string value.

    Returns:
        The bucket width in seconds (always ``>= 1``).

    Raises:
        HTTPException: ``422`` if the value is not a recognised duration or is
            not positive.
    """
    raw = value.strip()
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bucket must not be empty",
        )

    unit = raw[-1].lower()
    if unit in _DURATION_UNITS:
        seconds = _scale_duration(raw[:-1], _DURATION_UNITS[unit], original=raw)
    else:
        seconds = _scale_duration(raw, 1, original=raw)

    if seconds <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"bucket must be positive: {value!r}",
        )
    return seconds


def _scale_duration(magnitude: str, unit_seconds: int, *, original: str) -> int:
    """Parse the numeric ``magnitude`` and scale it by ``unit_seconds``."""
    try:
        return int(magnitude) * unit_seconds
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid bucket: {original!r}",
        ) from exc


def _try_parse_epoch(value: str) -> float | None:
    """Return ``value`` as epoch seconds, or ``None`` if it is not numeric."""
    try:
        return float(value)
    except ValueError:
        return None


def _try_parse_iso(value: str) -> datetime | None:
    """Return ``value`` parsed as ISO-8601, or ``None`` if it is not ISO."""
    candidate = value.strip()
    if candidate.endswith(("z", "Z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None
