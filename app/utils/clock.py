"""Clock implementations (HLD 12.6 — deterministic dwell/debounce tests)."""

from __future__ import annotations

import time

from app.domain.interfaces import Clock


class SystemClock:
    """Wall-clock time. Production default."""

    def now(self) -> float:
        return time.time()


class FakeClock:
    """Controllable clock for tests. ``advance`` moves time forward."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        self._t += seconds
        return self._t

    def set(self, t: float) -> None:
        self._t = t


_default_clock: Clock = SystemClock()


def system_clock() -> Clock:
    return _default_clock
