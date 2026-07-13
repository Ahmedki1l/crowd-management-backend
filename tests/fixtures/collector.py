"""A thread-safe bus subscriber for driving the real, threaded pipeline in tests.

The pipeline publishes from its worker thread, so collection must be guarded. A
:class:`threading.Condition` lets a test block until a predicate over the collected
events holds (e.g. "three OccupancyUpdates have arrived") instead of sleeping for a
fixed time; the condition is notified on every published event.
"""

from __future__ import annotations

import threading

from app.events.events import Event


class EventCollector:
    """Records every event published to the bus, and lets a test await a count."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._events: list[Event] = []

    def __call__(self, event: Event) -> None:
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    def of_type(self, event_type: type) -> list[Event]:
        """Return a snapshot of the collected events of ``event_type``."""
        with self._cond:
            return [e for e in self._events if isinstance(e, event_type)]

    def wait_for_count(self, event_type: type, count: int, timeout: float) -> bool:
        """Block until ``count`` events of ``event_type`` have been collected.

        Returns ``True`` once the threshold is reached, ``False`` on timeout. The wait
        is driven by the bus condition (woken on each event), so it returns as soon as
        the condition is met — never a fixed sleep. The predicate runs while the lock is
        held, so it reads ``self._events`` directly rather than re-entering
        :meth:`of_type`.
        """

        def _reached() -> bool:
            return sum(isinstance(e, event_type) for e in self._events) >= count

        with self._cond:
            return self._cond.wait_for(_reached, timeout=timeout)
