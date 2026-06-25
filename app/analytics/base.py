"""Shared base for the analytics calculators (HLD 7).

Calculators are *pure*: each holds per-camera in-memory state and turns
localisation outputs (confirmed membership, crossings, transitions, tracked
detections) into immutable :mod:`app.events.events` events. They never touch the
database or the event bus — the engine publishes whatever ``process`` returns and
owns all persistence/snapshot enrichment.

This module keeps only what is genuinely shared (the clock dependency); per-HLD
each calculator owns its own state shape, so there is no premature abstraction
here (YAGNI).
"""

from __future__ import annotations

from app.domain.interfaces import Clock


class BaseCalculator:
    """Common base for analytics calculators.

    Subclasses implement a ``process(...)`` (or ``accumulate``/``tick``) method
    that returns a list of events. The only shared dependency is the
    :class:`~app.domain.interfaces.Clock`, exposed so subclasses and tests can
    reason about time deterministically. Pipeline timestamps still flow in as
    explicit ``ts`` arguments (float epoch seconds); the clock is used only for
    derived bookkeeping that has no incoming frame timestamp.
    """

    def __init__(self, clock: Clock) -> None:
        """Store the time source shared by all calculators.

        Args:
            clock: The pipeline time source (epoch seconds).
        """
        self.clock: Clock = clock
