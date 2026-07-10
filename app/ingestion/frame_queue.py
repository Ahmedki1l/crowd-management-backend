"""Bounded, drop-oldest frame queue decoupling capture from inference (HLD 5.3).

Capture threads and the inference worker run at different, fluctuating rates: an
RTSP decoder delivers frames at the camera's wall-clock cadence, while inference
latency varies with model load and batch size. Coupling them with a blocking
queue would let a slow consumer back-pressure the decoder, which in turn stalls
the FFmpeg/RTSP read loop and eventually corrupts the stream (the decoder cannot
"pause" a live source — frames keep arriving and buffers overflow lower down).

This queue therefore favours *freshness over completeness*: when full, ``put``
silently discards the OLDEST frame to make room for the newest one and never
blocks the producer. For analytics on a live feed the most recent frame is the
most valuable; dropping stale frames under load is the correct behaviour, not an
error. The number of dropped frames is counted so the loss is observable rather
than hidden.
"""

from __future__ import annotations

import threading
from collections import deque

from app.domain.models import FramePacket
from app.utils.logging import get_logger

logger = get_logger(__name__)


class BoundedFrameQueue:
    """Thread-safe, bounded, drop-oldest queue of :class:`FramePacket`.

    Built on :class:`collections.deque` guarded by a :class:`threading.Condition`.
    A single lock protects the deque and the dropped-frame counter; the condition
    wakes blocked consumers when a packet becomes available.

    The producer never blocks: a full queue drops its oldest packet on ``put``.
    Consumers may block in ``get`` up to an optional timeout.
    """

    def __init__(self, maxsize: int) -> None:
        """Create a queue holding at most ``maxsize`` packets.

        Args:
            maxsize: Maximum number of buffered packets. Must be >= 1.

        Raises:
            ValueError: If ``maxsize`` is less than 1.
        """
        if maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {maxsize}")
        self._maxsize = maxsize
        self._cond = threading.Condition()
        self._buffer: deque[FramePacket] = deque()
        self._dropped = 0

    @property
    def maxsize(self) -> int:
        """The configured maximum number of buffered packets."""
        return self._maxsize

    @property
    def dropped(self) -> int:
        """Total number of packets dropped because the queue was full."""
        with self._cond:
            return self._dropped

    def put(self, packet: FramePacket) -> None:
        """Enqueue ``packet``, dropping the oldest packet if the queue is full.

        Never blocks the caller. When the queue is at capacity the oldest packet
        is discarded (and counted) before the new one is appended, so the most
        recent frame is always retained.

        Args:
            packet: The freshly decoded frame to enqueue.
        """
        with self._cond:
            if len(self._buffer) >= self._maxsize:
                self._buffer.popleft()
                self._dropped += 1
            self._buffer.append(packet)
            self._cond.notify()

    def get(self, timeout: float | None = None) -> FramePacket | None:
        """Dequeue the oldest packet, blocking until one is available.

        Args:
            timeout: Maximum seconds to wait for a packet. ``None`` waits
                indefinitely; ``0`` (or a negative value) polls without waiting.

        Returns:
            The oldest buffered :class:`FramePacket`, or ``None`` if the timeout
            elapses with the queue still empty.
        """
        with self._cond:
            if not self._buffer:
                # Condition.wait returns False on timeout (Python 3.2+).
                got = self._cond.wait_for(lambda: bool(self._buffer), timeout=timeout)
                if not got:
                    return None
            return self._buffer.popleft()

    def get_latest(self, timeout: float | None = None) -> FramePacket | None:
        """Dequeue the NEWEST packet, discarding any older ones still buffered.

        For roles that only report the current state of a scene (snapshot
        occupancy), a queued backlog is worse than useless: each stale packet
        costs a full inference and answers a question about the past. Because
        :meth:`get` is FIFO while :meth:`put` drops from the front, a depth-``N``
        queue silently adds ``(N-1) x producer_period`` of latency the moment the
        consumer falls behind the producer — the buffer becomes a delay line, not
        a shock absorber. Draining to the freshest packet keeps that latency at
        zero while still absorbing bursts.

        Skipped packets are counted as drops, so the loss stays observable. The
        packets returned are strictly newer over time, so downstream timestamps
        remain monotonic (unlike a plain LIFO pop, which would reorder frames and
        corrupt tracking/crossing logic).

        Args:
            timeout: Maximum seconds to wait for a packet. ``None`` waits
                indefinitely; ``0`` (or negative) polls without waiting.

        Returns:
            The most recent buffered :class:`FramePacket`, or ``None`` if the
            timeout elapses with the queue still empty.
        """
        with self._cond:
            if not self._buffer:
                got = self._cond.wait_for(lambda: bool(self._buffer), timeout=timeout)
                if not got:
                    return None
            skipped = len(self._buffer) - 1
            if skipped:
                self._dropped += skipped
            packet = self._buffer[-1]
            self._buffer.clear()
            return packet

    def qsize(self) -> int:
        """Return the number of packets currently buffered.

        This is a point-in-time snapshot; it may be stale the moment it returns
        in the presence of concurrent producers/consumers.
        """
        with self._cond:
            return len(self._buffer)

    def clear(self) -> None:
        """Discard all buffered packets without counting them as drops."""
        with self._cond:
            self._buffer.clear()
