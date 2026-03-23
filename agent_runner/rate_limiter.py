"""Rate limiting primitives for concurrent agent operations.

Provides a token bucket, sliding window counter, and a high-level
run limiter that enforces per-model and global concurrency limits.

Usage::

    limiter = RunLimiter(max_concurrent=5, max_per_minute=20)
    with limiter.acquire_run(model="claude-sonnet-4-20250514"):
        result = runner.run(prompt)
"""

from __future__ import annotations

import collections
import contextlib
import logging
import threading
import time
from typing import Generator

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a rate limit is hit."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ------------------------------------------------------------------
# Token bucket
# ------------------------------------------------------------------

class TokenBucket:
    """Classic token bucket algorithm — thread-safe.

    Parameters
    ----------
    capacity : int
        Maximum number of tokens the bucket can hold.
    refill_rate : float
        Tokens added per second.
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be positive")

        self.capacity = capacity
        self.refill_rate = refill_rate
        self._tokens: float = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Add tokens based on elapsed time.  Must hold lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        self._last_refill = now

    def consume(self, n: int = 1) -> bool:
        """Try to consume *n* tokens.

        Returns True if tokens were available, False otherwise.
        """
        with self._lock:
            self._refill()
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False

    def wait(self, n: int = 1, timeout: float | None = None) -> bool:
        """Block until *n* tokens are available, then consume them.

        Parameters
        ----------
        n : int
            Number of tokens to consume.
        timeout : float | None
            Maximum seconds to wait.  ``None`` means wait forever.

        Returns
        -------
        bool
            True if tokens were consumed, False if timed out.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return True
                # Calculate how long until enough tokens refill
                deficit = n - self._tokens
                wait_time = deficit / self.refill_rate

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait_time = min(wait_time, remaining)

            time.sleep(min(wait_time, 0.1))  # Cap sleep interval for responsiveness

    @property
    def available(self) -> float:
        """Current number of available tokens (approximate)."""
        with self._lock:
            self._refill()
            return self._tokens


# ------------------------------------------------------------------
# Sliding window counter
# ------------------------------------------------------------------

class SlidingWindowCounter:
    """Counts events in a sliding time window — thread-safe.

    Stores individual event timestamps so the window slides continuously
    rather than snapping to fixed intervals.
    """

    def __init__(self, max_window: float = 3600.0) -> None:
        self._max_window = max_window
        self._events: collections.deque[float] = collections.deque()
        self._lock = threading.Lock()

    def record(self) -> None:
        """Record that an event happened now."""
        now = time.monotonic()
        with self._lock:
            self._events.append(now)
            self._prune(now)

    def count(self, window_seconds: float) -> int:
        """Number of events in the last *window_seconds*."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            self._prune(now)
            return sum(1 for t in self._events if t >= cutoff)

    def is_within_limit(self, limit: int, window_seconds: float) -> bool:
        """Return True if event count in the window is below *limit*."""
        return self.count(window_seconds) < limit

    def _prune(self, now: float) -> None:
        """Remove events older than max_window. Must hold lock."""
        cutoff = now - self._max_window
        while self._events and self._events[0] < cutoff:
            self._events.popleft()


# ------------------------------------------------------------------
# RunLimiter — high-level rate limiter for agent runs
# ------------------------------------------------------------------

class RunLimiter:
    """Rate-limits agent runs by concurrency, per-minute, per-hour, and per-model.

    Parameters
    ----------
    max_concurrent : int
        Maximum simultaneous runs (0 = unlimited).
    max_per_minute : int
        Maximum runs started per minute (0 = unlimited).
    max_per_hour : int
        Maximum runs started per hour (0 = unlimited).
    per_model_limits : dict[str, int] | None
        Optional map of model -> max concurrent runs for that model.
    """

    def __init__(
        self,
        max_concurrent: int = 0,
        max_per_minute: int = 0,
        max_per_hour: int = 0,
        per_model_limits: dict[str, int] | None = None,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._max_per_minute = max_per_minute
        self._max_per_hour = max_per_hour
        self._per_model_limits = per_model_limits or {}

        # Concurrency tracking
        self._semaphore = threading.Semaphore(max_concurrent) if max_concurrent > 0 else None
        self._model_semaphores: dict[str, threading.Semaphore] = {}
        self._model_sem_lock = threading.Lock()

        # Rate tracking
        self._run_counter = SlidingWindowCounter(max_window=3600.0)

        # Active run count (for introspection)
        self._active_runs = 0
        self._active_lock = threading.Lock()

    def _get_model_semaphore(self, model: str) -> threading.Semaphore | None:
        """Lazily create per-model semaphore."""
        limit = self._per_model_limits.get(model)
        if limit is None or limit <= 0:
            return None
        with self._model_sem_lock:
            if model not in self._model_semaphores:
                self._model_semaphores[model] = threading.Semaphore(limit)
            return self._model_semaphores[model]

    @contextlib.contextmanager
    def acquire_run(self, model: str | None = None) -> Generator[None, None, None]:
        """Context manager that enforces all rate limits.

        Raises ``RateLimitExceeded`` if any limit is exceeded.

        Usage::

            with limiter.acquire_run(model="claude-sonnet-4-20250514"):
                result = runner.run(prompt)
        """
        # Check rate limits (per-minute / per-hour)
        if self._max_per_minute > 0:
            if not self._run_counter.is_within_limit(self._max_per_minute, 60.0):
                raise RateLimitExceeded(
                    f"Rate limit exceeded: max {self._max_per_minute} runs per minute",
                    retry_after=60.0,
                )

        if self._max_per_hour > 0:
            if not self._run_counter.is_within_limit(self._max_per_hour, 3600.0):
                raise RateLimitExceeded(
                    f"Rate limit exceeded: max {self._max_per_hour} runs per hour",
                    retry_after=3600.0,
                )

        # Check global concurrency
        if self._semaphore is not None:
            acquired = self._semaphore.acquire(blocking=False)
            if not acquired:
                raise RateLimitExceeded(
                    f"Concurrency limit exceeded: max {self._max_concurrent} concurrent runs",
                )
        else:
            acquired = False

        # Check per-model concurrency
        model_sem = self._get_model_semaphore(model) if model else None
        model_acquired = False
        if model_sem is not None:
            model_acquired = model_sem.acquire(blocking=False)
            if not model_acquired:
                # Release global semaphore if we took it
                if acquired:
                    self._semaphore.release()  # type: ignore[union-attr]
                limit = self._per_model_limits.get(model, 0)
                raise RateLimitExceeded(
                    f"Per-model concurrency limit exceeded: max {limit} concurrent runs for {model}",
                )

        # Record the run
        self._run_counter.record()
        with self._active_lock:
            self._active_runs += 1

        try:
            yield
        finally:
            with self._active_lock:
                self._active_runs -= 1
            if model_acquired and model_sem is not None:
                model_sem.release()
            if acquired and self._semaphore is not None:
                self._semaphore.release()

    @property
    def active_runs(self) -> int:
        """Number of currently active runs."""
        with self._active_lock:
            return self._active_runs

    @property
    def stats(self) -> dict[str, int]:
        """Snapshot of rate limiter statistics."""
        return {
            "active_runs": self.active_runs,
            "runs_last_minute": self._run_counter.count(60.0),
            "runs_last_hour": self._run_counter.count(3600.0),
        }
