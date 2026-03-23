"""Circuit breaker pattern for protecting external service calls.

Implements the standard three-state circuit breaker (CLOSED -> OPEN -> HALF_OPEN)
to prevent cascading failures when downstream services are unhealthy.

Usage::

    cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    result = cb.call(some_function, arg1, arg2)

    # Or as a decorator:
    @circuit_breaker(failure_threshold=3)
    def fragile_call():
        ...
"""

from __future__ import annotations

import enum
import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    """States of the circuit breaker."""

    CLOSED = "closed"  # Normal operation — requests pass through
    OPEN = "open"  # Failing — requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing recovery — limited requests allowed


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the circuit is open."""

    def __init__(self, message: str = "Circuit breaker is open", *, remaining: float = 0.0) -> None:
        super().__init__(message)
        self.remaining = remaining


@dataclass
class CircuitBreakerConfig:
    """Configuration for a CircuitBreaker instance.

    Parameters
    ----------
    failure_threshold : int
        Number of consecutive failures before the circuit opens.
    recovery_timeout : float
        Seconds to wait before transitioning from OPEN to HALF_OPEN.
    success_threshold : int
        Number of consecutive successes in HALF_OPEN state to close the circuit.
    excluded_exceptions : tuple
        Exception types that should NOT count as failures (e.g. validation errors).
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2
    excluded_exceptions: tuple[type[BaseException], ...] = ()


@dataclass
class CircuitBreakerStats:
    """Snapshot of circuit breaker statistics."""

    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: float | None = None


class CircuitBreaker:
    """Thread-safe circuit breaker protecting a callable.

    Parameters
    ----------
    config : CircuitBreakerConfig
        Tuning parameters for the breaker.
    name : str
        Human-readable name for logging.
    on_open : callable
        Callback fired when the circuit transitions to OPEN.
    on_close : callable
        Callback fired when the circuit transitions to CLOSED.
    on_half_open : callable
        Callback fired when the circuit transitions to HALF_OPEN.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        name: str = "default",
        on_open: Callable[[], Any] | None = None,
        on_close: Callable[[], Any] | None = None,
        on_half_open: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config or CircuitBreakerConfig()
        self._name = name
        self._lock = threading.Lock()

        # State
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes_half_open = 0
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None

        # Stats
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

        # Callbacks
        self._on_open = on_open
        self._on_close = on_close
        self._on_half_open = on_half_open

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition from OPEN -> HALF_OPEN on read)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Return a snapshot of accumulated statistics."""
        with self._lock:
            return CircuitBreakerStats(
                total_calls=self._total_calls,
                total_failures=self._total_failures,
                total_successes=self._total_successes,
                consecutive_failures=self._consecutive_failures,
                last_failure_time=self._last_failure_time,
            )

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *fn* through the circuit breaker.

        Raises ``CircuitOpenError`` if the circuit is OPEN and the recovery
        timeout has not elapsed.
        """
        with self._lock:
            self._total_calls += 1
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                remaining = self._recovery_remaining()
                raise CircuitOpenError(
                    f"Circuit breaker '{self._name}' is open — retry after {remaining:.1f}s",
                    remaining=remaining,
                )

        # Execute outside the lock so we don't block other threads
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            self._handle_failure(exc)
            raise
        else:
            self._handle_success()
            return result

    def reset(self) -> None:
        """Force the circuit back to CLOSED state."""
        with self._lock:
            old = self._state
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes_half_open = 0
            self._opened_at = None
            if old != CircuitState.CLOSED:
                logger.info("Circuit breaker '%s' manually reset to CLOSED", self._name)
                self._fire_callback(self._on_close)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """Must be called while holding ``_lock``."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes_half_open = 0
                logger.info("Circuit breaker '%s' transitioning to HALF_OPEN", self._name)
                self._fire_callback(self._on_half_open)

    def _recovery_remaining(self) -> float:
        """Seconds until OPEN -> HALF_OPEN transition. Must hold lock."""
        if self._opened_at is None:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self._config.recovery_timeout - elapsed)

    def _handle_success(self) -> None:
        with self._lock:
            self._total_successes += 1
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes_half_open += 1
                if self._consecutive_successes_half_open >= self._config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._opened_at = None
                    logger.info("Circuit breaker '%s' recovered -> CLOSED", self._name)
                    self._fire_callback(self._on_close)

    def _handle_failure(self, exc: BaseException) -> None:
        # Skip excluded exception types
        if isinstance(exc, self._config.excluded_exceptions):
            return

        with self._lock:
            self._total_failures += 1
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately re-opens
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "Circuit breaker '%s' failure in HALF_OPEN -> OPEN",
                    self._name,
                )
                self._fire_callback(self._on_open)

            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "Circuit breaker '%s' threshold reached (%d failures) -> OPEN",
                        self._name,
                        self._consecutive_failures,
                    )
                    self._fire_callback(self._on_open)

    @staticmethod
    def _fire_callback(cb: Callable[[], Any] | None) -> None:
        if cb is not None:
            try:
                cb()
            except Exception:
                logger.exception("Circuit breaker callback raised an exception")


# ------------------------------------------------------------------
# Decorator form
# ------------------------------------------------------------------

def circuit_breaker(
    *,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    success_threshold: int = 2,
    excluded_exceptions: tuple[type[BaseException], ...] = (),
    name: str | None = None,
    on_open: Callable[[], Any] | None = None,
    on_close: Callable[[], Any] | None = None,
    on_half_open: Callable[[], Any] | None = None,
) -> Callable:
    """Decorator that wraps a function with a circuit breaker.

    Usage::

        @circuit_breaker(failure_threshold=3, recovery_timeout=10)
        def call_external_service():
            ...
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        excluded_exceptions=excluded_exceptions,
    )

    def decorator(fn: Callable) -> Callable:
        cb = CircuitBreaker(
            config=config,
            name=name or fn.__qualname__,
            on_open=on_open,
            on_close=on_close,
            on_half_open=on_half_open,
        )

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cb.call(fn, *args, **kwargs)

        # Expose the breaker instance for introspection / testing
        wrapper.circuit_breaker = cb  # type: ignore[attr-defined]
        return wrapper

    return decorator
