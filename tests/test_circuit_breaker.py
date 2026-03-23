"""Tests for agent_runner.circuit_breaker."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock

import pytest

from agent_runner.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    circuit_breaker,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

class TransientError(Exception):
    """Simulates a transient downstream failure."""


class ValidationError(Exception):
    """An error type we want to exclude from failure counting."""


def _succeed() -> str:
    return "ok"


def _fail() -> str:
    raise TransientError("boom")


# ------------------------------------------------------------------
# State transitions
# ------------------------------------------------------------------

class TestStateTransitions:
    """CLOSED -> OPEN -> HALF_OPEN -> CLOSED lifecycle."""

    def test_starts_closed(self) -> None:
        cb = CircuitBreaker()
        assert cb.state is CircuitState.CLOSED

    def test_opens_after_threshold(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0)
        cb = CircuitBreaker(config=cfg)

        for _ in range(3):
            with pytest.raises(TransientError):
                cb.call(_fail)

        assert cb.state is CircuitState.OPEN

    def test_rejects_calls_when_open(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)

        with pytest.raises(CircuitOpenError):
            cb.call(_succeed)

    def test_transitions_to_half_open_after_timeout(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)
        assert cb.state is CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state is CircuitState.HALF_OPEN

    def test_closes_after_success_threshold_in_half_open(self) -> None:
        cfg = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.05,
            success_threshold=2,
        )
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)
        assert cb.state is CircuitState.OPEN

        time.sleep(0.1)
        assert cb.state is CircuitState.HALF_OPEN

        # Two successes to close
        assert cb.call(_succeed) == "ok"
        assert cb.state is CircuitState.HALF_OPEN  # still half-open after 1
        assert cb.call(_succeed) == "ok"
        assert cb.state is CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05)
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)
        time.sleep(0.1)
        assert cb.state is CircuitState.HALF_OPEN

        with pytest.raises(TransientError):
            cb.call(_fail)
        assert cb.state is CircuitState.OPEN


# ------------------------------------------------------------------
# Failure counting
# ------------------------------------------------------------------

class TestFailureCounting:
    def test_consecutive_failures_tracked(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(config=cfg)

        for _ in range(4):
            with pytest.raises(TransientError):
                cb.call(_fail)

        assert cb.stats.consecutive_failures == 4
        assert cb.stats.total_failures == 4

    def test_success_resets_consecutive_failures(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(config=cfg)

        for _ in range(3):
            with pytest.raises(TransientError):
                cb.call(_fail)

        cb.call(_succeed)
        assert cb.stats.consecutive_failures == 0
        assert cb.stats.total_failures == 3
        assert cb.stats.total_successes == 1

    def test_last_failure_time_set(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(config=cfg)

        assert cb.stats.last_failure_time is None
        with pytest.raises(TransientError):
            cb.call(_fail)
        assert cb.stats.last_failure_time is not None

    def test_total_calls_includes_rejected(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)
        with pytest.raises(CircuitOpenError):
            cb.call(_succeed)

        assert cb.stats.total_calls == 2


# ------------------------------------------------------------------
# Recovery timeout
# ------------------------------------------------------------------

class TestRecoveryTimeout:
    def test_circuit_open_error_has_remaining(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)

        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(_succeed)

        assert exc_info.value.remaining > 0

    def test_short_recovery_timeout(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05)
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)
        assert cb.state is CircuitState.OPEN

        time.sleep(0.1)
        # Should be able to attempt a call now
        result = cb.call(_succeed)
        assert result == "ok"


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_calls(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=100)
        cb = CircuitBreaker(config=cfg)
        results = []

        def worker() -> None:
            try:
                r = cb.call(_succeed)
                results.append(r)
            except Exception:
                pass

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50
        assert cb.stats.total_successes == 50

    def test_concurrent_failures(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=100)
        cb = CircuitBreaker(config=cfg)

        def worker() -> None:
            try:
                cb.call(_fail)
            except TransientError:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(worker) for _ in range(50)]
            concurrent.futures.wait(futures)

        assert cb.stats.total_failures == 50


# ------------------------------------------------------------------
# Decorator usage
# ------------------------------------------------------------------

class TestDecoratorUsage:
    def test_decorator_passes_through_success(self) -> None:
        @circuit_breaker(failure_threshold=3)
        def greet(name: str) -> str:
            return f"hello {name}"

        assert greet("world") == "hello world"

    def test_decorator_opens_circuit(self) -> None:
        call_count = 0

        @circuit_breaker(failure_threshold=2, recovery_timeout=60.0)
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            raise TransientError("oops")

        for _ in range(2):
            with pytest.raises(TransientError):
                flaky()

        with pytest.raises(CircuitOpenError):
            flaky()

        # The third call should have been rejected before calling the fn
        assert call_count == 2

    def test_decorator_exposes_circuit_breaker(self) -> None:
        @circuit_breaker(failure_threshold=5, name="my_fn")
        def my_fn() -> str:
            return "ok"

        assert hasattr(my_fn, "circuit_breaker")
        assert isinstance(my_fn.circuit_breaker, CircuitBreaker)

    def test_decorator_preserves_function_name(self) -> None:
        @circuit_breaker(failure_threshold=5)
        def important_function() -> str:
            """Docstring."""
            return "ok"

        assert important_function.__name__ == "important_function"
        assert important_function.__doc__ == "Docstring."


# ------------------------------------------------------------------
# Excluded exceptions
# ------------------------------------------------------------------

class TestExcludedExceptions:
    def test_excluded_exceptions_dont_count(self) -> None:
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValidationError,),
        )
        cb = CircuitBreaker(config=cfg)

        def raise_validation() -> None:
            raise ValidationError("bad input")

        # These should not count as failures
        for _ in range(5):
            with pytest.raises(ValidationError):
                cb.call(raise_validation)

        assert cb.state is CircuitState.CLOSED
        assert cb.stats.consecutive_failures == 0
        assert cb.stats.total_failures == 0

    def test_non_excluded_exceptions_still_count(self) -> None:
        cfg = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValidationError,),
        )
        cb = CircuitBreaker(config=cfg)

        for _ in range(2):
            with pytest.raises(TransientError):
                cb.call(_fail)

        assert cb.state is CircuitState.OPEN


# ------------------------------------------------------------------
# Event callbacks
# ------------------------------------------------------------------

class TestCallbacks:
    def test_on_open_called(self) -> None:
        on_open = MagicMock()
        cfg = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=cfg, on_open=on_open)

        with pytest.raises(TransientError):
            cb.call(_fail)

        on_open.assert_called_once()

    def test_on_close_called(self) -> None:
        on_close = MagicMock()
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05, success_threshold=1)
        cb = CircuitBreaker(config=cfg, on_close=on_close)

        with pytest.raises(TransientError):
            cb.call(_fail)
        time.sleep(0.1)
        cb.call(_succeed)

        on_close.assert_called_once()

    def test_on_half_open_called(self) -> None:
        on_half_open = MagicMock()
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.05)
        cb = CircuitBreaker(config=cfg, on_half_open=on_half_open)

        with pytest.raises(TransientError):
            cb.call(_fail)
        time.sleep(0.1)
        _ = cb.state  # triggers transition

        on_half_open.assert_called_once()

    def test_reset_fires_on_close(self) -> None:
        on_close = MagicMock()
        cfg = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config=cfg, on_close=on_close)

        with pytest.raises(TransientError):
            cb.call(_fail)

        cb.reset()
        on_close.assert_called_once()
        assert cb.state is CircuitState.CLOSED


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self) -> None:
        cfg = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60.0)
        cb = CircuitBreaker(config=cfg)

        with pytest.raises(TransientError):
            cb.call(_fail)
        assert cb.state is CircuitState.OPEN

        cb.reset()
        assert cb.state is CircuitState.CLOSED
        assert cb.stats.consecutive_failures == 0

        # Should be able to call again
        assert cb.call(_succeed) == "ok"
