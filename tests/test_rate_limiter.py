"""Tests for agent_runner.rate_limiter."""

from __future__ import annotations

import concurrent.futures
import threading
import time

import pytest

from agent_runner.rate_limiter import (
    RateLimitExceeded,
    RunLimiter,
    SlidingWindowCounter,
    TokenBucket,
)


# ------------------------------------------------------------------
# TokenBucket
# ------------------------------------------------------------------

class TestTokenBucket:
    def test_consume_within_capacity(self) -> None:
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        for _ in range(5):
            assert bucket.consume() is True

    def test_consume_exhausts_tokens(self) -> None:
        bucket = TokenBucket(capacity=3, refill_rate=1.0)
        assert bucket.consume(3) is True
        assert bucket.consume(1) is False

    def test_consume_multiple_tokens(self) -> None:
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(5) is True
        assert bucket.consume(5) is True
        assert bucket.consume(1) is False

    def test_refill_over_time(self) -> None:
        bucket = TokenBucket(capacity=5, refill_rate=100.0)  # fast refill
        assert bucket.consume(5) is True
        assert bucket.consume(1) is False

        time.sleep(0.1)  # ~10 tokens refilled, capped at 5
        assert bucket.consume(1) is True

    def test_refill_does_not_exceed_capacity(self) -> None:
        bucket = TokenBucket(capacity=3, refill_rate=100.0)
        time.sleep(0.1)
        # Even after waiting, available should be capped at capacity
        assert bucket.available <= 3.0

    def test_wait_blocks_until_available(self) -> None:
        bucket = TokenBucket(capacity=1, refill_rate=50.0)  # fast refill
        assert bucket.consume(1) is True

        start = time.monotonic()
        result = bucket.wait(1, timeout=1.0)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed < 1.0

    def test_wait_timeout(self) -> None:
        bucket = TokenBucket(capacity=1, refill_rate=0.1)  # slow refill
        assert bucket.consume(1) is True

        result = bucket.wait(1, timeout=0.1)
        assert result is False

    def test_invalid_capacity_raises(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(capacity=0, refill_rate=1.0)

    def test_invalid_refill_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="refill_rate"):
            TokenBucket(capacity=5, refill_rate=0)

    def test_thread_safety(self) -> None:
        bucket = TokenBucket(capacity=100, refill_rate=0.0001)  # near-zero refill
        consumed = []
        lock = threading.Lock()

        def worker() -> None:
            result = bucket.consume(1)
            with lock:
                consumed.append(result)

        threads = [threading.Thread(target=worker) for _ in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        successes = sum(1 for r in consumed if r)
        # Should be exactly capacity (100) successful, rest fail
        assert successes == 100
        assert len(consumed) == 200


# ------------------------------------------------------------------
# SlidingWindowCounter
# ------------------------------------------------------------------

class TestSlidingWindowCounter:
    def test_count_empty(self) -> None:
        counter = SlidingWindowCounter()
        assert counter.count(60.0) == 0

    def test_record_and_count(self) -> None:
        counter = SlidingWindowCounter()
        for _ in range(5):
            counter.record()
        assert counter.count(60.0) == 5

    def test_count_only_within_window(self) -> None:
        counter = SlidingWindowCounter()
        counter.record()
        time.sleep(0.15)
        counter.record()
        counter.record()

        # Short window: only recent events
        assert counter.count(0.1) == 2
        # Longer window: all events
        assert counter.count(1.0) == 3

    def test_is_within_limit_true(self) -> None:
        counter = SlidingWindowCounter()
        for _ in range(3):
            counter.record()
        assert counter.is_within_limit(5, 60.0) is True

    def test_is_within_limit_false(self) -> None:
        counter = SlidingWindowCounter()
        for _ in range(5):
            counter.record()
        assert counter.is_within_limit(5, 60.0) is False

    def test_events_expire(self) -> None:
        counter = SlidingWindowCounter(max_window=0.2)
        counter.record()
        assert counter.count(1.0) == 1

        time.sleep(0.3)
        # Event should have been pruned
        assert counter.count(1.0) == 0

    def test_thread_safety(self) -> None:
        counter = SlidingWindowCounter()

        def worker() -> None:
            for _ in range(10):
                counter.record()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.count(60.0) == 100


# ------------------------------------------------------------------
# RunLimiter — concurrent access
# ------------------------------------------------------------------

class TestRunLimiter:
    def test_allows_within_concurrent_limit(self) -> None:
        limiter = RunLimiter(max_concurrent=3)
        with limiter.acquire_run():
            assert limiter.active_runs == 1
        assert limiter.active_runs == 0

    def test_rejects_over_concurrent_limit(self) -> None:
        limiter = RunLimiter(max_concurrent=1)

        with limiter.acquire_run():
            with pytest.raises(RateLimitExceeded, match="Concurrency"):
                with limiter.acquire_run():
                    pass  # pragma: no cover

    def test_rejects_over_per_minute_limit(self) -> None:
        limiter = RunLimiter(max_per_minute=2)

        with limiter.acquire_run():
            pass
        with limiter.acquire_run():
            pass

        with pytest.raises(RateLimitExceeded, match="per minute"):
            with limiter.acquire_run():
                pass  # pragma: no cover

    def test_rejects_over_per_hour_limit(self) -> None:
        limiter = RunLimiter(max_per_hour=2)

        with limiter.acquire_run():
            pass
        with limiter.acquire_run():
            pass

        with pytest.raises(RateLimitExceeded, match="per hour"):
            with limiter.acquire_run():
                pass  # pragma: no cover

    def test_concurrent_runs_counted(self) -> None:
        limiter = RunLimiter(max_concurrent=5)
        barrier = threading.Barrier(3)
        active_counts: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            with limiter.acquire_run():
                barrier.wait(timeout=2)
                with lock:
                    active_counts.append(limiter.active_runs)
                barrier.wait(timeout=2)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max(active_counts) == 3

    def test_unlimited_when_zero(self) -> None:
        limiter = RunLimiter(max_concurrent=0, max_per_minute=0, max_per_hour=0)
        # Should not raise
        for _ in range(10):
            with limiter.acquire_run():
                pass

    def test_stats(self) -> None:
        limiter = RunLimiter()
        with limiter.acquire_run():
            stats = limiter.stats
            assert stats["active_runs"] == 1
            assert stats["runs_last_minute"] >= 1


# ------------------------------------------------------------------
# Per-model limits
# ------------------------------------------------------------------

class TestPerModelLimits:
    def test_per_model_limit_enforced(self) -> None:
        limiter = RunLimiter(
            max_concurrent=10,
            per_model_limits={"claude-sonnet-4-20250514": 1},
        )

        with limiter.acquire_run(model="claude-sonnet-4-20250514"):
            with pytest.raises(RateLimitExceeded, match="Per-model"):
                with limiter.acquire_run(model="claude-sonnet-4-20250514"):
                    pass  # pragma: no cover

    def test_different_models_independent(self) -> None:
        limiter = RunLimiter(
            max_concurrent=10,
            per_model_limits={
                "claude-sonnet-4-20250514": 1,
                "claude-haiku-35": 1,
            },
        )

        with limiter.acquire_run(model="claude-sonnet-4-20250514"):
            # Different model should work fine
            with limiter.acquire_run(model="claude-haiku-35"):
                assert limiter.active_runs == 2

    def test_model_without_limit_allowed(self) -> None:
        limiter = RunLimiter(
            max_concurrent=10,
            per_model_limits={"claude-sonnet-4-20250514": 1},
        )

        # Model not in per_model_limits — no per-model restriction
        with limiter.acquire_run(model="other-model"):
            with limiter.acquire_run(model="other-model"):
                assert limiter.active_runs == 2

    def test_per_model_releases_global_on_failure(self) -> None:
        """When per-model limit fails, global semaphore should be released."""
        limiter = RunLimiter(
            max_concurrent=2,
            per_model_limits={"claude-sonnet-4-20250514": 1},
        )

        with limiter.acquire_run(model="claude-sonnet-4-20250514"):
            with pytest.raises(RateLimitExceeded):
                with limiter.acquire_run(model="claude-sonnet-4-20250514"):
                    pass  # pragma: no cover

        # Global semaphore should still have capacity
        with limiter.acquire_run(model="claude-sonnet-4-20250514"):
            assert limiter.active_runs == 1


# ------------------------------------------------------------------
# RateLimitExceeded exception
# ------------------------------------------------------------------

class TestRateLimitExceeded:
    def test_has_retry_after(self) -> None:
        exc = RateLimitExceeded("too fast", retry_after=30.0)
        assert exc.retry_after == 30.0
        assert "too fast" in str(exc)

    def test_retry_after_none_by_default(self) -> None:
        exc = RateLimitExceeded("too fast")
        assert exc.retry_after is None
