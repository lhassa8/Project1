"""Graceful shutdown manager for the agent runner.

Handles OS signals (SIGTERM, SIGINT), tracks active operations, and runs
registered shutdown hooks in order when the process is asked to stop.

Usage::

    from agent_runner.graceful import ShutdownManager

    sm = ShutdownManager(grace_period=30.0)
    sm.register_hook("close_mcp", mcp_client.stop)

    with sm.operation("run_agent"):
        runner.run(prompt)

    # On SIGTERM: waits for active operations, then runs hooks.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ShutdownManager:
    """Coordinate graceful shutdown of the agent runner process.

    Parameters
    ----------
    grace_period : float
        Maximum seconds to wait for active operations to drain before
        forcefully running hooks and exiting.
    install_signals : bool
        If ``True`` (default), install handlers for SIGTERM and SIGINT.
    """

    def __init__(
        self,
        grace_period: float = 30.0,
        install_signals: bool = True,
    ) -> None:
        self._grace_period = grace_period
        self._lock = threading.Lock()
        self._shutdown_requested = threading.Event()
        self._hooks: list[tuple[str, Callable[[], Any]]] = []
        self._active_ops: dict[str, int] = {}
        self._active_count = 0
        self._drain_event = threading.Event()
        self._drain_event.set()  # Start as "drained" (no active ops)
        self._hooks_executed = False

        if install_signals:
            self._install_signal_handlers()

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install handlers for SIGTERM and SIGINT.

        Only installs from the main thread (signal module constraint).
        """
        if threading.current_thread() is not threading.main_thread():
            logger.debug("ShutdownManager: not main thread, skipping signal install")
            return
        try:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
        except (OSError, ValueError):
            logger.debug("ShutdownManager: could not install signal handlers", exc_info=True)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — initiating graceful shutdown", sig_name)
        self.request_shutdown()

    # ------------------------------------------------------------------
    # Shutdown lifecycle
    # ------------------------------------------------------------------

    def request_shutdown(self) -> None:
        """Request a graceful shutdown.

        Can be called from any thread (e.g. a signal handler or health
        check).  Starts the drain-then-hook sequence in a background
        thread so the caller is not blocked.
        """
        if self._shutdown_requested.is_set():
            return
        self._shutdown_requested.set()
        t = threading.Thread(target=self._run_shutdown, daemon=True, name="shutdown-manager")
        t.start()

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested.is_set()

    def _run_shutdown(self) -> None:
        """Background thread that waits for drain then runs hooks."""
        logger.info(
            "Shutdown: waiting up to %.1fs for %d active operation(s) to complete",
            self._grace_period, self._active_count,
        )
        drained = self.wait_for_drain(self._grace_period)
        if not drained:
            logger.warning(
                "Shutdown: grace period expired with %d active operation(s) remaining",
                self._active_count,
            )
        self._execute_hooks()

    def _execute_hooks(self) -> None:
        with self._lock:
            if self._hooks_executed:
                return
            self._hooks_executed = True
            hooks = list(self._hooks)

        for name, fn in hooks:
            try:
                logger.info("Shutdown hook: %s", name)
                fn()
            except Exception:
                logger.exception("Shutdown hook '%s' failed", name)

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register_hook(self, name: str, fn: Callable[[], Any]) -> None:
        """Register a callable to run during shutdown.

        Hooks execute in registration order.
        """
        with self._lock:
            self._hooks.append((name, fn))

    # ------------------------------------------------------------------
    # Active operation tracking
    # ------------------------------------------------------------------

    def begin_operation(self, name: str = "default") -> None:
        """Mark the start of an active operation."""
        with self._lock:
            self._active_ops[name] = self._active_ops.get(name, 0) + 1
            self._active_count += 1
            self._drain_event.clear()

    def end_operation(self, name: str = "default") -> None:
        """Mark the end of an active operation."""
        with self._lock:
            current = self._active_ops.get(name, 0)
            if current > 1:
                self._active_ops[name] = current - 1
            else:
                self._active_ops.pop(name, None)
            self._active_count = max(0, self._active_count - 1)
            if self._active_count == 0:
                self._drain_event.set()

    @contextmanager
    def operation(self, name: str = "default"):
        """Context manager that tracks an active operation.

        If a shutdown is requested while inside the context, the manager
        waits for the block to finish before running hooks.

        Raises :class:`ShutdownInProgress` if shutdown was already
        requested when entering the context.
        """
        if self._shutdown_requested.is_set():
            raise ShutdownInProgress(f"Cannot start operation '{name}': shutdown in progress")
        self.begin_operation(name)
        try:
            yield
        finally:
            self.end_operation(name)

    @property
    def active_count(self) -> int:
        return self._active_count

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    def wait_for_drain(self, timeout: float | None = None) -> bool:
        """Block until all active operations complete.

        Returns ``True`` if drained, ``False`` if *timeout* expired.
        """
        if timeout is None:
            timeout = self._grace_period
        return self._drain_event.wait(timeout)


class ShutdownInProgress(RuntimeError):
    """Raised when a new operation is attempted after shutdown was requested."""
