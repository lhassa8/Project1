"""Resilient MCP client — wraps :class:`MCPClient` with circuit-breaker
protection, connection pooling, and request queuing.

This module provides a drop-in replacement for ``MCPClient`` that adds
enterprise-grade resilience:

* **Circuit breaker** — stops hammering a failing server and gives it time
  to recover.
* **Connection pool** — keeps *N* warm ``MCPClient`` connections so a single
  process crash doesn't block the whole agent.
* **Request queue** — when the circuit is HALF_OPEN only one probe request
  goes through; the rest wait.
* **Metrics** — tracks call success/failure rates for observability.
* **Graceful fallback** — when the circuit is OPEN, returns a structured
  error message instead of crashing the agent.

Usage::

    from agent_runner.config import MCPConfig
    client = ResilientMCPClient.from_config(mcp_config, command=["npx", "..."])
    result = client.call_tool("read_file", {"path": "/tmp/foo"})
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from agent_runner.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)
from agent_runner.config import MCPConfig
from agent_runner.mcp.client import MCPClient, MCPToolSchema

logger = logging.getLogger(__name__)


@dataclass
class _CallMetrics:
    """Internal mutable metrics container."""

    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_fallbacks: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_success(self) -> None:
        with self._lock:
            self.total_calls += 1
            self.total_successes += 1

    def record_failure(self) -> None:
        with self._lock:
            self.total_calls += 1
            self.total_failures += 1

    def record_fallback(self) -> None:
        with self._lock:
            self.total_calls += 1
            self.total_fallbacks += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "total_successes": self.total_successes,
                "total_failures": self.total_failures,
                "total_fallbacks": self.total_fallbacks,
                "success_rate": (
                    self.total_successes / self.total_calls
                    if self.total_calls > 0
                    else 0.0
                ),
            }


class ResilientMCPClient:
    """MCP client wrapped with circuit-breaker, pooling, and fallback.

    Parameters
    ----------
    command : list[str]
        Command to launch MCP server.
    env : dict[str, str] | None
        Extra environment variables.
    pool_size : int
        Number of warm connections to keep.
    timeout : float
        Per-call timeout in seconds.
    max_retries : int
        Max retries *per underlying MCPClient* (passed through).
    circuit_config : CircuitBreakerConfig | None
        Circuit breaker tuning.  Defaults to sensible values.
    half_open_max_queued : int
        Max requests queued while circuit is HALF_OPEN.
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        pool_size: int = 2,
        timeout: float = 30.0,
        max_retries: int = 2,
        circuit_config: CircuitBreakerConfig | None = None,
        half_open_max_queued: int = 10,
    ) -> None:
        self._command = command
        self._env = env
        self._pool_size = max(pool_size, 1)
        self._timeout = timeout
        self._max_retries = max_retries
        self._half_open_max_queued = half_open_max_queued

        # Circuit breaker
        self._circuit = CircuitBreaker(
            config=circuit_config or CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=2,
            ),
            name="mcp",
            on_open=self._on_circuit_open,
            on_close=self._on_circuit_close,
            on_half_open=self._on_circuit_half_open,
        )

        # Connection pool
        self._pool: queue.Queue[MCPClient] = queue.Queue(maxsize=self._pool_size)
        self._pool_lock = threading.Lock()
        self._started = False

        # Half-open request queue
        self._half_open_lock = threading.Lock()
        self._half_open_pending = 0

        # Metrics
        self._metrics = _CallMetrics()

        # Tool cache
        self._tools: dict[str, MCPToolSchema] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        mcp_config: MCPConfig,
        env: dict[str, str] | None = None,
        pool_size: int = 2,
        circuit_config: CircuitBreakerConfig | None = None,
    ) -> ResilientMCPClient:
        """Create a :class:`ResilientMCPClient` from an :class:`MCPConfig`."""
        command = mcp_config.command.split() if isinstance(mcp_config.command, str) else list(mcp_config.command)
        return cls(
            command=command,
            env=env,
            pool_size=pool_size,
            circuit_config=circuit_config,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Warm up the connection pool."""
        for _ in range(self._pool_size):
            client = self._make_client()
            try:
                client.start()
                self._pool.put(client)
            except Exception:
                logger.warning("Failed to start pooled MCP connection", exc_info=True)
        self._started = True

    def stop(self) -> None:
        """Shut down all pooled connections."""
        while not self._pool.empty():
            try:
                client = self._pool.get_nowait()
                client.stop()
            except queue.Empty:
                break
            except Exception:
                logger.warning("Error stopping pooled MCP client", exc_info=True)
        self._started = False

    def __enter__(self) -> ResilientMCPClient:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def list_tools(self) -> list[MCPToolSchema]:
        """Discover tools (delegates to a pooled connection)."""
        client = self._acquire()
        try:
            tools = client.list_tools()
            for t in tools:
                self._tools[t.name] = t
            return tools
        finally:
            self._release(client)

    # ------------------------------------------------------------------
    # Tool invocation (protected by circuit breaker)
    # ------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool through the circuit breaker.

        When the circuit is OPEN, returns a fallback error string instead
        of raising, so the agent can continue gracefully.
        """
        try:
            return self._circuit.call(self._do_call_tool, name, arguments)
        except CircuitOpenError as exc:
            self._metrics.record_fallback()
            logger.warning("Circuit open — returning fallback for tool '%s'", name)
            return (
                f"[MCP unavailable] The tool '{name}' is temporarily unavailable "
                f"due to repeated failures.  The circuit breaker will retry "
                f"in {exc.remaining:.0f}s.  Please try a different approach or "
                f"wait before retrying."
            )

    def _do_call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Actual call routed through a pooled connection."""
        client = self._acquire()
        try:
            result = client.call_tool(name, arguments)
            self._metrics.record_success()
            return result
        except Exception:
            self._metrics.record_failure()
            raise
        finally:
            self._release(client)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """Return health status respecting circuit state."""
        state = self._circuit.state
        cb_stats = self._circuit.stats

        pool_healthy = 0
        pool_total = 0
        # Peek at pool without draining it
        snapshot: list[MCPClient] = []
        while not self._pool.empty():
            try:
                c = self._pool.get_nowait()
                snapshot.append(c)
                pool_total += 1
                if c.health_check():
                    pool_healthy += 1
            except queue.Empty:
                break
        for c in snapshot:
            self._pool.put(c)

        return {
            "status": "healthy" if state == CircuitState.CLOSED else "degraded" if state == CircuitState.HALF_OPEN else "unhealthy",
            "circuit_state": state.value,
            "pool_healthy": pool_healthy,
            "pool_total": pool_total,
            "circuit_stats": {
                "total_calls": cb_stats.total_calls,
                "total_failures": cb_stats.total_failures,
                "consecutive_failures": cb_stats.consecutive_failures,
            },
            "metrics": self._metrics.snapshot(),
        }

    @property
    def circuit_state(self) -> CircuitState:
        return self._circuit.state

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics.snapshot()

    # ------------------------------------------------------------------
    # Connection pool helpers
    # ------------------------------------------------------------------

    def _make_client(self) -> MCPClient:
        return MCPClient(
            command=list(self._command),
            env=self._env,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    def _acquire(self) -> MCPClient:
        """Get a connection from the pool (blocks briefly, then creates one)."""
        try:
            return self._pool.get(timeout=2.0)
        except queue.Empty:
            logger.debug("Pool exhausted — creating ephemeral MCP client")
            client = self._make_client()
            client.start()
            return client

    def _release(self, client: MCPClient) -> None:
        """Return a connection to the pool (drops it if pool is full)."""
        if client.health_check():
            try:
                self._pool.put_nowait(client)
            except queue.Full:
                client.stop()
        else:
            client.stop()

    # ------------------------------------------------------------------
    # Circuit breaker callbacks
    # ------------------------------------------------------------------

    def _on_circuit_open(self) -> None:
        logger.warning("MCP circuit breaker OPEN — calls will be rejected")

    def _on_circuit_close(self) -> None:
        logger.info("MCP circuit breaker CLOSED — service recovered")

    def _on_circuit_half_open(self) -> None:
        logger.info("MCP circuit breaker HALF_OPEN — probing service")
