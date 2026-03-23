"""Health check and readiness system for the agent runner.

Provides liveness, readiness, and detailed health reporting for
deployment environments (Kubernetes, Docker, load balancers).

Usage::

    from agent_runner.health import HealthCheck, HealthEndpoint

    hc = HealthCheck()
    hc.register("database", check_db, critical=True)
    hc.register("cache", check_cache, critical=False)

    report = await hc.check_all()
    # GET /health, /health/live, /health/ready
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable, Awaitable, Union

logger = logging.getLogger(__name__)

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class HealthStatus(Enum):
    """Overall or per-component health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Result of a single component health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    last_checked: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 3),
            "last_checked": self.last_checked,
        }


# Type alias for health check callables.
# They can be sync (returning ComponentHealth) or async.
HealthCheckFn = Union[
    Callable[[], ComponentHealth],
    Callable[[], Awaitable[ComponentHealth]],
]


# ---------------------------------------------------------------------------
# HealthCheck registry
# ---------------------------------------------------------------------------


class HealthCheck:
    """Registry of health-check functions with critical / advisory semantics.

    Parameters
    ----------
    version : str
        Application version string reported in health payloads.
    """

    def __init__(self, version: str = __version__) -> None:
        self._checks: dict[str, tuple[HealthCheckFn, bool]] = {}
        self._version = version
        self._start_time = time.time()
        self._shutting_down = False
        self._active_runs = 0

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        check_fn: HealthCheckFn,
        critical: bool = True,
    ) -> None:
        """Register a health-check function for *name*.

        Parameters
        ----------
        name:
            Unique component identifier (e.g. ``"mcp"``, ``"database"``).
        check_fn:
            Callable that returns a :class:`ComponentHealth`.  May be sync
            or async.
        critical:
            If ``True`` an unhealthy result makes the overall status
            unhealthy.  Advisory (``False``) components can be degraded
            without affecting readiness.
        """
        self._checks[name] = (check_fn, critical)

    def unregister(self, name: str) -> None:
        """Remove a previously registered check."""
        self._checks.pop(name, None)

    # -- active run tracking ------------------------------------------------

    def set_active_runs(self, count: int) -> None:
        self._active_runs = count

    def inc_active_runs(self) -> None:
        self._active_runs += 1

    def dec_active_runs(self) -> None:
        self._active_runs = max(0, self._active_runs - 1)

    # -- shutdown -----------------------------------------------------------

    def begin_shutdown(self) -> None:
        """Signal that the process is shutting down (liveness returns False)."""
        self._shutting_down = True

    # -- core checks --------------------------------------------------------

    def check_all(self) -> dict[str, Any]:
        """Run every registered check and return a full health report.

        Returns a dict suitable for JSON serialisation::

            {
                "status": "healthy",
                "version": "0.2.0",
                "uptime_seconds": 1234.5,
                "active_runs": 2,
                "components": { ... }
            }

        Async check functions are executed in a new event loop if no loop
        is running, otherwise they are awaited directly.
        """
        components: dict[str, dict[str, Any]] = {}
        overall = HealthStatus.HEALTHY

        for name, (fn, critical) in self._checks.items():
            start = time.time()
            try:
                result = self._invoke(fn)
                result.latency_ms = (time.time() - start) * 1000
                result.last_checked = time.time()
            except Exception as exc:
                result = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check raised: {exc}",
                    latency_ms=(time.time() - start) * 1000,
                    last_checked=time.time(),
                )

            components[name] = result.to_dict()

            # Roll up overall status
            if result.status == HealthStatus.UNHEALTHY and critical:
                overall = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall != HealthStatus.UNHEALTHY:
                overall = HealthStatus.DEGRADED
            elif result.status == HealthStatus.UNHEALTHY and not critical:
                if overall == HealthStatus.HEALTHY:
                    overall = HealthStatus.DEGRADED

        return {
            "status": overall.value,
            "version": self._version,
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "active_runs": self._active_runs,
            "components": components,
        }

    def is_ready(self) -> bool:
        """Return ``True`` if all **critical** components are healthy."""
        for name, (fn, critical) in self._checks.items():
            if not critical:
                continue
            try:
                result = self._invoke(fn)
            except Exception:
                return False
            if result.status == HealthStatus.UNHEALTHY:
                return False
        return True

    def is_live(self) -> bool:
        """Return ``True`` if the process is alive (not shutting down)."""
        return not self._shutting_down

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _invoke(fn: HealthCheckFn) -> ComponentHealth:
        """Invoke a sync or async check function and return the result."""
        result = fn()
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # We're already in an async context — can't use run().
                # Create a task; but for simplicity in health checks
                # we use a new loop in a thread.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, result)
                    return future.result(timeout=10)
            else:
                return asyncio.run(result)
        return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# HTTP handler mixin
# ---------------------------------------------------------------------------


class HealthEndpoint:
    """Mixin / helper that adds health endpoints to an ``HTTPServer`` handler.

    Attach to your existing ``BaseHTTPRequestHandler`` by calling
    :meth:`try_handle` from ``do_GET``::

        class MyHandler(BaseHTTPRequestHandler):
            health_endpoint = HealthEndpoint(health_check)

            def do_GET(self):
                if self.health_endpoint.try_handle(self):
                    return
                # ... other routes ...
    """

    def __init__(self, health_check: HealthCheck) -> None:
        self._hc = health_check

    def try_handle(self, handler: BaseHTTPRequestHandler) -> bool:
        """Attempt to handle the request.  Returns ``True`` if handled."""
        from urllib.parse import urlparse

        path = urlparse(handler.path).path.rstrip("/")

        if path == "/health":
            report = self._hc.check_all()
            status_code = 200 if report["status"] != "unhealthy" else 503
            self._respond_json(handler, report, status_code)
            return True

        if path == "/health/live":
            if self._hc.is_live():
                self._respond_json(handler, {"status": "alive"}, 200)
            else:
                self._respond_json(handler, {"status": "shutting_down"}, 503)
            return True

        if path == "/health/ready":
            if self._hc.is_ready():
                self._respond_json(handler, {"status": "ready"}, 200)
            else:
                self._respond_json(handler, {"status": "not_ready"}, 503)
            return True

        return False

    @staticmethod
    def _respond_json(
        handler: BaseHTTPRequestHandler,
        data: Any,
        status: int = 200,
    ) -> None:
        body = json.dumps(data, indent=2, default=str).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
