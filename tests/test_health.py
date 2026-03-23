"""Tests for agent_runner.health — health check and readiness system."""

from __future__ import annotations

import io
import json
import time
from http.server import BaseHTTPRequestHandler
from unittest.mock import MagicMock

import pytest

from agent_runner.health import (
    HealthStatus,
    ComponentHealth,
    HealthCheck,
    HealthEndpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _healthy_check() -> ComponentHealth:
    return ComponentHealth(name="db", status=HealthStatus.HEALTHY, message="ok")


def _degraded_check() -> ComponentHealth:
    return ComponentHealth(name="cache", status=HealthStatus.DEGRADED, message="slow")


def _unhealthy_check() -> ComponentHealth:
    return ComponentHealth(name="mcp", status=HealthStatus.UNHEALTHY, message="down")


def _exploding_check() -> ComponentHealth:
    raise RuntimeError("connection refused")


# ---------------------------------------------------------------------------
# HealthStatus enum
# ---------------------------------------------------------------------------


class TestHealthStatus:
    def test_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


# ---------------------------------------------------------------------------
# ComponentHealth dataclass
# ---------------------------------------------------------------------------


class TestComponentHealth:
    def test_to_dict(self):
        ch = ComponentHealth(
            name="db",
            status=HealthStatus.HEALTHY,
            message="connected",
            latency_ms=1.234,
        )
        d = ch.to_dict()
        assert d["name"] == "db"
        assert d["status"] == "healthy"
        assert d["message"] == "connected"
        assert d["latency_ms"] == 1.234

    def test_default_last_checked(self):
        before = time.time()
        ch = ComponentHealth(name="x", status=HealthStatus.HEALTHY)
        after = time.time()
        assert before <= ch.last_checked <= after


# ---------------------------------------------------------------------------
# HealthCheck — registration & execution
# ---------------------------------------------------------------------------


class TestHealthCheckRegistration:
    def test_register_and_check_all_healthy(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check, critical=True)
        report = hc.check_all()
        assert report["status"] == "healthy"
        assert "db" in report["components"]
        assert report["components"]["db"]["status"] == "healthy"

    def test_register_multiple(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check)
        hc.register("cache", _degraded_check)
        report = hc.check_all()
        assert len(report["components"]) == 2

    def test_unregister(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check)
        hc.unregister("db")
        report = hc.check_all()
        assert len(report["components"]) == 0


# ---------------------------------------------------------------------------
# HealthCheck — critical vs advisory
# ---------------------------------------------------------------------------


class TestCriticalVsAdvisory:
    def test_critical_unhealthy_makes_overall_unhealthy(self):
        hc = HealthCheck()
        hc.register("db", _unhealthy_check, critical=True)
        report = hc.check_all()
        assert report["status"] == "unhealthy"

    def test_advisory_unhealthy_makes_overall_degraded(self):
        hc = HealthCheck()
        hc.register("cache", _unhealthy_check, critical=False)
        report = hc.check_all()
        assert report["status"] == "degraded"

    def test_advisory_degraded_does_not_override_healthy_to_unhealthy(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check, critical=True)
        hc.register("cache", _degraded_check, critical=False)
        report = hc.check_all()
        assert report["status"] == "degraded"

    def test_critical_healthy_advisory_unhealthy(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check, critical=True)
        hc.register("cache", _unhealthy_check, critical=False)
        report = hc.check_all()
        assert report["status"] == "degraded"

    def test_exception_in_check_becomes_unhealthy(self):
        hc = HealthCheck()
        hc.register("flaky", _exploding_check, critical=True)
        report = hc.check_all()
        assert report["status"] == "unhealthy"
        assert "connection refused" in report["components"]["flaky"]["message"]


# ---------------------------------------------------------------------------
# HealthCheck — is_ready / is_live
# ---------------------------------------------------------------------------


class TestReadyLive:
    def test_is_ready_all_critical_healthy(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check, critical=True)
        hc.register("cache", _unhealthy_check, critical=False)
        assert hc.is_ready() is True

    def test_is_ready_critical_unhealthy(self):
        hc = HealthCheck()
        hc.register("db", _unhealthy_check, critical=True)
        assert hc.is_ready() is False

    def test_is_ready_critical_exception(self):
        hc = HealthCheck()
        hc.register("db", _exploding_check, critical=True)
        assert hc.is_ready() is False

    def test_is_ready_no_checks(self):
        hc = HealthCheck()
        assert hc.is_ready() is True

    def test_is_live_default(self):
        hc = HealthCheck()
        assert hc.is_live() is True

    def test_is_live_after_shutdown(self):
        hc = HealthCheck()
        hc.begin_shutdown()
        assert hc.is_live() is False


# ---------------------------------------------------------------------------
# HealthCheck — metadata in report
# ---------------------------------------------------------------------------


class TestReportMetadata:
    def test_version_in_report(self):
        hc = HealthCheck(version="1.2.3")
        report = hc.check_all()
        assert report["version"] == "1.2.3"

    def test_uptime_in_report(self):
        hc = HealthCheck()
        report = hc.check_all()
        assert report["uptime_seconds"] >= 0

    def test_active_runs_in_report(self):
        hc = HealthCheck()
        hc.set_active_runs(5)
        report = hc.check_all()
        assert report["active_runs"] == 5

    def test_inc_dec_active_runs(self):
        hc = HealthCheck()
        hc.inc_active_runs()
        hc.inc_active_runs()
        hc.dec_active_runs()
        report = hc.check_all()
        assert report["active_runs"] == 1

    def test_dec_does_not_go_negative(self):
        hc = HealthCheck()
        hc.dec_active_runs()
        report = hc.check_all()
        assert report["active_runs"] == 0


# ---------------------------------------------------------------------------
# HealthCheck — async check functions
# ---------------------------------------------------------------------------


class TestAsyncChecks:
    def test_async_check_function(self):
        async def async_healthy() -> ComponentHealth:
            return ComponentHealth(name="async_db", status=HealthStatus.HEALTHY, message="ok")

        hc = HealthCheck()
        hc.register("async_db", async_healthy, critical=True)
        report = hc.check_all()
        assert report["components"]["async_db"]["status"] == "healthy"


# ---------------------------------------------------------------------------
# HealthEndpoint — HTTP handler mixin
# ---------------------------------------------------------------------------


class _FakeHandler:
    """Minimal stand-in for BaseHTTPRequestHandler for testing."""

    def __init__(self, path: str):
        self.path = path
        self._status: int | None = None
        self._headers: dict[str, str] = {}
        self._body = b""
        self.wfile = io.BytesIO()

    def send_response(self, code: int) -> None:
        self._status = code

    def send_header(self, key: str, value: str) -> None:
        self._headers[key] = value

    def end_headers(self) -> None:
        pass

    @property
    def response_json(self) -> dict:
        return json.loads(self.wfile.getvalue().decode())


class TestHealthEndpoint:
    def test_health_route(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check)
        ep = HealthEndpoint(hc)

        handler = _FakeHandler("/health")
        assert ep.try_handle(handler) is True
        assert handler._status == 200
        body = handler.response_json
        assert body["status"] == "healthy"

    def test_health_unhealthy_returns_503(self):
        hc = HealthCheck()
        hc.register("db", _unhealthy_check, critical=True)
        ep = HealthEndpoint(hc)

        handler = _FakeHandler("/health")
        ep.try_handle(handler)
        assert handler._status == 503

    def test_live_route(self):
        hc = HealthCheck()
        ep = HealthEndpoint(hc)

        handler = _FakeHandler("/health/live")
        assert ep.try_handle(handler) is True
        assert handler._status == 200
        assert handler.response_json["status"] == "alive"

    def test_live_shutting_down(self):
        hc = HealthCheck()
        hc.begin_shutdown()
        ep = HealthEndpoint(hc)

        handler = _FakeHandler("/health/live")
        ep.try_handle(handler)
        assert handler._status == 503

    def test_ready_route(self):
        hc = HealthCheck()
        hc.register("db", _healthy_check, critical=True)
        ep = HealthEndpoint(hc)

        handler = _FakeHandler("/health/ready")
        assert ep.try_handle(handler) is True
        assert handler._status == 200

    def test_ready_not_ready(self):
        hc = HealthCheck()
        hc.register("db", _unhealthy_check, critical=True)
        ep = HealthEndpoint(hc)

        handler = _FakeHandler("/health/ready")
        ep.try_handle(handler)
        assert handler._status == 503

    def test_unhandled_route(self):
        hc = HealthCheck()
        ep = HealthEndpoint(hc)

        handler = _FakeHandler("/other")
        assert ep.try_handle(handler) is False
