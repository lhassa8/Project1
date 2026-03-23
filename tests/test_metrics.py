"""Tests for agent_runner.metrics — metrics types, registry, and collector."""

from __future__ import annotations

import math
import re
import time

import pytest

from agent_runner.metrics import (
    Counter,
    Histogram,
    Gauge,
    MetricsRegistry,
    MetricsCollector,
    DEFAULT_BUCKETS,
)
from agent_runner.events import EventStream, AgentEvent, make_event


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class TestCounter:
    def test_inc_default(self):
        c = Counter("test_counter")
        c.inc()
        assert c.get() == 1.0

    def test_inc_custom_value(self):
        c = Counter("test_counter")
        c.inc(5.0)
        assert c.get() == 5.0

    def test_inc_accumulates(self):
        c = Counter("test_counter")
        c.inc(2.0)
        c.inc(3.0)
        assert c.get() == 5.0

    def test_inc_negative_raises(self):
        c = Counter("test_counter")
        with pytest.raises(ValueError, match="non-negative"):
            c.inc(-1.0)

    def test_labels(self):
        c = Counter("http_requests", label_names=("method", "status"))
        c.inc(labels={"method": "GET", "status": "200"})
        c.inc(labels={"method": "POST", "status": "201"})
        c.inc(labels={"method": "GET", "status": "200"})
        assert c.get(labels={"method": "GET", "status": "200"}) == 2.0
        assert c.get(labels={"method": "POST", "status": "201"}) == 1.0

    def test_get_nonexistent_labels_returns_zero(self):
        c = Counter("test_counter", label_names=("x",))
        assert c.get(labels={"x": "missing"}) == 0.0

    def test_collect(self):
        c = Counter("test_counter", label_names=("a",))
        c.inc(labels={"a": "1"})
        c.inc(2.0, labels={"a": "2"})
        pairs = c.collect()
        assert len(pairs) == 2

    def test_prometheus_export(self):
        c = Counter("my_counter", "A test counter", label_names=("env",))
        c.inc(labels={"env": "prod"})
        text = c.export_prometheus()
        assert "# HELP my_counter A test counter" in text
        assert "# TYPE my_counter counter" in text
        assert 'my_counter{env="prod"} 1.0' in text

    def test_prometheus_export_no_values(self):
        c = Counter("empty_counter")
        text = c.export_prometheus()
        assert "empty_counter 0" in text

    def test_to_dict(self):
        c = Counter("test_counter")
        c.inc(3.0)
        d = c.to_dict()
        assert d["name"] == "test_counter"
        assert d["type"] == "counter"
        assert d["values"][0]["value"] == 3.0


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_observe_single(self):
        h = Histogram("latency", buckets=(0.1, 0.5, 1.0))
        h.observe(0.3)
        data = h.get()
        assert data["count"] == 1
        assert data["sum"] == 0.3

    def test_observe_multiple(self):
        h = Histogram("latency", buckets=(0.1, 0.5, 1.0))
        h.observe(0.05)
        h.observe(0.3)
        h.observe(0.8)
        data = h.get()
        assert data["count"] == 3
        assert abs(data["sum"] - 1.15) < 1e-9

    def test_bucket_counts(self):
        h = Histogram("latency", buckets=(0.1, 0.5, 1.0))
        h.observe(0.05)  # fits in 0.1, 0.5, 1.0, +Inf
        h.observe(0.3)   # fits in 0.5, 1.0, +Inf
        h.observe(0.8)   # fits in 1.0, +Inf
        h.observe(2.0)   # fits in +Inf only
        data = h.get()
        buckets = data["buckets"]
        assert buckets[0.1] == 1
        assert buckets[0.5] == 2
        assert buckets[1.0] == 3
        assert buckets[float("inf")] == 4

    def test_labels(self):
        h = Histogram("latency", buckets=(1.0,), label_names=("endpoint",))
        h.observe(0.5, labels={"endpoint": "/api"})
        h.observe(0.8, labels={"endpoint": "/health"})
        api = h.get(labels={"endpoint": "/api"})
        health = h.get(labels={"endpoint": "/health"})
        assert api["count"] == 1
        assert health["count"] == 1

    def test_get_nonexistent_labels(self):
        h = Histogram("latency", label_names=("x",))
        data = h.get(labels={"x": "missing"})
        assert data["count"] == 0
        assert data["sum"] == 0.0

    def test_prometheus_export(self):
        h = Histogram("req_duration", "Request duration", buckets=(0.5, 1.0))
        h.observe(0.3)
        h.observe(0.7)
        text = h.export_prometheus()
        assert "# HELP req_duration Request duration" in text
        assert "# TYPE req_duration histogram" in text
        assert "req_duration_bucket" in text
        assert "req_duration_sum" in text
        assert "req_duration_count" in text
        # le="+Inf" should be present
        assert '+Inf' in text

    def test_to_dict(self):
        h = Histogram("latency", buckets=(1.0,))
        h.observe(0.5)
        d = h.to_dict()
        assert d["name"] == "latency"
        assert d["type"] == "histogram"
        assert len(d["values"]) == 1
        assert d["values"][0]["count"] == 1


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------


class TestGauge:
    def test_set(self):
        g = Gauge("temperature")
        g.set(42.0)
        assert g.get() == 42.0

    def test_inc(self):
        g = Gauge("connections")
        g.inc()
        g.inc()
        assert g.get() == 2.0

    def test_dec(self):
        g = Gauge("connections")
        g.set(5.0)
        g.dec(2.0)
        assert g.get() == 3.0

    def test_dec_below_zero(self):
        g = Gauge("connections")
        g.dec()
        assert g.get() == -1.0

    def test_labels(self):
        g = Gauge("pool_size", label_names=("pool",))
        g.set(10.0, labels={"pool": "main"})
        g.set(5.0, labels={"pool": "secondary"})
        assert g.get(labels={"pool": "main"}) == 10.0
        assert g.get(labels={"pool": "secondary"}) == 5.0

    def test_get_default_zero(self):
        g = Gauge("empty")
        assert g.get() == 0.0

    def test_prometheus_export(self):
        g = Gauge("active_conns", "Active connections")
        g.set(7.0)
        text = g.export_prometheus()
        assert "# HELP active_conns Active connections" in text
        assert "# TYPE active_conns gauge" in text
        assert "active_conns 7.0" in text

    def test_to_dict(self):
        g = Gauge("active")
        g.set(3.0)
        d = g.to_dict()
        assert d["type"] == "gauge"
        assert d["values"][0]["value"] == 3.0


# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


class TestMetricsRegistry:
    def test_counter_creation(self):
        r = MetricsRegistry()
        c = r.counter("test_total", "A counter")
        assert isinstance(c, Counter)

    def test_counter_idempotent(self):
        r = MetricsRegistry()
        c1 = r.counter("test_total")
        c2 = r.counter("test_total")
        assert c1 is c2

    def test_histogram_creation(self):
        r = MetricsRegistry()
        h = r.histogram("latency", buckets=(1.0, 5.0))
        assert isinstance(h, Histogram)

    def test_gauge_creation(self):
        r = MetricsRegistry()
        g = r.gauge("active")
        assert isinstance(g, Gauge)

    def test_export_prometheus(self):
        r = MetricsRegistry()
        c = r.counter("requests_total", "Total requests")
        c.inc()
        g = r.gauge("active_requests")
        g.set(3.0)
        text = r.export_prometheus()
        assert "requests_total" in text
        assert "active_requests" in text

    def test_export_json(self):
        r = MetricsRegistry()
        r.counter("c1").inc()
        r.gauge("g1").set(5)
        r.histogram("h1").observe(0.1)
        data = r.export_json()
        assert "c1" in data["counters"]
        assert "g1" in data["gauges"]
        assert "h1" in data["histograms"]


# ---------------------------------------------------------------------------
# MetricsCollector — bridging EventStream to metrics
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def _make_registry(self) -> MetricsRegistry:
        """Create a fresh registry with the standard pre-defined metrics."""
        r = MetricsRegistry()
        r.counter("agent_runs_total", labels=("status",))
        r.counter("agent_tool_calls_total", labels=("tool", "action"))
        r.histogram("agent_tool_duration_seconds", labels=("tool",))
        r.counter("agent_api_requests_total", labels=("model", "status"))
        r.histogram("agent_api_latency_seconds", labels=("model",))
        r.counter("agent_tokens_total", labels=("direction", "model"))
        r.counter("agent_cost_usd_total", labels=("model",))
        r.gauge("agent_active_runs")
        r.counter("agent_shadow_captures_total")
        r.counter("agent_errors_total", labels=("type",))
        return r

    def test_run_start_increments_active(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("run_start", "r1", 0, {}))
        assert reg.gauge("agent_active_runs").get() == 1.0

    def test_run_complete_decrements_active_and_counts(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("run_start", "r1", 0, {}))
        stream.record(make_event("run_complete", "r1", 3, {
            "status": "success",
            "cost_usd": 0.05,
            "model": "claude-3",
        }))
        assert reg.gauge("agent_active_runs").get() == 0.0
        assert reg.counter("agent_runs_total").get(labels={"status": "success"}) == 1.0
        assert reg.counter("agent_cost_usd_total").get(labels={"model": "claude-3"}) == 0.05

    def test_tool_call_counted(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("tool_call", "r1", 1, {
            "tool": "read_file",
            "action": "allow",
            "duration_ms": 150.0,
        }))
        assert reg.counter("agent_tool_calls_total").get(
            labels={"tool": "read_file", "action": "allow"}
        ) == 1.0
        hist = reg.histogram("agent_tool_duration_seconds").get(labels={"tool": "read_file"})
        assert hist["count"] == 1
        assert abs(hist["sum"] - 0.15) < 1e-9

    def test_tool_call_error_counted(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("tool_call", "r1", 1, {
            "tool": "shell",
            "action": "allow",
            "error": "command not found",
        }))
        assert reg.counter("agent_errors_total").get(labels={"type": "tool_error"}) == 1.0

    def test_tool_call_shadow_capture(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("tool_call", "r1", 1, {
            "tool": "write_file",
            "action": "capture",
            "shadow_captured": True,
        }))
        assert reg.counter("agent_shadow_captures_total").get() == 1.0

    def test_api_request_counted(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("api_request", "r1", 1, {
            "model": "claude-3",
            "status": "success",
            "latency_seconds": 1.5,
            "input_tokens": 500,
            "output_tokens": 200,
        }))
        assert reg.counter("agent_api_requests_total").get(
            labels={"model": "claude-3", "status": "success"}
        ) == 1.0
        lat = reg.histogram("agent_api_latency_seconds").get(labels={"model": "claude-3"})
        assert lat["count"] == 1
        assert abs(lat["sum"] - 1.5) < 1e-9
        assert reg.counter("agent_tokens_total").get(
            labels={"direction": "input", "model": "claude-3"}
        ) == 500.0
        assert reg.counter("agent_tokens_total").get(
            labels={"direction": "output", "model": "claude-3"}
        ) == 200.0

    def test_error_event_counted(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("error", "r1", 2, {
            "error_type": "rate_limit",
        }))
        assert reg.counter("agent_errors_total").get(labels={"type": "rate_limit"}) == 1.0

    def test_unknown_event_does_not_crash(self):
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("some_future_event", "r1", 0, {"foo": "bar"}))
        # Should not raise

    def test_full_run_lifecycle(self):
        """Simulate a complete agent run and verify final metric state."""
        stream = EventStream()
        reg = self._make_registry()
        MetricsCollector(stream, reg)

        stream.record(make_event("run_start", "r1", 0, {}))
        stream.record(make_event("api_request", "r1", 1, {
            "model": "claude-3", "status": "success",
            "input_tokens": 100, "output_tokens": 50, "latency_seconds": 0.8,
        }))
        stream.record(make_event("tool_call", "r1", 1, {
            "tool": "read_file", "action": "allow", "duration_ms": 10.0,
        }))
        stream.record(make_event("api_request", "r1", 2, {
            "model": "claude-3", "status": "success",
            "input_tokens": 200, "output_tokens": 100, "latency_seconds": 1.2,
        }))
        stream.record(make_event("run_complete", "r1", 2, {
            "status": "success", "cost_usd": 0.01, "model": "claude-3",
        }))

        assert reg.gauge("agent_active_runs").get() == 0.0
        assert reg.counter("agent_runs_total").get(labels={"status": "success"}) == 1.0
        assert reg.counter("agent_tool_calls_total").get(
            labels={"tool": "read_file", "action": "allow"}
        ) == 1.0
        assert reg.counter("agent_api_requests_total").get(
            labels={"model": "claude-3", "status": "success"}
        ) == 2.0
        assert reg.counter("agent_tokens_total").get(
            labels={"direction": "input", "model": "claude-3"}
        ) == 300.0
        assert reg.counter("agent_tokens_total").get(
            labels={"direction": "output", "model": "claude-3"}
        ) == 150.0


# ---------------------------------------------------------------------------
# Prometheus format validation
# ---------------------------------------------------------------------------


class TestPrometheusFormat:
    def test_counter_format_matches_spec(self):
        r = MetricsRegistry()
        c = r.counter("http_requests_total", "Total HTTP requests", labels=("method",))
        c.inc(labels={"method": "GET"})
        text = r.export_prometheus()
        # Must contain HELP, TYPE, and a sample line
        assert re.search(r"^# HELP http_requests_total", text, re.MULTILINE)
        assert re.search(r"^# TYPE http_requests_total counter", text, re.MULTILINE)
        assert re.search(r'^http_requests_total\{method="GET"\} 1\.0', text, re.MULTILINE)

    def test_histogram_format_has_bucket_sum_count(self):
        r = MetricsRegistry()
        h = r.histogram("duration_seconds", buckets=(0.5, 1.0))
        h.observe(0.3)
        text = r.export_prometheus()
        assert "duration_seconds_bucket" in text
        assert "duration_seconds_sum" in text
        assert "duration_seconds_count" in text

    def test_gauge_format(self):
        r = MetricsRegistry()
        g = r.gauge("temperature", "Current temperature")
        g.set(36.6)
        text = r.export_prometheus()
        assert "# TYPE temperature gauge" in text
        assert "temperature 36.6" in text
