"""OpenTelemetry-compatible metrics with Prometheus exposition — no external deps.

Provides Counter, Histogram, and Gauge metric types plus a registry that
can export in Prometheus text format or JSON.  A :class:`MetricsCollector`
bridges :class:`~agent_runner.events.EventStream` events to metrics so the
runner needs no manual instrumentation.

Usage::

    from agent_runner.metrics import get_registry, MetricsCollector
    from agent_runner.events import EventStream

    stream = EventStream()
    registry = get_registry()
    collector = MetricsCollector(stream, registry)
    # metrics are now automatically updated as events flow

    # Prometheus scrape endpoint
    print(registry.export_prometheus())
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from typing import Any, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default histogram buckets (seconds), matching OTel SDK defaults.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, float("inf"),
)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _labels_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Return a hashable, sorted representation of label pairs."""
    return tuple(sorted(labels.items()))


def _labels_to_prometheus(labels: dict[str, str]) -> str:
    """Format labels as ``{key="value",...}``."""
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{v}"' for k, v in sorted(labels.items())
    )
    return "{" + inner + "}"


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------


class Counter:
    """Monotonically increasing counter, optionally labelled."""

    def __init__(self, name: str, description: str = "", label_names: Sequence[str] = ()) -> None:
        self.name = name
        self.description = description
        self.label_names = tuple(label_names)
        self._lock = threading.Lock()
        # _values: label_key -> float
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        """Increment the counter."""
        if value < 0:
            raise ValueError("Counter.inc value must be non-negative")
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def get(self, labels: dict[str, str] | None = None) -> float:
        """Return the current value for the given label set."""
        key = _labels_key(labels or {})
        with self._lock:
            return self._values.get(key, 0.0)

    def collect(self) -> list[tuple[dict[str, str], float]]:
        """Return all (labels, value) pairs."""
        with self._lock:
            return [(dict(k), v) for k, v in self._values.items()]

    def export_prometheus(self) -> str:
        lines: list[str] = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} counter")
        with self._lock:
            for lk, val in sorted(self._values.items()):
                lbl = _labels_to_prometheus(dict(lk))
                lines.append(f"{self.name}{lbl} {val}")
        # If no values recorded yet, emit a zero line.
        if not self._values:
            lines.append(f"{self.name} 0")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "counter",
            "description": self.description,
            "values": [{"labels": dict(k), "value": v} for k, v in sorted(self._values.items())],
        }


class Histogram:
    """Cumulative histogram with configurable buckets."""

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Sequence[float] = DEFAULT_BUCKETS,
        label_names: Sequence[str] = (),
    ) -> None:
        self.name = name
        self.description = description
        self.label_names = tuple(label_names)
        self._buckets = tuple(sorted(set(buckets) | {float("inf")}))
        self._lock = threading.Lock()
        # Per label-set: (bucket_counts, sum, count)
        self._data: dict[
            tuple[tuple[str, str], ...],
            tuple[dict[float, int], float, int],
        ] = {}

    def _ensure(self, key: tuple[tuple[str, str], ...]) -> tuple[dict[float, int], float, int]:
        if key not in self._data:
            self._data[key] = ({b: 0 for b in self._buckets}, 0.0, 0)
        return self._data[key]

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        """Record an observation."""
        key = _labels_key(labels or {})
        with self._lock:
            buckets, total, count = self._ensure(key)
            for b in self._buckets:
                if value <= b:
                    buckets[b] += 1
            self._data[key] = (buckets, total + value, count + 1)

    def get(self, labels: dict[str, str] | None = None) -> dict[str, Any]:
        """Return sum, count, buckets for the given label set."""
        key = _labels_key(labels or {})
        with self._lock:
            if key not in self._data:
                return {"sum": 0.0, "count": 0, "buckets": {}}
            buckets, total, count = self._data[key]
            return {"sum": total, "count": count, "buckets": dict(buckets)}

    def export_prometheus(self) -> str:
        lines: list[str] = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} histogram")
        with self._lock:
            for lk, (buckets, total, count) in sorted(self._data.items()):
                lbl_dict = dict(lk)
                for bound in sorted(buckets):
                    le_val = "+Inf" if math.isinf(bound) else str(bound)
                    merged = {**lbl_dict, "le": le_val}
                    lines.append(f"{self.name}_bucket{_labels_to_prometheus(merged)} {buckets[bound]}")
                lbl = _labels_to_prometheus(lbl_dict)
                lines.append(f"{self.name}_sum{lbl} {total}")
                lines.append(f"{self.name}_count{lbl} {count}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        result: list[dict[str, Any]] = []
        with self._lock:
            for lk, (buckets, total, count) in sorted(self._data.items()):
                result.append({
                    "labels": dict(lk),
                    "sum": total,
                    "count": count,
                    "buckets": {
                        ("+Inf" if math.isinf(b) else b): v
                        for b, v in sorted(buckets.items())
                    },
                })
        return {
            "name": self.name,
            "type": "histogram",
            "description": self.description,
            "values": result,
        }


class Gauge:
    """A value that can go up and down."""

    def __init__(self, name: str, description: str = "", label_names: Sequence[str] = ()) -> None:
        self.name = name
        self.description = description
        self.label_names = tuple(label_names)
        self._lock = threading.Lock()
        self._values: dict[tuple[tuple[str, str], ...], float] = {}

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] = value

    def inc(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = _labels_key(labels or {})
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) - value

    def get(self, labels: dict[str, str] | None = None) -> float:
        key = _labels_key(labels or {})
        with self._lock:
            return self._values.get(key, 0.0)

    def export_prometheus(self) -> str:
        lines: list[str] = []
        if self.description:
            lines.append(f"# HELP {self.name} {self.description}")
        lines.append(f"# TYPE {self.name} gauge")
        with self._lock:
            for lk, val in sorted(self._values.items()):
                lbl = _labels_to_prometheus(dict(lk))
                lines.append(f"{self.name}{lbl} {val}")
        if not self._values:
            lines.append(f"{self.name} 0")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "gauge",
            "description": self.description,
            "values": [{"labels": dict(k), "value": v} for k, v in sorted(self._values.items())],
        }


# ---------------------------------------------------------------------------
# MetricsRegistry
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """Central registry for all metrics.  Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._gauges: dict[str, Gauge] = {}

    def counter(
        self,
        name: str,
        description: str = "",
        labels: Sequence[str] = (),
    ) -> Counter:
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description, labels)
            return self._counters[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Sequence[float] = DEFAULT_BUCKETS,
        labels: Sequence[str] = (),
    ) -> Histogram:
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, buckets, labels)
            return self._histograms[name]

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: Sequence[str] = (),
    ) -> Gauge:
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, labels)
            return self._gauges[name]

    def export_prometheus(self) -> str:
        """Return all metrics in Prometheus text exposition format."""
        sections: list[str] = []
        with self._lock:
            for c in self._counters.values():
                sections.append(c.export_prometheus())
            for h in self._histograms.values():
                sections.append(h.export_prometheus())
            for g in self._gauges.values():
                sections.append(g.export_prometheus())
        return "\n\n".join(sections) + "\n"

    def export_json(self) -> dict[str, Any]:
        """Return all metrics as a JSON-serialisable dict."""
        with self._lock:
            return {
                "counters": {n: c.to_dict() for n, c in self._counters.items()},
                "histograms": {n: h.to_dict() for n, h in self._histograms.items()},
                "gauges": {n: g.to_dict() for n, g in self._gauges.items()},
            }


# ---------------------------------------------------------------------------
# Metrics HTTP endpoint
# ---------------------------------------------------------------------------


class MetricsEndpoint:
    """Handler mixin that serves ``/metrics`` in Prometheus format."""

    def __init__(self, registry: MetricsRegistry) -> None:
        self._registry = registry

    def try_handle(self, handler: BaseHTTPRequestHandler) -> bool:
        path = urlparse(handler.path).path.rstrip("/")
        if path == "/metrics":
            body = self._registry.export_prometheus().encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
            return True
        if path == "/metrics/json":
            data = json.dumps(self._registry.export_json(), indent=2, default=str).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
            return True
        return False


# ---------------------------------------------------------------------------
# Pre-defined metrics (singleton registry)
# ---------------------------------------------------------------------------

_default_registry: MetricsRegistry | None = None


def get_registry() -> MetricsRegistry:
    """Return the global default :class:`MetricsRegistry`, creating it on
    first call together with the pre-defined agent metrics."""
    global _default_registry
    if _default_registry is None:
        _default_registry = _create_default_registry()
    return _default_registry


def _create_default_registry() -> MetricsRegistry:
    r = MetricsRegistry()

    r.counter("agent_runs_total", "Total agent runs", labels=("status",))
    r.counter("agent_tool_calls_total", "Total tool calls", labels=("tool", "action"))
    r.histogram(
        "agent_tool_duration_seconds",
        "Tool call duration in seconds",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf")),
        labels=("tool",),
    )
    r.counter("agent_api_requests_total", "Total API requests", labels=("model", "status"))
    r.histogram(
        "agent_api_latency_seconds",
        "API request latency in seconds",
        labels=("model",),
    )
    r.counter("agent_tokens_total", "Total tokens consumed", labels=("direction", "model"))
    r.counter("agent_cost_usd_total", "Estimated cost in USD", labels=("model",))
    r.gauge("agent_active_runs", "Currently active agent runs")
    r.counter("agent_shadow_captures_total", "Shadow-captured writes")
    r.counter("agent_errors_total", "Total errors", labels=("type",))

    return r


# ---------------------------------------------------------------------------
# MetricsCollector — bridges EventStream -> metrics
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Subscribes to an :class:`~agent_runner.events.EventStream` and
    automatically updates the pre-defined metrics.

    Parameters
    ----------
    event_stream:
        The stream to listen to.
    registry:
        Metrics registry (defaults to the global singleton).
    """

    def __init__(
        self,
        event_stream: Any,  # EventStream, but kept as Any to avoid circular imports
        registry: MetricsRegistry | None = None,
    ) -> None:
        self._registry = registry or get_registry()
        self._stream = event_stream
        self._stream.on_event(self._handle_event)

    def _handle_event(self, event: Any) -> None:
        """Route an AgentEvent to the appropriate metric."""
        try:
            etype = event.event_type
            data = event.data

            if etype == "run_start":
                self._registry.gauge("agent_active_runs").inc()

            elif etype == "run_complete":
                self._registry.gauge("agent_active_runs").dec()
                status = data.get("status", "success")
                self._registry.counter("agent_runs_total").inc(labels={"status": status})
                # Cost
                cost = data.get("cost_usd", 0.0)
                model = data.get("model", "unknown")
                if cost:
                    self._registry.counter("agent_cost_usd_total").inc(cost, labels={"model": model})

            elif etype == "tool_call":
                tool = data.get("tool", "unknown")
                action = data.get("action", "allow")
                self._registry.counter("agent_tool_calls_total").inc(
                    labels={"tool": tool, "action": action},
                )
                duration_ms = data.get("duration_ms", 0.0)
                if duration_ms:
                    self._registry.histogram("agent_tool_duration_seconds").observe(
                        duration_ms / 1000.0, labels={"tool": tool},
                    )
                if data.get("error"):
                    self._registry.counter("agent_errors_total").inc(labels={"type": "tool_error"})
                if data.get("shadow_captured"):
                    self._registry.counter("agent_shadow_captures_total").inc()

            elif etype == "api_request":
                model = data.get("model", "unknown")
                status = data.get("status", "success")
                self._registry.counter("agent_api_requests_total").inc(
                    labels={"model": model, "status": status},
                )
                latency = data.get("latency_seconds", 0.0)
                if latency:
                    self._registry.histogram("agent_api_latency_seconds").observe(
                        latency, labels={"model": model},
                    )
                input_tokens = data.get("input_tokens", 0)
                output_tokens = data.get("output_tokens", 0)
                if input_tokens:
                    self._registry.counter("agent_tokens_total").inc(
                        input_tokens, labels={"direction": "input", "model": model},
                    )
                if output_tokens:
                    self._registry.counter("agent_tokens_total").inc(
                        output_tokens, labels={"direction": "output", "model": model},
                    )

            elif etype == "error":
                error_type = data.get("error_type", "unknown")
                self._registry.counter("agent_errors_total").inc(labels={"type": error_type})

        except Exception:
            logger.debug("MetricsCollector: failed to handle event %s", event, exc_info=True)
