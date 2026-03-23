"""Structured event stream for observability during agent runs.

Collects and emits structured events so callers can monitor, log, and
analyse agent behaviour without modifying the core runner logic.

Usage::

    from agent_runner.events import EventStream

    stream = EventStream()
    stream.on_event(lambda e: print(e.event_type))

    runner = AgentRunner(..., event_stream=stream)
    result = runner.run("Hello")

    print(stream.summary())
    print(stream.to_jsonl())
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class AgentEvent:
    """A single structured event emitted during an agent run."""

    timestamp: float
    event_type: str  # "run_start", "turn_start", "tool_call", "api_request", "error", "run_complete"
    run_id: str
    turn: int
    data: dict[str, Any]


class EventStream:
    """Collect and emit structured events during agent runs."""

    def __init__(self) -> None:
        self._events: list[AgentEvent] = []
        self._handlers: list[Callable[[AgentEvent], None]] = []

    def record(self, event: AgentEvent) -> None:
        """Record an event and notify handlers."""
        self._events.append(event)
        for handler in self._handlers:
            handler(event)

    def on_event(self, handler: Callable[[AgentEvent], None]) -> None:
        """Register a handler for events."""
        self._handlers.append(handler)

    @property
    def events(self) -> list[AgentEvent]:
        """Return a copy of the recorded events."""
        return list(self._events)

    def to_jsonl(self) -> str:
        """Export events as JSON-lines."""
        lines: list[str] = []
        for event in self._events:
            lines.append(json.dumps(asdict(event), default=str))
        return "\n".join(lines)

    def summary(self) -> dict:
        """Return a summary: total tokens, cost, duration, tool stats."""
        total_input_tokens = 0
        total_output_tokens = 0
        total_duration = 0.0
        cost_usd = 0.0
        tool_calls = 0
        errors = 0

        for event in self._events:
            if event.event_type == "api_request":
                total_input_tokens += event.data.get("input_tokens", 0)
                total_output_tokens += event.data.get("output_tokens", 0)
            elif event.event_type == "tool_call":
                tool_calls += 1
                if event.data.get("error"):
                    errors += 1
            elif event.event_type == "error":
                errors += 1
            elif event.event_type == "run_complete":
                total_duration += event.data.get("elapsed_seconds", 0.0)
                cost_usd += event.data.get("cost_usd", 0.0)

        return {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "cost_usd": cost_usd,
            "duration_seconds": total_duration,
            "tool_calls": tool_calls,
            "errors": errors,
        }

    def tool_stats(self) -> dict[str, dict]:
        """Per-tool statistics: call count, total duration, error count.

        Returns: {"tool_name": {"calls": 5, "errors": 1, "total_ms": 234.5}}
        """
        stats: dict[str, dict[str, Any]] = {}
        for event in self._events:
            if event.event_type != "tool_call":
                continue
            name = event.data.get("tool", "unknown")
            if name not in stats:
                stats[name] = {"calls": 0, "errors": 0, "total_ms": 0.0}
            stats[name]["calls"] += 1
            stats[name]["total_ms"] += event.data.get("duration_ms", 0.0)
            if event.data.get("error"):
                stats[name]["errors"] += 1
        return stats


def make_event(
    event_type: str, run_id: str, turn: int, data: dict[str, Any],
) -> AgentEvent:
    """Convenience factory for creating events with auto-timestamp."""
    return AgentEvent(
        timestamp=time.time(),
        event_type=event_type,
        run_id=run_id,
        turn=turn,
        data=data,
    )
