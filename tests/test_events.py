"""Tests for Phase 6: Observability (EventStream).

Covers event recording, handlers, JSONL export, summary, tool stats,
and integration with the runner.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_runner.events import AgentEvent, EventStream, make_event
from agent_runner.runner import AgentRunner
from agent_runner.tools.registry import ToolRegistry


# ---- helpers ----

def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, input_: dict, id_: str = "tool_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=id_)


def _api_response(content, stop_reason="end_turn", input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_runner(**kwargs):
    registry = ToolRegistry()
    registry.register(
        "echo", "Echo", {"type": "object", "properties": {}},
        lambda p: f"echoed: {p.get('msg', '')}",
    )
    defaults = dict(
        system_prompt="Test",
        tools=registry,
        api_key="fake-key",
    )
    defaults.update(kwargs)
    return AgentRunner(**defaults)


# ---- EventStream unit tests ----

class TestEventStreamRecords:
    def test_records_events(self):
        stream = EventStream()
        event = make_event("run_start", "abc123", 0, {"prompt": "hi"})
        stream.record(event)

        assert len(stream.events) == 1
        assert stream.events[0].event_type == "run_start"
        assert stream.events[0].run_id == "abc123"
        assert stream.events[0].data["prompt"] == "hi"

    def test_records_multiple_events(self):
        stream = EventStream()
        for i in range(5):
            stream.record(make_event("turn_start", "run1", i, {}))
        assert len(stream.events) == 5


class TestEventStreamHandler:
    def test_handler_called(self):
        stream = EventStream()
        received = []
        stream.on_event(lambda e: received.append(e))

        event = make_event("run_start", "abc", 0, {})
        stream.record(event)

        assert len(received) == 1
        assert received[0] is event

    def test_multiple_handlers(self):
        stream = EventStream()
        counts = {"a": 0, "b": 0}
        stream.on_event(lambda e: counts.__setitem__("a", counts["a"] + 1))
        stream.on_event(lambda e: counts.__setitem__("b", counts["b"] + 1))

        stream.record(make_event("test", "x", 0, {}))
        assert counts["a"] == 1
        assert counts["b"] == 1


class TestEventStreamJsonl:
    def test_to_jsonl(self):
        stream = EventStream()
        stream.record(make_event("run_start", "r1", 0, {"prompt": "hello"}))
        stream.record(make_event("turn_start", "r1", 1, {}))

        jsonl = stream.to_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2

        parsed_0 = json.loads(lines[0])
        assert parsed_0["event_type"] == "run_start"
        assert parsed_0["data"]["prompt"] == "hello"

        parsed_1 = json.loads(lines[1])
        assert parsed_1["event_type"] == "turn_start"

    def test_to_jsonl_empty(self):
        stream = EventStream()
        assert stream.to_jsonl() == ""


class TestEventStreamSummary:
    def test_summary(self):
        stream = EventStream()
        stream.record(make_event("api_request", "r1", 1, {"input_tokens": 100, "output_tokens": 50}))
        stream.record(make_event("api_request", "r1", 2, {"input_tokens": 200, "output_tokens": 80}))
        stream.record(make_event("tool_call", "r1", 1, {"tool": "echo", "duration_ms": 5.0, "action": "allow"}))
        stream.record(make_event("run_complete", "r1", 2, {
            "elapsed_seconds": 1.5,
            "cost_usd": 0.002,
            "total_input_tokens": 300,
            "total_output_tokens": 130,
        }))

        s = stream.summary()
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 130
        assert s["total_tokens"] == 430
        assert s["cost_usd"] == 0.002
        assert s["duration_seconds"] == 1.5
        assert s["tool_calls"] == 1
        assert s["errors"] == 0

    def test_summary_with_errors(self):
        stream = EventStream()
        stream.record(make_event("tool_call", "r1", 1, {
            "tool": "bad", "duration_ms": 1.0, "action": "allow", "error": "boom",
        }))
        stream.record(make_event("error", "r1", 1, {"tool": "bad", "error": "boom"}))

        s = stream.summary()
        # tool_call with error counts as tool_call, error event also counted
        assert s["tool_calls"] == 1
        assert s["errors"] >= 1


class TestEventStreamToolStats:
    def test_tool_stats(self):
        stream = EventStream()
        stream.record(make_event("tool_call", "r1", 1, {
            "tool": "echo", "duration_ms": 10.0, "action": "allow",
        }))
        stream.record(make_event("tool_call", "r1", 1, {
            "tool": "echo", "duration_ms": 5.0, "action": "allow",
        }))
        stream.record(make_event("tool_call", "r1", 2, {
            "tool": "shell", "duration_ms": 100.0, "action": "allow", "error": "fail",
        }))

        stats = stream.tool_stats()
        assert "echo" in stats
        assert stats["echo"]["calls"] == 2
        assert stats["echo"]["total_ms"] == 15.0
        assert stats["echo"]["errors"] == 0

        assert "shell" in stats
        assert stats["shell"]["calls"] == 1
        assert stats["shell"]["errors"] == 1
        assert stats["shell"]["total_ms"] == 100.0

    def test_tool_stats_empty(self):
        stream = EventStream()
        assert stream.tool_stats() == {}


# ---- Integration with AgentRunner ----

class TestEventStreamIntegration:
    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_events_emitted_on_simple_run(self, mock_cls):
        """A simple run should emit run_start, turn_start, api_request, run_complete."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Hello!")], "end_turn", 100, 50,
        )

        stream = EventStream()
        runner = _make_runner(event_stream=stream)
        result = runner.run("Hi")

        event_types = [e.event_type for e in stream.events]
        assert "run_start" in event_types
        assert "turn_start" in event_types
        assert "api_request" in event_types
        assert "run_complete" in event_types

        # All events should share the same run_id
        run_ids = {e.run_id for e in stream.events}
        assert len(run_ids) == 1
        assert len(list(run_ids)[0]) == 12

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_tool_call_events_emitted(self, mock_cls):
        """A run with tool calls should emit tool_call events."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block("echo", {"msg": "test"})], "tool_use", 200, 80),
            _api_response([_text_block("Done")], "end_turn", 300, 40),
        ]

        stream = EventStream()
        runner = _make_runner(event_stream=stream)
        result = runner.run("Echo test")

        event_types = [e.event_type for e in stream.events]
        assert "tool_call" in event_types

        tool_events = [e for e in stream.events if e.event_type == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0].data["tool"] == "echo"
        assert "duration_ms" in tool_events[0].data

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_error_events_on_tool_failure(self, mock_cls):
        """When a tool raises an error, an error event should be emitted."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            _api_response([_tool_use_block("bad_tool", {})], "tool_use", 100, 50),
            _api_response([_text_block("Recovered")], "end_turn", 100, 50),
        ]

        stream = EventStream()
        # Register a tool that does NOT exist (bad_tool not in registry)
        runner = _make_runner(event_stream=stream)
        result = runner.run("Call bad tool")

        # The echo tool IS registered but bad_tool is not, so error event should fire
        tool_events = [e for e in stream.events if e.event_type == "tool_call"]
        assert len(tool_events) == 1
        # The tool_call event should have error info since the tool is unknown
        assert tool_events[0].data.get("error") is not None

        error_events = [e for e in stream.events if e.event_type == "error"]
        assert len(error_events) == 1

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_summary_after_run(self, mock_cls):
        """Summary should reflect actual token usage after a run."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Hi")], "end_turn", 150, 30,
        )

        stream = EventStream()
        runner = _make_runner(event_stream=stream)
        runner.run("Hello")

        s = stream.summary()
        assert s["total_input_tokens"] == 150
        assert s["total_output_tokens"] == 30
        assert s["duration_seconds"] > 0

    @patch("agent_runner.runner.anthropic.Anthropic")
    def test_jsonl_after_run(self, mock_cls):
        """JSONL export should produce valid JSON on each line after a run."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _api_response(
            [_text_block("Hi")], "end_turn", 100, 50,
        )

        stream = EventStream()
        runner = _make_runner(event_stream=stream)
        runner.run("Hello")

        jsonl = stream.to_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) >= 3  # at least run_start, turn_start, api_request, run_complete

        for line in lines:
            parsed = json.loads(line)
            assert "event_type" in parsed
            assert "run_id" in parsed
